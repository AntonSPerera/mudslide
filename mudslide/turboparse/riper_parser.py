#!/usr/bin/env python
"""Parser for RIPER (periodic DFT) module output in Turbomole.

Handles periodic-system-specific data: lattice vectors, k-point meshes,
two basis sets (SCF + auxiliary RI-J), final energies, Fermi level
statistics, and the stress tensor.
"""
from __future__ import annotations

from typing import Any

from .section_parser import ParseSection
from .line_parser import BooleanLineParser, SimpleLineParser
from .common_parser import BasisParser, VarLineParser
from .stack_iterator import StackIterator

# Component order as printed by RIPER for the raw stress tensor
_STRESS_COMPONENTS = ["xx", "xy", "yy", "xz", "yz", "zz", "yx", "zx", "zy"]


class RIPERBasisParser(BasisParser):
    """Parse both SCF and auxiliary RI-J basis sets (multi=True gives a list)."""

    def __init__(self) -> None:
        super().__init__()
        self.multi = True


class PeriodicInfoParser(ParseSection):
    """Parse the 'Periodic system found' section.

    Captures: dimensionality (ndim), cell parameters (a, b, gamma),
    and direct/reciprocal space cell vectors.
    """
    name = "cell"

    def __init__(self) -> None:
        super().__init__(r"Periodic system found", r"Fractional crystal coordinates")
        self.parsers = [
            SimpleLineParser(
                r"Periodicity in (\d+) dimensions",
                ["ndim"],
                types=[int],
            ),
            # Cell parameter magnitudes: |a|  |b|  gamma(deg)
            SimpleLineParser(
                r"^\s*(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s*$",
                ["a", "b", "gamma"],
                types=[float, float, float],
            ),
            # Both direct and reciprocal vector lines share the same format:
            #   a/b/c   x   y   z
            # They are accumulated in order; clean() splits them by ndim.
            SimpleLineParser(
                r"^\s*([abc])\s+(-?\S+)\s+(-?\S+)\s+(-?\S+)\s*$",
                names=["label", "x", "y", "z"],
                types=[str, float, float, float],
                title="vectors",
                multi=True,
            ),
        ]

    def clean(self, liter: StackIterator, out: dict[str, Any]) -> None:
        ndim = out.get("ndim", 3)
        vectors = out.pop("vectors", [])
        out["direct"] = [[v["x"], v["y"], v["z"]] for v in vectors[:ndim]]
        out["reciprocal"] = [[v["x"], v["y"], v["z"]] for v in vectors[ndim:ndim * 2]]


class KPointDirParser(ParseSection):
    """Parse per-direction k-point data within the K-POINT MESH section.

    Captures: reciprocal vector label and components, number of k-points
    along the vector, and fractional coordinates.  multi=True since there
    is one block per reciprocal lattice direction.

    Note: uses 'Reciprocal space cell vector' (singular) as the head to
    avoid colliding with 'Reciprocal space cell vectors' (plural) in the
    lattice section parsed by PeriodicInfoParser.
    """
    name = "directions"

    def __init__(self) -> None:
        super().__init__(
            r"Reciprocal space cell vector \(au\):",  # singular - k-point section only
            r"^\s*$",                                  # blank line ends each direction block
            multi=True,
        )
        self.parsers = [
            SimpleLineParser(
                r"^\s*([abc])\s+(-?\S+)\s+(-?\S+)\s+(-?\S+)",
                names=["label", "x", "y", "z"],
                types=[str, float, float, float],
            ),
            SimpleLineParser(
                r"Number of k-points along this vector:\s*(\d+)",
                ["nkpoints"],
                types=[int],
            ),
            # Fractional coordinates span 1-2 lines with up to 6 values each.
            VarLineParser(
                r"^\s*(-?\d+\.\d+)(?:\s+(-?\d+\.\d+))?(?:\s+(-?\d+\.\d+))?"
                r"(?:\s+(-?\d+\.\d+))?(?:\s+(-?\d+\.\d+))?(?:\s+(-?\d+\.\d+))?\s*$",
                title="fractional_coords",
                vars_type=float,
            ),
        ]


class KPointMeshParser(ParseSection):
    """Parse the K-POINT MESH section: total/distinct k-point counts and per-direction data."""
    name = "kpoints"

    def __init__(self) -> None:
        super().__init__(r"K-POINT MESH", r"PBE functional")
        self.parsers = [
            SimpleLineParser(
                r"Total number of k points.*:\s*(\d+)",
                ["total"],
                types=[int],
            ),
            SimpleLineParser(
                r"Number of symmetry distinct k points.*:\s*(\d+)",
                ["distinct"],
                types=[int],
            ),
            KPointDirParser(),
        ]


class RIPEREnergyParser(ParseSection):
    """Parse the FINAL ENERGIES box (only the converged result, not per-iteration boxes)."""
    name = "energies"

    def __init__(self) -> None:
        # Tail is the last energy line; parsers run before tail check so it IS captured.
        super().__init__(r"FINAL ENERGIES", r"ENERGY \(sigma->0\)")
        self.parsers = [
            SimpleLineParser(r"\|\s*KINETIC ENERGY\s*=\s*(\S+)\s*\|", ["kinetic"], types=[float]),
            SimpleLineParser(r"\|\s*COULOMB ENERGY\s*=\s*(\S+)\s*\|", ["coulomb"], types=[float]),
            SimpleLineParser(
                r"\|\s*EXCH\. & CORR\. ENERGY\s*=\s*(\S+)\s*\|", ["xc"], types=[float]
            ),
            SimpleLineParser(r"\|\s*DFT-D3 ENERGY\s*=\s*(\S+)\s*\|", ["dftd3"], types=[float]),
            SimpleLineParser(r"\|\s*TOTAL ENERGY\s*=\s*(\S+)\s*\|", ["total"], types=[float]),
            SimpleLineParser(r"\|\s*T\*S\s*=\s*(\S+)\s*\|", ["ts"], types=[float]),
            SimpleLineParser(r"\|\s*FREE\s+ENERGY\s*=\s*(\S+)\s*\|", ["free"], types=[float]),
            SimpleLineParser(
                r"\|\s*ENERGY \(sigma->0\)\s*=\s*(\S+)\s*\|", ["sigma0"], types=[float]
            ),
        ]


class FermiLevelParser(ParseSection):
    """Parse the Fermi Level Statistics block."""
    name = "fermi"

    def __init__(self) -> None:
        super().__init__(r"Fermi Level Statistics", r"-{10,}")
        self.parsers = [
            SimpleLineParser(
                r"Lowest unoccupied band\s*=\s*(\S+)", ["lumo"], types=[float]
            ),
            SimpleLineParser(
                r"Highest occupied band\s*=\s*(\S+)", ["homo"], types=[float]
            ),
            SimpleLineParser(r"Band gap\s*=\s*(\S+)", ["band_gap"], types=[float]),
            SimpleLineParser(
                r"Band gap middle\s*=\s*(\S+)", ["band_gap_middle"], types=[float]
            ),
            SimpleLineParser(r"Fermi level\s*=\s*(\S+)", ["fermi_level"], types=[float]),
        ]


class StressTensorParser(ParseSection):
    """Parse the raw stress tensor (9 components, printed one per line).

    The order printed by RIPER is xx, xy, yy, xz, yz, zz, yx, zx, zy.
    After parsing, clean() builds a labeled dict under out["tensor"].
    """
    name = "stress"

    def __init__(self) -> None:
        super().__init__(r"stress tensor, raw:", r"-{20,}")
        self.parsers = [
            VarLineParser(
                r"^\s*(-?\d+\.\d+E[+-]\d+)\s*$",
                title="components",
                vars_type=float,
            ),
        ]

    def clean(self, liter: StackIterator, out: dict[str, Any]) -> None:
        comps = out.pop("components", [])
        out["tensor"] = dict(zip(_STRESS_COMPONENTS, comps))


class RIPERParser(ParseSection):
    """Top-level parser for the RIPER periodic DFT module output."""
    name = "riper"

    def __init__(self) -> None:
        super().__init__(
            r"riper\s*\(.*\)\s*:\s*TURBOMOLE",
            r"riper\s*:\s*all done",
        )
        self.parsers = [
            RIPERBasisParser(),
            PeriodicInfoParser(),
            KPointMeshParser(),
            BooleanLineParser(
                r"SCF converged within\s+\d+ cycles",
                r"a^",
                "converged",
            ),
            RIPEREnergyParser(),
            FermiLevelParser(),
            StressTensorParser(),
        ]

    def clean(self, liter: StackIterator, out: dict[str, Any]) -> None:
        try:
            out["energy"] = out["energies"]["total"]
        except KeyError:
            pass

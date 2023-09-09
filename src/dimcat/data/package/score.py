from __future__ import annotations

from dimcat.data.package import DimcatPackage
from dimcat.data.package.base import PathPackage


class ScorePathPackage(PathPackage):
    """A package containing resources that are (references to) scores."""

    pass


class MuseScorePackage(DimcatPackage):
    """A datapackage as created by the ms3 MuseScore parsing library. Contains TSV facets with the naming format
    ``<name>.<facet>[.tsv]``.
    """

    pass

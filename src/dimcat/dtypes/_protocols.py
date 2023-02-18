from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Collection,
    Dict,
    Iterator,
    Protocol,
    Sequence,
    Union,
    runtime_checkable,
)

from .base import PathLike, PieceID, TabularData, TypedSequence

if TYPE_CHECKING:
    from .facet import Available, FacetConfig, FacetName


@runtime_checkable
class PFacet(Protocol):
    """Protocol for all objects representing one data facet of one or several pieces."""

    def get_aspect(self, key: [str, Enum]) -> [TypedSequence, TabularData]:
        ...


@runtime_checkable
class PLoader(Protocol):
    def __init__(self, directory: Union[PathLike, Collection[PathLike]]):
        pass

    def iter_pieces(self) -> Iterator[PPiece]:
        ...


@runtime_checkable
class PNotesTable(Protocol):
    tpc: Sequence


@runtime_checkable
class PPiece(Protocol):
    piece_id: PieceID

    @abstractmethod
    def get_available_facets(self) -> Dict["FacetName", "Available"]:
        ...

    @abstractmethod
    def get_facet(self, facet=Union["FacetName", "FacetConfig"]) -> PFacet:
        ...

    @abstractmethod
    def is_facet_available(self, facet: "FacetConfig") -> "Available":
        ...

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass, replace
from typing import Dict, Optional, Protocol, Tuple, Union, runtime_checkable

import ms3
from dimcat.base import Data
from dimcat.data.facet import (
    Available,
    DefaultFacetConfig,
    Facet,
    FacetConfig,
    FacetIdentifiers,
    FacetName,
    PFacet,
    get_facet_class,
)
from dimcat.dtypes import Configuration, PieceID


@runtime_checkable
class PPiece(Protocol):
    piece_id: PieceID

    @abstractmethod
    def check_facet_availability(
        self, facet: Union[FacetName, Configuration]
    ) -> Available:
        ...

    @abstractmethod
    def get_available_facets(
        self, min_availability: Optional[Available] = None
    ) -> Dict[FacetName, Available]:
        ...

    @abstractmethod
    def get_facet(self, facet=Union[FacetName, Configuration]) -> PFacet:
        ...

    @abstractmethod
    def is_facet_available(
        self, facet: Union[FacetName, Configuration], min_availability: Available
    ) -> bool:
        ...


@dataclass(frozen=True)
class DcmlPiece(Data):
    piece_id: PieceID
    source_object: ms3.Piece
    extractable_facets: Tuple[FacetName] = FacetName.make_tuple(
        (
            "Measures",
            "Notes",
            "Rests",
            "NotesAndRests",
            "Labels",
            "Harmonies",
            "FormLabels",
            "Cadences",
            "Events",
            "Positions",
        )
    )

    @staticmethod
    def _internal_keyword2facet(keyword: str) -> FacetName:
        keyword2facet = {
            "measures": FacetName.Measures,
            "notes": FacetName.Notes,
            "rests": FacetName.Rests,
            "notes_and_rests": FacetName.NotesAndRests,
            "labels": FacetName.Labels,
            "expanded": FacetName.Harmonies,
            "form_labels": FacetName.FormLabels,
            "cadences": FacetName.Cadences,
            "events": FacetName.Events,
            "chords": FacetName.Positions,
        }
        facet = keyword2facet.get(keyword)
        if facet is None:
            raise KeyError(
                f"'{keyword}' is not a valid ms3 keyowrd. Expected one of {list(keyword2facet.keys())}"
            )
        return facet

    @staticmethod
    def _facet2internal_keyword(facet: FacetName) -> str:
        facet2keyword = {
            FacetName.Measures: "measures",
            FacetName.Notes: "notes",
            FacetName.Rests: "rests",
            FacetName.NotesAndRests: "notes_and_rests",
            FacetName.Labels: "labels",
            FacetName.Harmonies: "expanded",
            FacetName.FormLabels: "form_labels",
            FacetName.Cadences: "cadences",
            FacetName.Events: "events",
            FacetName.Positions: "chords",
        }
        keyword = facet2keyword.get(facet)
        if keyword is None:
            raise KeyError(
                f"'{facet}' is not a facet known to ms3. Expected one of {list(facet2keyword.keys())}"
            )
        return keyword

    def check_facet_availability(
        self, facet: Union[FacetName, Configuration]
    ) -> Available:
        config = self._facet_argument2config(facet)
        if config.dtype not in self.extractable_facets:
            return Available.NOT
        facet2availability = self.get_available_facets()
        availability = facet2availability.get(config.dtype)
        if availability:
            return availability
        return Available.EXTERNALLY

    def get_available_facets(
        self, min_availability: Optional[Available] = None
    ) -> Dict[FacetName, Available]:
        facet2available_files = self.source_object.get_files(
            facets="tsvs",
            view_name="default",
            parsed=True,
            unparsed=True,
            choose="all",
            flat=False,
            include_empty=False,
        )
        result = {
            self._internal_keyword2facet(key): Available.INDIVIDUALLY
            for key in facet2available_files.keys()
        }
        if min_availability is None:
            return result
        return {
            facet: avail for facet, avail in result.items() if avail >= min_availability
        }

    def get_facet(self, facet=Union[FacetName, Configuration]) -> Facet:
        config = self._facet_argument2config(facet)
        availability = self.check_facet_availability(config)
        if availability is Available.INDIVIDUALLY:
            keyword = self._facet2internal_keyword(config.dtype)
            file, facet_df = self.source_object.get_parsed(
                facet=keyword,
                view_name="default",
                unfold=config.unfold,
                interval_index=config.interval_index,
            )
            facet_class = get_facet_class(config.dtype)
            identifiers = FacetIdentifiers(
                piece_id=self.piece_id, file_path=file.full_path
            )
            facet = facet_class.from_config(
                df=facet_df, config=config, identifiers=identifiers
            )
            return facet
        # elif other cases
        else:
            raise ValueError(
                f"{self.piece_id}: {config.dtype} has availability {availability}"
            )

    def is_facet_available(
        self,
        facet: Union[FacetName, Configuration],
        min_availability: Available = Available.INDIVIDUALLY,
    ) -> bool:
        availability = self.check_facet_availability(facet=facet)
        return availability >= min_availability

    def _facet_argument2config(
        self, facet=Union[FacetName, FacetConfig]
    ) -> FacetConfig:
        if isinstance(facet, Configuration):
            config = FacetConfig.from_dataclass(facet)
            if isinstance(config.dtype, str):
                replace(config, dtype=FacetName(config.dtype))
        else:
            config = DefaultFacetConfig(dtype=FacetName(facet))
        return config


# assert isinstance(DcmlPiece, PPiece), "DcmlPiece does not correctly implement the PPiece protocol."

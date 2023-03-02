import logging
import os
import re
from collections import defaultdict
from dataclasses import replace
from typing import (
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Set,
    Type,
    Union,
    overload,
    runtime_checkable,
)

import ms3
import pandas as pd
from dimcat.data.facet import (
    Available,
    DefaultStackedFacetConfig,
    Facet,
    FacetConfig,
    FacetID,
    FacetName,
    StackedFacet,
    StackedFacetConfig,
    StackedFacetID,
    get_stacked_facet_class,
)
from dimcat.data.piece import DcmlPiece, DcmlPieceBySlicing, PPiece
from dimcat.dtypes import Configuration, PathLike, PieceID, PieceIndex
from dimcat.utils.constants import DCML_FACETS
from dimcat.utils.functions import resolve_dir

logger = logging.getLogger(__name__)


def facet_argument2config(facet=Union[FacetName, Configuration]) -> StackedFacetConfig:
    if isinstance(facet, Configuration):
        config = StackedFacetConfig.from_dataclass(facet)
        if isinstance(config.dtype, str):
            config = replace(config, dtype=FacetName(config.dtype))
    else:
        config = DefaultStackedFacetConfig(dtype=FacetName(facet))
    return config


@runtime_checkable
class PLoader(Protocol):
    def __init__(self, directory: Union[PathLike, Collection[PathLike]]):
        pass

    def iter_pieces(self) -> Iterator[PPiece]:
        ...


class DcmlLoader(PLoader):
    def __init__(
        self,
        directory: Optional[Union[PathLike, Collection[PathLike]]] = None,
        use_concatenated: bool = True,
        **kwargs,
    ):
        self.directories = []
        self.loader = ms3.Parse()
        self.use_concatenated = use_concatenated

        if isinstance(directory, str):
            directory = [directory]
        if directory is None:
            return
        for d in directory:
            self.add_dir(directory=d, **kwargs)

    def iter_pieces(self) -> Iterator[DcmlPiece]:
        for corpus_name, ms3_corpus in self.loader.iter_corpora():
            for fname, ms3_piece in ms3_corpus.iter_pieces():
                PID = PieceID(corpus_name, fname)
                yield DcmlPiece(piece_id=PID, source_object=ms3_piece)

    def set_loader(self, new_loader: ms3.Parse):
        self.loader = new_loader

    def add_dir(self, directory: PathLike, **kwargs):
        self.directories.append(resolve_dir(directory))
        self.loader.add_dir(directory=directory, **kwargs)


def str2camel_case(keyword: str) -> str:
    return "".join(comp.title() for comp in keyword.split("_"))


class StackedFacetLoader(PLoader):
    def __init__(
        self,
        directory: Union[PathLike, Collection[PathLike]],
        **kwargs,
    ):
        self.directory = resolve_dir(directory)
        self.id2facets: Dict[PieceID, Set[FacetName]] = defaultdict(set)
        self.facet2ids: Dict[FacetName, List[PieceID]] = defaultdict(list)
        self.facet2file_path: Dict[FacetName, str] = {}
        self.facet_store: Dict[FacetName, StackedFacet] = {}

        for f in os.listdir(self.directory):
            m = re.match(r"^concatenated_(.*)\.tsv$", f)
            if m is None:
                continue
            camel_case = str2camel_case(m.group(1))
            try:
                facet_name = FacetName(camel_case)
            except ValueError:
                logger.warning(
                    f"'{f}' concatenates {camel_case} which is not a recognized facet."
                )
                continue
            file_path = os.path.join(directory, f)
            self.facet2file_path[facet_name] = file_path
            idx = pd.read_csv(file_path, sep="\t", usecols=["corpus", "fname"])
            for corpus, fname in idx.groupby(["corpus", "fname"]).size().index:
                PID = PieceID(corpus, fname)
                self.id2facets[PID].add(facet_name)
                self.facet2ids[facet_name].append(PID)

    def check_facet_availability(
        self, facet: Union[FacetName, Configuration]
    ) -> Available:
        config = facet_argument2config(facet)
        facet_name = config.dtype
        if config.dtype not in DCML_FACETS:
            return Available.NOT
        facet2availability = self.get_available_facets()
        if facet_name in facet2availability:
            available_ids = self.facet2ids[facet_name]
            if hasattr(facet, "piece_id"):
                selected_ids = facet.piece_id
                selected_ids_available = set(selected_ids).intersection(
                    set(available_ids)
                )
                n_selected, n_available = len(selected_ids), len(selected_ids_available)
                if n_available == 0:
                    return Available.EXTERNALLY
                if n_available < n_selected:
                    return Available.PARTIALLY
                if available_ids > n_selected:
                    return Available.BY_SLICING
                # otherwise: n_available == n_selected == len(available_ids)
            return facet2availability[facet_name]
        return Available.EXTERNALLY

    def get_available_facets(
        self, min_availability: Optional[Available] = None
    ) -> Dict[FacetName, Available]:
        return {
            facet_name: Available.AVAILABLE
            for facet_name in self.facet2file_path.keys()
        }

    def _get_stored_facet(self, facet_name: FacetName) -> StackedFacet:
        if facet_name not in self.facet_store:
            file_path = self.facet2file_path[facet_name]
            stacked_df = ms3.load_tsv(file_path, index_col=[0, 1, 2])
            piece_index = PieceIndex(self.facet2ids[facet_name])
            facet_class = get_stacked_facet_class(facet_name)
            self.facet_store[facet_name] = facet_class.from_df(
                df=stacked_df, piece_index=piece_index, file_path=file_path
            )
        return self.facet_store[facet_name]

    @overload
    def get_facet(self, facet=FacetID) -> Facet:
        ...

    @overload
    def get_facet(
        self, facet=Union[FacetName, StackedFacetID, StackedFacetConfig, FacetConfig]
    ) -> StackedFacet:
        ...

    def get_facet(
        self, facet=Union[FacetName, Configuration]
    ) -> Union[StackedFacet, Facet]:
        config = facet_argument2config(facet)
        facet_name = config.dtype
        if isinstance(facet, StackedFacetID):
            available_ids = self.facet2ids[facet_name]
            selected_ids = set(facet.piece_index).intersection(set(available_ids))
            if len(selected_ids) == 0:
                raise KeyError(f"None of the requested piece IDs available: {facet}")
            stacked_facet = self._get_stored_facet(facet_name)
            if len(selected_ids) == len(available_ids):
                return stacked_facet
            raise NotImplementedError
        elif isinstance(facet, FacetID):
            if facet.piece_id in self.facet2ids[facet_name]:
                stacked_facet = self._get_stored_facet(facet_name)
                sliced_facet = stacked_facet.get_facet(facet.piece_id)
                return sliced_facet
            raise KeyError(f"Piece ID not available: {facet.piece_id}")
        else:
            raise NotImplementedError(f"Don't know how to deal with {facet}")

    def iter_pieces(self) -> Iterator[PPiece]:
        for piece_id, facets in self.id2facets.items():
            yield DcmlPieceBySlicing(
                piece_id=piece_id, source_object=self, available_facets=tuple(facets)
            )


assert isinstance(
    DcmlLoader, PLoader
), "DcmlLoader does not correctly implement the PLoader protocol."


def infer_data_loader(directory: str) -> Type[PLoader]:
    return DcmlLoader


if __name__ == "__main__":
    from dimcat.data.facet import FacetName, PNotesTable

    loader = StackedFacetLoader(directory="~/corelli")
    assert isinstance(loader, PLoader)
    for piece in loader.iter_pieces():
        assert isinstance(piece, PPiece)
        facet = piece.get_facet("Notes")
        assert isinstance(facet, PNotesTable)
        print(f"{piece.piece_id} yields a {type(facet)} with {len(facet)} notes")
    print("OK")

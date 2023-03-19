import logging
import os
import re
from collections import defaultdict
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
from dimcat.base import Configuration, PieceStackIdentifier, StackID
from dimcat.data.facet import (
    Available,
    Facet,
    FacetConfig,
    FacetID,
    FacetName,
    StackedFacet,
    facet_argument2config,
    get_stacked_facet_class,
)
from dimcat.data.piece import DcmlPiece, DcmlPieceBySlicing, PPiece
from dimcat.dtypes import PathLike, PieceID, PieceIndex
from dimcat.utils.constants import DCML_FACETS
from dimcat.utils.functions import resolve_dir

logger = logging.getLogger(__name__)


@runtime_checkable
class PLoader(Protocol):
    def __init__(self, directory: Union[PathLike, Collection[PathLike]]):
        pass

    def iter_pieces(self) -> Iterator[PPiece]:
        ...

    def get_facet(self, facet=Union[FacetName, Configuration]) -> Union[StackedFacet]:
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

    @overload
    def get_facet(self, facet=FacetID) -> Facet:
        ...

    @overload
    def get_facet(self, facet=Union[FacetName, StackID, FacetConfig]) -> StackedFacet:
        ...

    def get_facet(
        self, facet=Union[FacetName, Configuration]
    ) -> Union[StackedFacet, Facet]:
        config = facet_argument2config(facet)
        facet_name = config.dtype
        piece_index = None
        if isinstance(facet, StackID):
            piece_index = facet.piece_index
        elif isinstance(facet, FacetID):
            piece_index = PieceIndex([facet.piece_id])
        config2facet_objects: Dict[FacetConfig, Dict[PieceID, Facet]] = defaultdict(
            dict
        )
        for piece_obj in self.iter_pieces():
            if piece_index is not None and piece_obj.piece_id not in piece_index:
                continue
            try:
                facet_obj = piece_obj.get_facet(facet=facet)
            except Exception:
                logger.debug(f"{facet} not available for {piece_obj.piece_id}")
                continue
            facet_config = FacetConfig.from_dataclass(facet_obj)
            config2facet_objects[facet_config][facet_obj.piece_id] = facet_obj
        if len(config2facet_objects) > 1:
            raise NotImplementedError(
                f"Currently, facets with diverging configs cannot be concatenated:\n"
                f"{set(config2facet_objects.keys())}"
            )
        concatenated_per_config = []
        piece_ids = []
        for config, facet_objects in config2facet_objects.items():
            concatenated_per_config.append(
                config.concat_method(
                    facet_objects, names=["corpus", "piece", f"{facet_name}_i"]
                )
            )
            piece_ids.extend(facet_objects.keys())
        facet_constructor = get_stacked_facet_class(facet_name)
        if piece_index is None:
            piece_index = PieceIndex(piece_ids)
        identifier = PieceStackIdentifier(piece_index=piece_index)
        result = facet_constructor.from_df(
            df=concatenated_per_config[0], configuration=config, identifier=identifier
        )
        return result

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
            feature_name = m.group(1)
            try:
                facet_name = DcmlPiece._internal_keyword2facet(feature_name)
            except KeyError:
                logger.warning(
                    f"'{f}' concatenates {m.group(1)} which is not a recognized facet."
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
        file_path = self.facet2file_path[facet_name]
        if facet_name not in self.facet_store:
            logger.debug(f"Loading {file_path}...")
            stacked_df = ms3.load_tsv(file_path, index_col=[0, 1, 2])
            piece_index = PieceIndex(self.facet2ids[facet_name])
            identifier = PieceStackIdentifier(piece_index=piece_index)
            facet_class = get_stacked_facet_class(facet_name)
            self.facet_store[facet_name] = facet_class.from_default(
                df=stacked_df, identifier=identifier
            )
        else:
            logger.debug(f"Using previously loaded {file_path}...")
        return self.facet_store[facet_name]

    @overload
    def get_facet(self, facet=FacetID) -> Facet:
        ...

    @overload
    def get_facet(self, facet=Union[FacetName, StackID, FacetConfig]) -> StackedFacet:
        ...

    def get_facet(
        self, facet=Union[FacetName, Configuration]
    ) -> Union[StackedFacet, Facet]:
        config = facet_argument2config(facet)
        facet_name = config.dtype
        if isinstance(facet, StackID):
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
                sliced_facet = stacked_facet.get_piece(facet.piece_id)
                return sliced_facet
            raise KeyError(f"Piece ID not available: {facet.piece_id}")
        else:
            return self._get_stored_facet(facet_name)

    def iter_pieces(self) -> Iterator[PPiece]:
        for piece_id, facets in self.id2facets.items():
            yield DcmlPieceBySlicing(
                piece_id=piece_id, source_object=self, available_facets=tuple(facets)
            )


for loader_class in (DcmlLoader, StackedFacetLoader):
    assert issubclass(
        loader_class, PLoader
    ), f"{loader_class} does not correctly implement the PLoader protocol."


def infer_data_loader(directory: str) -> Type[PLoader]:
    return DcmlLoader


if __name__ == "__main__":
    from dimcat.data.facet import FacetName, PNotesTable

    loader = StackedFacetLoader(directory="~/corelli")
    assert isinstance(loader, PLoader)
    for piece in loader.iter_pieces():
        assert isinstance(piece, PPiece)
        print(
            f"{piece.name} with available facets {piece.get_available_facets()}, and "
            f".check_facet_availability('Markup') yielding {piece.check_facet_availability('Markup')!r}"
        )
        facet = piece.get_facet("Notes")
        assert isinstance(facet, PNotesTable)
        print(f"{piece.piece_id} yields a {type(facet)} with {len(facet)} notes")

    print("OK")

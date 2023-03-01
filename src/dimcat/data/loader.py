import os
import re
from collections import defaultdict
from pprint import pprint
from typing import (
    Collection,
    Iterator,
    Optional,
    Protocol,
    Type,
    Union,
    runtime_checkable,
)

import ms3
import pandas as pd
from dimcat.data.piece import DcmlPiece, PPiece
from dimcat.dtypes import PathLike, PieceID
from dimcat.utils.functions import resolve_dir


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


class ConcatenatedFacetLoader(PLoader):
    def __init__(
        self,
        directory: Optional[Union[PathLike, Collection[PathLike]]] = None,
        **kwargs,
    ):
        self.directories = []

        if isinstance(directory, str):
            directory = [directory]
        if directory is None:
            return
        for d in directory:
            self.add_dir(directory=d, **kwargs)

    def add_dir(self, directory: PathLike, **kwargs):
        self.directories.append(resolve_dir(directory))
        id2concatenated_facets = defaultdict(list)
        if self.use_concatenated:
            for directory in self.directories:
                for f in os.listdir(directory):
                    m = re.match(r"^concatenated_(.*)\.tsv$", f)
                    if m is None:
                        continue
                    facet = m.group(1)
                    file_path = os.path.join(directory, f)
                    idx = pd.read_csv(file_path, sep="\t", usecols=["corpus", "fname"])
                    for corpus, fname in idx.groupby(["corpus", "fname"]).size().index:
                        id2concatenated_facets[PieceID(corpus, fname)].append(facet)
        pprint(id2concatenated_facets)


assert isinstance(
    DcmlLoader, PLoader
), "DcmlLoader does not correctly implement the PLoader protocol."


def infer_data_loader(directory: str) -> Type[PLoader]:
    return DcmlLoader


if __name__ == "__main__":
    from dimcat.data.facet import PNotesTable

    loader = DcmlLoader(directory="~/dcml_corpora")
    assert isinstance(loader, PLoader)
    for piece in loader.iter_pieces():
        assert isinstance(piece, PPiece)
        facet = piece.get_facet("Notes")
        assert isinstance(facet, PNotesTable)
        print(f"{piece.piece_id} yields a PNotesTable with {len(facet)} notes")
    print("OK")

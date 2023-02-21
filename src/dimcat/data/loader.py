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
from dimcat.data.piece import DcmlPiece, PPiece
from dimcat.dtypes import PathLike, PieceID


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
        parse_scores: bool = False,
        parse_tsv: bool = True,
        **kwargs,
    ):
        self.parse_scores = parse_scores
        self.parse_tsv = parse_tsv
        self.directories = []
        self.loader = ms3.Parse()
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
        self.directories.append(directory)
        self.loader.add_dir(directory=directory, **kwargs)


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

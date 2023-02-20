from dataclasses import asdict, dataclass
from typing import Collection, Dict, Iterator, Optional, Tuple, Type, Union

import ms3
from dimcat.dtypes import Configuration, PathLike, PieceID, PLoader, PNotesTable, PPiece

from .facet import Available, Facet, FacetConfig, FacetName


@dataclass
class DcmlPiece(PPiece):
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

    def get_available_facets(self) -> Dict[FacetName, Available]:
        facet2available_files = self.source_object.get_files(
            facets="tsvs",
            view_name="default",
            parsed=True,
            unparsed=True,
            choose="all",
            flat=False,
            include_empty=False,
        )
        replacements = {
            "expanded": "harmonies",
            "chords": "positions",
        }
        available_facet_str = [
            key if key not in replacements else replacements[key]
            for key in facet2available_files.keys()
        ]
        return {
            FacetName(facet_str): Available.INDIVIDUALLY
            for facet_str in available_facet_str
        }

    def is_facet_available(self, facet: Union[FacetName, Configuration]) -> Available:
        config = self._facet_argument2config(facet)
        if config.facet not in self.extractable_facets:
            return Available.NOT
        facet2availability = self.get_available_facets()
        availability = facet2availability.get(config.facet)
        if availability:
            return availability
        return Available.EXTERNALLY

    def get_facet(self, facet=Union[FacetName, Configuration]) -> Facet:
        config = self._facet_argument2config(facet)
        availability = self.is_facet_available(config)
        if availability is Available.INDIVIDUALLY:
            file, facet_df = self.source_object.get_parsed(
                facet=config.facet.value,
                view_name="default",
                unfold=config.unfold,
                interval_index=config.interval_index,
            )
            facet_config_args = {
                k: v for k, v in asdict(config).items() if k != "facet"
            }
            file_identifier_args = {
                "piece_id": self.piece_id,
                "file_path": file.full_path,
            }
            facet = FacetName(config.facet).from_df(
                df=facet_df, **file_identifier_args, **facet_config_args
            )
            return facet

    def _facet_argument2config(
        self, facet=Union[FacetName, FacetConfig]
    ) -> FacetConfig:
        if isinstance(facet, Configuration):
            config = FacetConfig.from_dataclass(facet)
        else:
            config = FacetName(facet).default_config
        return config


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
    loader = DcmlLoader(directory="~/dcml_corpora")
    assert isinstance(loader, PLoader)
    for piece in loader.iter_pieces():
        assert isinstance(piece, PPiece)
        facet = piece.get_facet("notes")
        assert isinstance(facet, PNotesTable)
        print(f"{piece.piece_id} yields a PNotesTable with {len(facet)} notes")
    print("OK")

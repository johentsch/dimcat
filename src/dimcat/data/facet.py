from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from functools import cached_property, lru_cache, partial
from typing import Callable, Dict, Iterable, Sequence, Tuple, Type, Union

import pandas as pd
from dimcat.dtypes.base import Configuration, PieceID, TabularData, TypedSequence
from dimcat.dtypes.sequence import Bigrams, ContiguousSequence

# region Helpers


@lru_cache()
def _get_enum_constructor_by_name(name: str) -> Type[Enum]:
    """Allow dataclasses to use the Enums defined underneath them."""
    return globals()[name]


@lru_cache()
def _get_enum_member(name: str, value: str) -> Enum:
    """Allow dataclasses to use the Enums defined underneath them."""
    constructor = _get_enum_constructor_by_name(name)
    if constructor is None:
        raise KeyError(f"Can't find Enum of name '{name}'.")
    return constructor(value)


@lru_cache()
def _get_facet_constructor_by_name(name: str) -> Type[Facet]:
    """Allow dataclasses to use the Enums defined underneath them."""
    return globals()[name]


# endregion Helpers


# region Enums and Configs


class Aspect(str, Enum):
    GLOBALKEY = "globalkey"
    LOCALKEY = "localkey"
    TPC = "tpc"


class Available(IntEnum):
    """Expresses the availability of a requested facet for a given piece. Value 0 corresponds to never available.
    All following values have increasingly higher values following the logic "the higher the value, the cheaper to get".
    That enables checking for a minimal status, e.g. ``if availability > Available.BY_TRANSFORMING``.
    """

    NOT = 0
    EXTERNALLY = auto()
    BY_TRANSFORMING = auto()
    BY_SLICING = auto()
    INDIVIDUALLY = auto()


class DfType(str, Enum):
    PANDAS = "pandas"
    MODIN = "modin"


@dataclass(frozen=True)
class FacetConfig(Configuration):
    facet: FacetName
    df_type: DfType
    unfold: bool
    interval_index: bool
    concat_method: Callable[[Dict[PieceID, Facet], Sequence[str]], Facet]


@dataclass(frozen=True)
class DefaultFacetConfig(FacetConfig):
    facet: FacetName
    df_type: DfType = DfType.PANDAS
    unfold: bool = False
    interval_index: bool = True
    concat_method: Callable[[Dict[PieceID, Facet], Sequence[str]], Facet] = pd.concat


@dataclass(frozen=True)
class FacetIdentifier(FacetConfig):
    piece_id: PieceID
    file_path: str


FacetNamePromise = partial(_get_enum_member, "FacetName")
CadencePromise = partial(FacetNamePromise, "cadence")

# endregion Enums and Configs

# region facet tables


@dataclass(frozen=True)
class Facet(FacetIdentifier, TabularData):
    """Classes structurally implementing the PFacet protocol."""

    def get_aspect(self, key: Union[str, Enum]) -> ContiguousSequence:
        """In its basic form, get one of the columns as a :obj:`TypedSequence`.
        Subclasses may offer additional aspects, such as transformed columns or subsets of the table.
        """
        series: pd.Series = self.df[key]
        sequential_data = ContiguousSequence(series)
        return sequential_data

    pass


@dataclass(frozen=True)
class Cadences(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)
    pass


@dataclass(frozen=True)
class Events(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)
    pass


@dataclass(frozen=True)
class FormLabels(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)
    pass


@dataclass(frozen=True)
class Harmonies(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)

    @cached_property
    def globalkey(self) -> str:
        return self.get_aspect(key=Aspect.GLOBALKEY)[0]

    def get_localkey_bigrams(self) -> Bigrams:
        """Returns a TypedSequence of bigram tuples representing modulations between local keys."""
        localkey_list = self.get_aspect(key=Aspect.LOCALKEY).get_changes()
        return localkey_list.get_n_grams(n=2)

    def get_chord_bigrams(self) -> Bigrams:
        chords = self.get_aspect("chord")
        return chords.get_n_grams(2)


@dataclass(frozen=True)
class Labels(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)
    pass


@dataclass(frozen=True)
class Measures(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)
    pass


@dataclass(frozen=True)
class Notes(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)

    @cached_property
    def tpc(self) -> TypedSequence:
        series = self.get_aspect(Aspect.TPC)
        return TypedSequence(series)


@dataclass(frozen=True)
class NotesAndRests(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)
    pass


@dataclass(frozen=True)
class Positions(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)
    pass


@dataclass(frozen=True)
class Rests(Facet):
    facet: FacetName = field(init=False, default=CadencePromise)
    pass


# endregion facet tables


class FacetName(str, Enum):
    """Identifies the various types of data facets and makes accessible their default configs and TabularData."""

    default_config: FacetConfig
    """Attribute of each enum member to retrieve the respective default configuration."""

    def __new__(cls, name: str, class_name: str, docstring: str = "") -> FacetName:
        """Add attributes to the Enum members, namely defaultdict, make_config, and from_df.
        Credits for this solution go to
        https://rednafi.github.io/reflections/add-additional-attributes-to-enum-members-in-python.html
        """
        obj = str.__new__(cls, name)
        obj._value_ = name
        obj.default_config = DefaultFacetConfig(facet=obj)
        obj.make_config = partial(FacetConfig, facet=obj)
        facet_class = _get_facet_constructor_by_name(class_name)
        obj.from_df = facet_class.from_df
        obj.__doc__ = docstring
        return obj

    @classmethod
    def make_tuple(cls, facets: Iterable[Union[FacetName, str]]) -> Tuple[FacetName]:
        return tuple(cls(c) for c in facets)

    # NAME = ("value", "FacetClass", "docstring")
    MEASURES = "measures", "Measures", "Measure map."
    NOTES = "notes", "Notes", "Note table"
    RESTS = "rests", "Rests"
    NOTES_AND_RESTS = "notes_and_rests", "NotesAndRests"
    LABELS = "labels", "Labels"
    HARMONIES = (
        "harmonies",
        "Harmonies",
        "Harmony annotations divided into feature columns.",
    )
    FORM_LABELS = "form_labels", "FormLabels"
    CADENCES = "cadences", "Cadences"
    EVENTS = (
        "events",
        "Events",
        "One large table representing all available events in their raw form.",
    )
    POSITIONS = (
        "positions",
        "Positions",
        "All positions which have a note or some markup attached, and markup such as "
        "dynamics, articulation, fermatas, lyrics, bass figures, pedaling, etc.",
    )


if __name__ == "__main__":
    import ms3

    file_path = "~/corelli/harmonies/op01n01a.tsv"
    harmonies = Harmonies(
        df=ms3.load_tsv(file_path),
        df_type=DfType.PANDAS,
        unfold=False,
        interval_index=False,
        concat_method=pd.concat,
        piece_id=PieceID("corelli", "op01n01a"),
        file_path=file_path,
    )
    print(
        f"Modulations in the globalkey {harmonies.globalkey}: {harmonies.get_localkey_bigrams()}"
    )

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum, auto
from functools import cached_property
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import pandas as pd
from dimcat.dtypes.base import (
    Configuration,
    DataBackend,
    DataframeType,
    PieceID,
    SeriesType,
    TabularData,
    TypedSequence,
)
from dimcat.dtypes.sequence import Bigrams, ContiguousSequence
from dimcat.utils.decorators import config_dataclass
from typing_extensions import Self

logger = logging.getLogger(__name__)
# region Enums and Configs


class FacetName(str, Enum):
    """Identifies the various types of data facets and makes accessible their default configs and TabularData."""

    @classmethod
    def make_tuple(cls, facets: Iterable[Union[FacetName, str]]) -> Tuple[FacetName]:
        return tuple(cls(c) for c in facets)

    Measures = "Measures"
    Notes = "Notes"
    Rests = "Rests"
    NotesAndRests = "NotesAndRests"
    Labels = "Labels"
    Harmonies = "Harmonies"
    FormLabels = "FormLabels"
    Cadences = "Cadences"
    Events = "Events"
    Positions = "Positions"


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


@dataclass(frozen=True)
class FacetConfig(Configuration):
    dtype: FacetName
    df_type: DataBackend
    unfold: bool
    interval_index: bool
    concat_method: Callable[[Dict[PieceID, Facet], Sequence[str]], Facet]


@dataclass(frozen=True)
class DefaultFacetConfig(FacetConfig):
    """Configuration for any facet."""

    dtype: FacetName
    df_type: DataBackend = DataBackend.PANDAS
    unfold: bool = False
    interval_index: bool = True
    concat_method: Callable[[Dict[PieceID, Facet], Sequence[str]], Facet] = pd.concat


@dataclass(frozen=True)
class FacetIdentifiers(Configuration):
    """Fields serving to identify the facet of one particular piece."""

    piece_id: PieceID
    file_path: str


@config_dataclass(dtype=FacetName, df_type=DataBackend)
class FacetID(FacetConfig, FacetIdentifiers):
    """Config + Identifier"""

    pass


# endregion Enums and Configs

# region Facet types


@config_dataclass(dtype=FacetName, df_type=DataBackend)
class Facet(TabularData, FacetID):
    """Classes structurally implementing the PFacet protocol."""

    def get_aspect(self, key: Union[str, Enum]) -> ContiguousSequence:
        """In its basic form, get one of the columns as a :obj:`TypedSequence`.
        Subclasses may offer additional aspects, such as transformed columns or subsets of the table.
        """
        series: SeriesType = self.df[key]
        sequential_data = ContiguousSequence(series)
        return sequential_data

    @classmethod
    @property
    def dtype(cls) -> FacetName:
        """Name of the class as enum member."""
        return FacetName(cls.name)

    @classmethod
    def from_config(
        cls,
        df: DataframeType,
        config: FacetConfig,
        identifiers: Optional[FacetIdentifiers] = None,
        **kwargs,
    ) -> Self:
        """Create a Facet from a dataframe and a :obj:`FacetConfig`. The required identifiers can be given either
        as :obj:`FacetIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        cfg_kwargs = FacetConfig.dict_from_dataclass(config)
        if identifiers is not None:
            id_kwargs = FacetIdentifiers.dict_from_dataclass(identifiers)
            cfg_kwargs.update(id_kwargs)
        cfg_kwargs.update(kwargs)
        if cfg_kwargs["dtype"] != cls.dtype:
            cfg_class = config.__class__.__name__
            raise TypeError(
                f"Cannot initiate {cls.name} with {cfg_class}.dtype={config.dtype}."
            )
        return cls(df=df, **cfg_kwargs)

    @classmethod
    def from_df(
        cls,
        df: DataframeType,
        identifiers: Optional[FacetIdentifiers] = None,
        **kwargs,
    ) -> Self:
        """Create a Facet from a dataframe and a :obj:`FacetConfig`. The required identifiers can be given either
        as :obj:`FacetIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        config = cls.get_default_config()
        return cls.from_config(df=df, config=config, identifiers=identifiers, **kwargs)

    @classmethod
    def from_id(cls, df: DataframeType, facet_id: FacetID):
        kwargs = asdict(FacetID.from_dataclass(facet_id))
        return cls(df=df, **kwargs)

    @classmethod
    def get_default_config(cls) -> DefaultFacetConfig:
        return DefaultFacetConfig(dtype=cls.dtype)

    def get_identifier(self) -> FacetID:
        return FacetID(self)


class Cadences(Facet):
    pass


@dataclass(frozen=True)
class Events(Facet):
    pass


@dataclass(frozen=True)
class FormLabels(Facet):
    pass


@dataclass(frozen=True)
class Harmonies(Facet):
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
    pass


@dataclass(frozen=True)
class Measures(Facet):
    pass


@dataclass(frozen=True)
class Notes(Facet):
    @cached_property
    def tpc(self) -> TypedSequence:
        series = self.get_aspect(Aspect.TPC)
        return TypedSequence(series)


@dataclass(frozen=True)
class NotesAndRests(Facet):
    pass


@dataclass(frozen=True)
class Positions(Facet):
    pass


@dataclass(frozen=True)
class Rests(Facet):
    pass


# endregion Facet types


if __name__ == "__main__":
    print(DefaultFacetConfig(dtype="Notes"))
    import ms3

    file_path = "~/corelli/harmonies/op01n01a.tsv"
    df = ms3.load_tsv(file_path)
    harmonies1 = Harmonies(
        dtype="Harmonies",
        df=df,
        df_type=DataBackend.PANDAS,
        unfold=False,
        interval_index=True,
        concat_method=pd.concat,
        piece_id=PieceID("corelli", "op01n01a"),
        file_path=file_path,
    )
    f_cfg = Harmonies.get_default_config()
    id_dict = dict(
        piece_id=PieceID("corelli", "op01n01a"),
        file_path=file_path,
    )
    f_id = FacetIdentifiers(**id_dict)
    harmonies2 = Harmonies.from_config(df=df, config=f_cfg, identifiers=f_id)
    assert harmonies1 == harmonies2
    harmonies3 = Harmonies.from_id(df, facet_id=harmonies2)
    assert harmonies2 == harmonies3
    harmonies4 = Harmonies.from_df(df=df, **id_dict)
    harmonies3 == harmonies4
    print(
        f"Modulations in the globalkey {harmonies2.globalkey}: {harmonies2.get_localkey_bigrams()}"
    )
    # c = Cadences()

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, IntEnum, auto
from functools import cached_property, lru_cache
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import pandas as pd
from dimcat.dtypes.base import (
    Configuration,
    ConfiguredDataframe,
    DataBackend,
    PieceID,
    SomeDataframe,
    SomeFeature,
    SomeSeries,
    TypedSequence,
    WrappedSeries,
)
from dimcat.dtypes.sequence import Bigrams, ContiguousSequence
from dimcat.utils.decorators import config_dataclass
from dimcat.utils.functions import get_value_profile_mask
from typing_extensions import Self

logger = logging.getLogger(__name__)


@runtime_checkable
class PFacet(Protocol):
    """Protocol for all objects representing one data facet of one or several pieces."""

    def get_feature(self, feature: [str, Enum]) -> SomeFeature:
        ...


@runtime_checkable
class PNotesTable(Protocol):
    tpc: Sequence


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


class FeatureName(str, Enum):
    GLOBALKEY = "globalkey"
    LOCALKEY = "localkey"
    TPC = "tpc"


class Available(IntEnum):
    """Expresses the availability of a requested facet for a given piece. Value 0 corresponds to never available.
    All following values have increasingly higher values following the logic "the higher the value, the cheaper to get".
    That enables checking for a minimal status, e.g. ``if availability > Available.BY_TRANSFORMING``. It implies
    that every availability includes all higher availabilities.
    """

    NOT = 0
    EXTERNALLY = (
        auto()
    )  # means: theoretically available but unable to verify; external check required
    BY_TRANSFORMING = auto()
    BY_SLICING = auto()
    INDIVIDUALLY = auto()


@dataclass(frozen=True)
class FeatureConfig(Configuration):
    dtype: FeatureName
    df_type: DataBackend
    unfold: bool
    interval_index: bool
    concat_method: Callable[
        [Dict[PieceID, TabularFeature], Sequence[str]], TabularFeature
    ]


@dataclass(frozen=True)
class DefaultFeatureConfig(FeatureConfig):
    """Configuration for any facet."""

    dtype: FeatureName
    df_type: DataBackend = DataBackend.PANDAS
    unfold: bool = False
    interval_index: bool = True
    concat_method: Callable[
        [Dict[PieceID, TabularFeature], Sequence[str]], TabularFeature
    ] = pd.concat


@dataclass(frozen=True)
class FeatureIdentifiers(Configuration):
    """Fields serving to identify the facet of one particular piece."""

    piece_id: PieceID
    file_path: str


@config_dataclass(dtype=FeatureName, df_type=DataBackend)
class FeatureID(FeatureIdentifiers, FeatureConfig):
    """Config + Identifier"""

    pass


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

# region Features


@dataclass(frozen=True)
class TabularFeature(FeatureID, ConfiguredDataframe):
    _config_type: ClassVar[Type[FeatureConfig]] = FeatureConfig
    _default_config_type: ClassVar[Type[DefaultFeatureConfig]] = DefaultFeatureConfig
    _id_config_type: ClassVar[Type[FeatureIdentifiers]] = FeatureIdentifiers
    _id_type: ClassVar[Type[FeatureID]] = FeatureID
    _enum_type: ClassVar[Type[Enum]] = FeatureName

    @classmethod
    def from_df(
        cls,
        df: SomeDataframe,
        identifiers: Optional[FeatureIdentifiers] = None,
        **kwargs,
    ) -> Self:
        """Create a Feature from a dataframe and a :obj:`Configuration`. The required identifiers can be given either
        as :obj:`FeatureIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        return cls.from_default(df=df, identifiers=identifiers, **kwargs)


# endregion Features

# region Facets


@config_dataclass(dtype=FacetName, df_type=DataBackend)
class Facet(FacetID, ConfiguredDataframe):
    """Classes structurally implementing the PFacet protocol."""

    _config_type: ClassVar[Type[FacetConfig]] = FacetConfig
    _default_config_type: ClassVar[Type[DefaultFacetConfig]] = DefaultFacetConfig
    _id_config_type: ClassVar[Type[FacetIdentifiers]] = FacetIdentifiers
    _id_type: ClassVar[Type[FacetID]] = FacetID
    _enum_type: ClassVar[Type[Enum]] = FacetName

    # region Default methods repeated for type hints

    @property
    def config(self) -> FacetConfig:
        return self._config_type.from_dataclass(self)

    @property
    def identifier(self) -> FacetID:
        return self._id_type.from_dataclass(self)

    @classmethod
    def from_df(
        cls,
        df: SomeDataframe,
        identifiers: Optional[FacetIdentifiers] = None,
        **kwargs,
    ) -> Self:
        """Create a Facet from a dataframe and a :obj:`Configuration`. The required identifiers can be given either
        as :obj:`FacetIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        return cls.from_default(df=df, identifiers=identifiers, **kwargs)

    @classmethod
    def get_default_config(cls, **kwargs) -> DefaultFacetConfig:
        kwargs["dtype"] = cls.dtype
        return cls._default_config_type(**kwargs)

    # endregion Default methods repeated for type hints

    def get_feature(self, feature: Union[str, Enum]) -> WrappedSeries:
        """In its basic form, get one of the columns as a :obj:`WrappedSeries`.
        Subclasses may offer additional features, such as transformed columns or subsets of the table.
        """
        series: SomeSeries = self.df[feature]
        return WrappedSeries(series)


@dataclass(frozen=True)
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
        return self.get_feature(feature=FeatureName.GLOBALKEY)[0]

    def get_localkey_bigrams(self) -> Bigrams:
        """Returns a TypedSequence of bigram tuples representing modulations between local keys."""
        localkey_list = ContiguousSequence(
            self.get_feature(feature=FeatureName.LOCALKEY)
        ).get_changes()
        return localkey_list.get_n_grams(n=2)

    def get_chord_bigrams(self) -> Bigrams:
        chords = self.get_feature("chord")
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
        series = self.get_feature(FeatureName.TPC)
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


# endregion Facets


@lru_cache()
def get_facet_class(name: [FacetName, str]) -> Type[Facet]:
    try:
        facet_name = FacetName(name)
    except ValueError:
        raise ValueError(f"'{name}' is not a valid FacetName.")
    name2facet = {
        FacetName.Measures: Measures,
        FacetName.Notes: Notes,
        FacetName.Rests: Rests,
        FacetName.NotesAndRests: NotesAndRests,
        FacetName.Labels: Labels,
        FacetName.Harmonies: Harmonies,
        FacetName.FormLabels: FormLabels,
        FacetName.Cadences: Cadences,
        FacetName.Events: Events,
        FacetName.Positions: Positions,
    }
    return name2facet.get(facet_name)


if __name__ == "__main__":
    print(DefaultFacetConfig(dtype="Notes"))
    import ms3

    file_path = "~/corelli/harmonies/op01n01a.tsv"
    df = ms3.load_tsv(file_path)
    harmonies1 = Harmonies.from_default(
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
    harmonies3 = Harmonies.from_id(df=df, identifier=harmonies2)
    assert harmonies2 == harmonies3
    harmonies4 = Harmonies.from_df(df=df, **id_dict)
    assert harmonies3 == harmonies4
    chords_as_sequence = ContiguousSequence(
        harmonies2.get_feature(FeatureName.LOCALKEY)
    )
    chords_as_wrapped_series = harmonies1.get_feature(FeatureName.LOCALKEY).series
    value_profile_mask = get_value_profile_mask(
        chords_as_wrapped_series, na_values="ffill"
    )
    profile_by_masking = chords_as_wrapped_series[value_profile_mask]
    profile_from_sequence = chords_as_sequence.get_changes()
    print(profile_by_masking, profile_from_sequence)
    print(
        f"Modulations in the globalkey {harmonies2.globalkey}: {harmonies2.get_localkey_bigrams()}"
    )
    # c = Cadences()

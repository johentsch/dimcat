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
from dimcat.dtypes.sequence import Bigrams, ContiguousSequence, PieceIndex
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
    StackedMeasures = "StackedMeasures"
    StackedNotes = "StackedNotes"
    StackedRests = "StackedRests"
    StackedNotesAndRests = "StackedNotesAndRests"
    StackedLabels = "StackedLabels"
    StackedHarmonies = "StackedHarmonies"
    StackedFormLabels = "StackedFormLabels"
    StackedCadences = "StackedCadences"
    StackedEvents = "StackedEvents"
    StackedPositions = "StackedPositions"


class FeatureName(str, Enum):
    facet: FacetName

    def __new__(cls, name: str, facet: FacetName, description: str = "") -> FeatureName:
        """Adds attributes to Enum members. Thanks for this solution to Redowan Nafi via
        https://rednafi.github.io/reflections/add-additional-attributes-to-enum-members-in-python.html
        """
        obj = str.__new__(cls, name)
        obj._value_ = name

        obj.facet = facet
        obj.description = description
        return obj

    GLOBALKEY = ("globalkey", FacetName.Harmonies)
    LOCALKEY = ("localkey", FacetName.Harmonies)
    TPC = ("tpc", FacetName.Notes)
    CUSTOM = ("TabularFeature", None)

    @classmethod
    def _missing_(cls, value):
        return FeatureName.CUSTOM


def str2feature_name(name: Union[str, FeatureName]) -> FeatureName:
    try:
        feature_name = FeatureName(name)
    except ValueError:
        normalized_name = name.lower().strip("_")
        try:
            feature_name = FeatureName(normalized_name)
        except ValueError:
            raise ValueError(f"'{name}' is not a valid FeatureName.")
    return feature_name


class FeatureType(str, Enum):
    TabularFeature = "TabularFeature"


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
    PARTIALLY = auto()
    AVAILABLE = auto()


@dataclass(frozen=True)
class FeatureConfig(Configuration):
    feature_name: FeatureName
    dtype: FeatureType
    df_type: DataBackend
    unfold: bool
    interval_index: bool
    concat_method: Callable[
        [Dict[PieceID, TabularFeature], Sequence[str]], TabularFeature
    ]


@dataclass(frozen=True)
class DefaultFeatureConfig(FeatureConfig):
    """Configuration for any facet."""

    feature_name: FeatureName
    dtype: FeatureType = FeatureType.TabularFeature
    df_type: DataBackend = DataBackend.PANDAS
    unfold: bool = False
    interval_index: bool = True
    concat_method: Callable[
        [Dict[PieceID, TabularFeature], Sequence[str]], TabularFeature
    ] = pd.concat


@dataclass(frozen=True)
class FeatureIdentifiers(Configuration):
    """Fields serving to identify the facet of one particular piece."""

    pass
    # piece_id: PieceID
    # file_path: str


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


@dataclass(frozen=True)
class StackedFacetConfig(Configuration):
    dtype: FacetName
    df_type: DataBackend
    unfold: bool
    interval_index: bool
    concat_method: Callable[[Dict[PieceID, Facet], Sequence[str]], Facet]


@dataclass(frozen=True)
class DefaultStackedFacetConfig(StackedFacetConfig):
    """Configuration for any facet."""

    dtype: FacetName
    df_type: DataBackend = DataBackend.PANDAS
    unfold: bool = False
    interval_index: bool = True
    concat_method: Callable[[Dict[PieceID, Facet], Sequence[str]], Facet] = pd.concat


@dataclass(frozen=True)
class StackedFacetIdentifiers(Configuration):
    """Fields serving to identify the stacked facets of a set of pieces."""

    piece_index: PieceIndex
    file_path: str


@config_dataclass(dtype=FacetName, df_type=DataBackend)
class StackedFacetID(StackedFacetConfig, StackedFacetIdentifiers):
    """Config + Identifier"""

    pass


def feature_config2facet_config(feature: Configuration):
    feature_config = FeatureConfig.from_dataclass(feature)
    facet_name = feature_config.dtype.facet
    stacked_facet_class = get_stacked_facet_class(facet_name)
    return stacked_facet_class.get_default_config()


# endregion Enums and Configs

# region Features


@dataclass(frozen=True)
class TabularFeature(FeatureID, ConfiguredDataframe):
    _config_type: ClassVar[Type[FeatureConfig]] = FeatureConfig
    _default_config_type: ClassVar[Type[DefaultFeatureConfig]] = DefaultFeatureConfig
    _id_config_type: ClassVar[Type[FeatureIdentifiers]] = FeatureIdentifiers
    _id_type: ClassVar[Type[FeatureID]] = FeatureID
    _enum_type: ClassVar[Type[Enum]] = FeatureType

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


def str2facet_name(name: Union[FacetName, str]) -> FacetName:
    s2f = {
        "measures": FacetName.Measures,
        "notes": FacetName.Notes,
        "rests": FacetName.Rests,
        "notesandrests": FacetName.NotesAndRests,
        "labels": FacetName.Labels,
        "harmonies": FacetName.Harmonies,
        "formlabels": FacetName.FormLabels,
        "cadences": FacetName.Cadences,
        "events": FacetName.Events,
        "positions": FacetName.Positions,
        "stackedmeasures": FacetName.StackedMeasures,
        "stackednotes": FacetName.StackedNotes,
        "stackedrests": FacetName.StackedRests,
        "stackednotesandrests": FacetName.StackedNotesAndRests,
        "stackedlabels": FacetName.StackedLabels,
        "stackedharmonies": FacetName.StackedHarmonies,
        "stackedformlabels": FacetName.StackedFormLabels,
        "stackedcadences": FacetName.StackedCadences,
        "stackedevents": FacetName.StackedEvents,
        "stackedpositions": FacetName.StackedPositions,
    }
    try:
        facet_name = FacetName(name)
    except ValueError:
        normalized_name = name.lower().strip("_")
        if normalized_name in s2f:
            facet_name = s2f[normalized_name]
        else:
            raise ValueError(f"'{name}' is not a valid FacetName.")
    return facet_name


@lru_cache()
def get_facet_class(name: Union[FacetName, str]) -> Type[Facet]:
    facet_name = str2facet_name(name)
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
        FacetName.StackedMeasures: Measures,
        FacetName.StackedNotes: Notes,
        FacetName.StackedRests: Rests,
        FacetName.StackedNotesAndRests: NotesAndRests,
        FacetName.StackedLabels: Labels,
        FacetName.StackedHarmonies: Harmonies,
        FacetName.StackedFormLabels: FormLabels,
        FacetName.StackedCadences: Cadences,
        FacetName.StackedEvents: Events,
        FacetName.StackedPositions: Positions,
    }
    return name2facet.get(facet_name)


# endregion Facets

# region StackedFacet


@config_dataclass(dtype=FacetName, df_type=DataBackend)
class StackedFacet(StackedFacetID, ConfiguredDataframe):
    """Several facets stacked. De facto they are concatenated but the MultiIndex enables extracting them
    individually.
    """

    _config_type: ClassVar[Type[StackedFacetConfig]] = StackedFacetConfig
    _default_config_type: ClassVar[
        Type[DefaultStackedFacetConfig]
    ] = DefaultStackedFacetConfig
    _id_config_type: ClassVar[Type[StackedFacetIdentifiers]] = StackedFacetIdentifiers
    _id_type: ClassVar[Type[StackedFacetID]] = StackedFacetID
    _enum_type: ClassVar[Type[Enum]] = FacetName

    # region Default methods repeated for type hints

    @property
    def config(self) -> StackedFacetConfig:
        return self._config_type.from_dataclass(self)

    @property
    def identifier(self) -> StackedFacetID:
        return self._id_type.from_dataclass(self)

    @classmethod
    def from_df(
        cls,
        df: SomeDataframe,
        identifiers: Optional[StackedFacetIdentifiers] = None,
        **kwargs,
    ) -> Self:
        """Create a Facet from a dataframe and a :obj:`Configuration`. The required identifiers can be given either
        as :obj:`FacetIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        return cls.from_default(df=df, identifiers=identifiers, **kwargs)

    @classmethod
    def get_default_config(cls, **kwargs) -> DefaultStackedFacetConfig:
        kwargs["dtype"] = cls.dtype
        return cls._default_config_type(**kwargs)

    # endregion Default methods repeated for type hints

    def get_feature(self, feature: Union[FeatureName, FeatureConfig]) -> TabularFeature:
        """In its basic form, get one of the columns as a :obj:`WrappedSeries`.
        Subclasses may offer additional features, such as transformed columns or subsets of the table.
        """
        if isinstance(feature, Configuration):
            if isinstance(feature, FeatureID):
                raise NotImplementedError("Not accepting IDs as of now, only configs.")
            feature_config = FeatureConfig.from_dataclass(feature)
            feature_name = feature_config.feature_name
        else:
            feature_name = str2feature_name(feature)
            feature_config = DefaultFeatureConfig(dtype=feature_name)
        selected_columns = self.df.columns.to_list()[:6] + [feature_name.value]
        feature = TabularFeature.from_config(
            config=feature_config, df=self.df[selected_columns]
        )
        return feature

    def get_facet(self, piece_id: PieceID) -> Facet:
        try:
            sliced = self.df.loc[piece_id,]
        except KeyError:
            raise KeyError(
                f"This {self.name} cannot be subscripted with '{piece_id}'. Pass a PieceID"
            )
        facet_constructor = get_facet_class(self.dtype)
        return facet_constructor.from_id(
            config_id=self, df=sliced, piece_id=piece_id, file_path=self.file_path
        )


@dataclass(frozen=True)
class StackedMeasures(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedNotes(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedRests(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedNotesAndRests(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedLabels(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedHarmonies(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedFormLabels(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedCadences(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedEvents(StackedFacet):
    pass


@dataclass(frozen=True)
class StackedPositions(StackedFacet):
    pass


@lru_cache()
def get_stacked_facet_class(name: [FacetName, str]) -> Type[StackedFacet]:
    facet_name = str2facet_name(name)
    name2facet = {
        FacetName.Measures: StackedMeasures,
        FacetName.Notes: StackedNotes,
        FacetName.Rests: StackedRests,
        FacetName.NotesAndRests: StackedNotesAndRests,
        FacetName.Labels: StackedLabels,
        FacetName.Harmonies: StackedHarmonies,
        FacetName.FormLabels: StackedFormLabels,
        FacetName.Cadences: StackedCadences,
        FacetName.Events: StackedEvents,
        FacetName.Positions: StackedPositions,
        FacetName.StackedMeasures: StackedMeasures,
        FacetName.StackedNotes: StackedNotes,
        FacetName.StackedRests: StackedRests,
        FacetName.StackedNotesAndRests: StackedNotesAndRests,
        FacetName.StackedLabels: StackedLabels,
        FacetName.StackedHarmonies: StackedHarmonies,
        FacetName.StackedFormLabels: StackedFormLabels,
        FacetName.StackedCadences: StackedCadences,
        FacetName.StackedEvents: StackedEvents,
        FacetName.StackedPositions: StackedPositions,
    }
    return name2facet.get(facet_name)


# endregion StackedFacet


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
    harmonies3 = Harmonies.from_id(config_id=harmonies2, df=df)
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

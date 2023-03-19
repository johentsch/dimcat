from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass, replace
from enum import Enum, IntEnum, auto
from functools import cached_property, lru_cache
from typing import (
    Callable,
    ClassVar,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Type,
    Union,
    runtime_checkable,
)

import pandas as pd
from dimcat.base import (
    Configuration,
    ConfiguredDataframe,
    PieceIdentifier,
    PieceStackIdentifier,
    Stack,
    StackConfig,
)
from dimcat.dtypes.base import (
    DataBackend,
    PieceID,
    SomeDataframe,
    SomeFeature,
    WrappedSeries,
)
from dimcat.dtypes.sequence import Bigrams, ContiguousSequence
from dimcat.utils.decorators import config_dataclass
from dimcat.utils.functions import get_value_profile_mask
from typing_extensions import Self

logger = logging.getLogger(__name__)

TIME_COLUMNS = [
    "mc",
    "mn",
    "quarterbeats",
    "duration_qb",
    "mc_onset",
    "mn_onset",
    "duration",
]
LAYER_COLUMNS = ["staff", "voice"]
CONTEXT_COLUMNS = TIME_COLUMNS + LAYER_COLUMNS


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
    # CUSTOM = ("TabularFeature", None)
    #
    # @classmethod
    # def _missing_(cls, value):
    #     return FeatureName.CUSTOM


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
    dtype: FeatureName
    feature_type: FeatureType
    n_columns: int
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
    feature_type: FeatureType = FeatureType.TabularFeature
    n_columns: int = 1
    df_type: DataBackend = DataBackend.PANDAS
    unfold: bool = False
    interval_index: bool = True
    concat_method: Callable[
        [Dict[PieceID, TabularFeature], Sequence[str]], TabularFeature
    ] = pd.concat


@config_dataclass(dtype=FeatureName, df_type=DataBackend)
class FeatureID(FeatureConfig):
    identifier: Union[PieceStackIdentifier, PieceIdentifier]


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


@config_dataclass(dtype=FacetName, df_type=DataBackend)
class FacetID(FacetConfig):
    identifier: Union[PieceStackIdentifier, PieceIdentifier]

    @property
    def piece_id(self) -> PieceID:
        return self.identifier.piece_id


def feature_config2facet_config(feature: Configuration):
    feature_config = FeatureConfig.from_dataclass(feature)
    facet_name = feature_config.dtype.facet
    stacked_facet_class = get_stacked_facet_class(facet_name)
    return stacked_facet_class.get_default_config()


# endregion Enums and Configs

# region Features


@dataclass(frozen=True)
class Feature(ConfiguredDataframe):
    """used as a Mixin; groups TabularFeature and StackedFeature"""

    _config_type: ClassVar[Type[FeatureConfig]] = FeatureConfig
    _default_config_type: ClassVar[Type[DefaultFeatureConfig]] = DefaultFeatureConfig
    _id_type: ClassVar[Type[FeatureID]] = FeatureID
    _enum_type: ClassVar[Type[Enum]] = FeatureType

    @classmethod
    def from_df(
        cls,
        df: SomeDataframe,
        identifier: Optional[Configuration] = None,
        **kwargs,
    ) -> Self:
        """Create a Feature from a dataframe and a :obj:`Configuration`. The required identifiers can be given either
        as :obj:`FeatureIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        return cls.from_default(df=df, identifier=identifier, **kwargs)


@dataclass(frozen=True)
class TabularFeature(FeatureID, Feature):
    pass


@dataclass(frozen=True)
class StackedFeature(Stack, Feature):
    pass


# endregion Features

# region Facets


@config_dataclass(dtype=FacetName, df_type=DataBackend)
class FacetMixin(ConfiguredDataframe):
    """Classes structurally implementing the PFacet protocol."""

    _config_type: ClassVar[Type[FacetConfig]] = FacetConfig
    _default_config_type: ClassVar[Type[DefaultFacetConfig]] = DefaultFacetConfig
    _id_type: ClassVar[Type[FacetID]] = FacetID
    _enum_type: ClassVar[Type[Enum]] = FacetName

    # region Default methods repeated for type hints

    @property
    def config(self) -> FacetConfig:
        return self._config_type.from_dataclass(self)

    @cached_property
    def context_columns(self) -> List[str]:
        return [col for col in self.df.columns if col in CONTEXT_COLUMNS]

    @classmethod
    @property
    def dtype(cls) -> Enum:
        """Name of the class as enum member."""
        return cls._enum_type(cls.name)

    @property
    def ID(self) -> FacetID:
        return self._id_type.from_dataclass(self)

    @classmethod
    def from_df(
        cls,
        df: Union[SomeDataframe, ConfiguredDataframe],
        identifier: Optional[PieceIdentifier] = None,
        **kwargs,
    ) -> Self:
        """Create a Facet from a dataframe and a :obj:`Configuration`. The required identifiers can be given either
        as :obj:`FacetIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        if identifier is None:
            try:
                identifier = PieceIdentifier.from_dataclass(df)
            except Exception:
                raise ValueError(
                    f"{type(df)!r} could not be turned into an identifier and none was given."
                )
        return cls.from_default(df=df, identifier=identifier, **kwargs)

    @classmethod
    def get_default_config(cls, **kwargs) -> DefaultFacetConfig:
        kwargs["dtype"] = cls.dtype
        return cls._default_config_type.from_dict(kwargs)

    # endregion Default methods repeated for type hints


@dataclass(frozen=True)
class Facet(FacetID, FacetMixin):
    def get_feature(self, feature: Union[FeatureName, FeatureConfig]) -> TabularFeature:
        """In its basic form, get one of the columns as a :obj:`WrappedSeries`.
        Subclasses may offer additional features, such as transformed columns or subsets of the table.
        """
        if isinstance(feature, Configuration):
            if isinstance(feature, FeatureID):
                raise NotImplementedError("Not accepting IDs as of now, only configs.")
            feature_config = FeatureConfig.from_dataclass(feature)
            feature_columns = [feature_config.dtype.value]
        else:
            try:
                feature_name = str2feature_name(feature)
                feature_columns = [feature_name.value]
            except ValueError:
                if isinstance(feature, Enum):
                    feature_name = feature.value
                    feature_columns = [feature_name]
                else:
                    feature_name = str(feature)
                    if isinstance(feature, str):
                        feature_columns = [feature]
                    else:
                        feature_columns = feature
            feature_config = DefaultFeatureConfig(dtype=feature_name)
        columns = self.context_columns + feature_columns
        result = TabularFeature.from_config(
            config=feature_config,
            identifier=self.identifier,
            df=self.df[columns],
        )
        return result


@dataclass(frozen=True)
class CadencesMixin(ABC):
    pass


@dataclass(frozen=True)
class EventsMixin(ABC):
    pass


@dataclass(frozen=True)
class FormLabelsMixin(ABC):
    pass


@dataclass(frozen=True)
class HarmoniesMixin(ABC):
    facet_name = FacetName.Harmonies

    @cached_property
    def globalkey(self) -> str:
        return self.get_column(FeatureName.GLOBALKEY)[0]

    def get_localkey_bigrams(self) -> Bigrams:
        """Returns a TypedSequence of bigram tuples representing modulations between local keys."""
        localkey_list = ContiguousSequence(
            self.get_column(FeatureName.LOCALKEY)
        ).get_changes()
        return localkey_list.get_n_grams(n=2)

    def get_chord_bigrams(self) -> Bigrams:
        chords = self.get_column("chord")
        return chords.get_n_grams(2)


@dataclass(frozen=True)
class LabelsMixin(ABC):
    pass


@dataclass(frozen=True)
class MeasuresMixin(ABC):
    pass


class NotesMixin(ABC):
    facet_name = FacetName.Notes

    def tpc(self) -> WrappedSeries:
        series = self.get_feature(FeatureName.TPC)
        return WrappedSeries(series)


@dataclass(frozen=True)
class NotesAndRestsMixin(ABC):
    pass


@dataclass(frozen=True)
class PositionsMixin(ABC):
    pass


@dataclass(frozen=True)
class RestsMixin(ABC):
    pass


@dataclass(frozen=True)
class Cadences(Facet, CadencesMixin):
    pass


@dataclass(frozen=True)
class Events(Facet, EventsMixin):
    pass


@dataclass(frozen=True)
class FormLabels(Facet, FormLabelsMixin):
    pass


@dataclass(frozen=True)
class Harmonies(Facet, HarmoniesMixin):
    pass


@dataclass(frozen=True)
class Labels(Facet, LabelsMixin):
    pass


@dataclass(frozen=True)
class Measures(Facet, MeasuresMixin):
    pass


@dataclass(frozen=True)
class Notes(Facet, NotesMixin):
    pass


@dataclass(frozen=True)
class NotesAndRests(Facet, NotesAndRestsMixin):
    pass


@dataclass(frozen=True)
class Positions(Facet, PositionsMixin):
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


@dataclass(frozen=True)
class StackedFacet(Stack, FacetMixin):
    @classmethod
    def get_default_config(cls, **kwargs) -> StackConfig:
        if "configuration" not in kwargs:
            kwargs["configuration"] = DefaultFacetConfig(dtype=cls.facet_name)
        return super().get_default_config(**kwargs)

    def get_feature(self, feature: Union[FeatureName, FeatureConfig]) -> StackedFeature:
        """In its basic form, get one of the columns as a :obj:`WrappedSeries`.
        Subclasses may offer additional features, such as transformed columns or subsets of the table.
        """
        if isinstance(feature, Configuration):
            if isinstance(feature, FeatureID):
                raise NotImplementedError("Not accepting IDs as of now, only configs.")
            feature_config = FeatureConfig.from_dataclass(feature)
            feature_columns = [feature_config.dtype.value]
        else:
            try:
                feature_name = str2feature_name(feature)
                feature_columns = [feature_name.value]
            except ValueError:
                if isinstance(feature, Enum):
                    feature_name = feature.value
                    feature_columns = [feature_name]
                else:
                    feature_name = str(feature)
                    if isinstance(feature, str):
                        feature_columns = [feature]
                    else:
                        feature_columns = feature
            feature_config = DefaultFeatureConfig(dtype=feature_name)
        columns = self.context_columns + feature_columns
        result = StackedFeature(
            df=self.df[columns],
            configuration=feature_config,
            identifier=self.identifier,
        )
        return result


@dataclass(frozen=True)
class StackedCadences(StackedFacet, CadencesMixin):
    pass


@dataclass(frozen=True)
class StackedEvents(StackedFacet, EventsMixin):
    pass


@dataclass(frozen=True)
class StackedFormLabels(StackedFacet, FormLabelsMixin):
    pass


@dataclass(frozen=True)
class StackedHarmonies(StackedFacet, HarmoniesMixin):
    pass


@dataclass(frozen=True)
class StackedLabels(StackedFacet, LabelsMixin):
    pass


@dataclass(frozen=True)
class StackedMeasures(StackedFacet, MeasuresMixin):
    pass


@dataclass(frozen=True)
class StackedNotes(StackedFacet, NotesMixin):
    pass


@dataclass(frozen=True)
class StackedNotesAndRests(StackedFacet, NotesAndRestsMixin):
    pass


@dataclass(frozen=True)
class StackedPositions(StackedFacet, PositionsMixin):
    pass


@dataclass(frozen=True)
class StackedRests(StackedFacet, RestsMixin):
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
    id_dict = dict(
        piece_id=PieceID("corelli", "op01n01a"),
    )
    f_id = PieceIdentifier.from_dict(id_dict)
    df = ms3.load_tsv(file_path)
    harmonies1 = Harmonies.from_default(df=df, identifier=f_id)
    f_cfg = Harmonies.get_default_config()
    harmonies2 = Harmonies.from_config(df=df, config=f_cfg, identifier=f_id)
    assert harmonies1 == harmonies2
    harmonies3 = Harmonies.from_id(config_id=harmonies2, df=df)
    assert harmonies2 == harmonies3
    harmonies4 = Harmonies.from_df(df=df, identifier=f_id)
    assert harmonies3 == harmonies4
    chords_as_sequence = ContiguousSequence(harmonies2[FeatureName.LOCALKEY])
    chords_as_wrapped_series = harmonies1[FeatureName.LOCALKEY]
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


def facet_argument2config(facet=Union[FacetName, Configuration]) -> FacetConfig:
    if isinstance(facet, Configuration):
        config = FacetConfig.from_dataclass(facet)
        if isinstance(config.dtype, str):
            config = replace(config, dtype=FacetName(config.dtype))
    else:
        facet_name = str2facet_name(facet)
        config = DefaultFacetConfig(dtype=FacetName(facet_name))
    return config

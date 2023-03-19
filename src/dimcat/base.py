from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, fields
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Collection,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)

from dimcat.dtypes.base import PieceID, SomeDataframe, WrappedDataframe
from dimcat.dtypes.sequence import PieceIndex
from typing_extensions import Self

if TYPE_CHECKING:
    from dimcat.data.facet import FacetConfig, FeatureConfig


class Data(ABC):
    """
    Subclasses are the dtypes that this library uses. Every PipelineStep accepts one or several
    dtypes.

    The initializer can set parameters influencing how the contained data will look and is able
    to create an object from an existing Data object to enable type conversion.
    """

    _registry: ClassVar[Dict[str, Type]] = {}
    """Register of all subclasses."""

    def __init__(self, **kwargs):
        pass

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls


class PipelineStep(ABC):
    """
    A PipelineStep object is able to transform some data in a pre-defined way.

    The initializer will set some parameters of the transformation, and then the
    `process_data` function is used to transform an input Data object, returning a copy.
    """

    _registry: ClassVar[Dict[str, Type]] = {}
    """Register of all subclasses."""

    def __init__(self, **kwargs):
        self.required_facets = []
        """Specifies a list of facets (such as 'notes' or 'labels') that the passed Data object
        needs to provide."""

    def __init_subclass__(cls, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__

    def check(self, _) -> Tuple[bool, str]:
        """Test piece of data for certain properties before computing analysis.

        Returns
        -------
        :obj:`bool`
            True if the passed data is eligible.
        :obj:`str`
            Error message in case the passed data is not eligible.
        """
        return True, ""

    def filename_factory(self):
        return self.name

    @abstractmethod
    def process_data(self, data: Data) -> Data:
        """
        Perform a transformation on an input Data object. This should never alter the
        Data or its properties in place, instead returning a copy or view of the input.

        Parameters
        ----------
        data : :obj:`Data`
            The data to be transformed. This should not be altered in place.

        Returns
        -------
        :obj:`Data`
            A copy or view of the input Data, transformed in some way defined by this
            PipelineStep.
        """


PS_TYPES = dict(PipelineStep._registry)
D_TYPES = dict(Data._registry)


@dataclass(frozen=True)
class Configuration(Data):
    @classmethod
    def from_dataclass(cls, config: Configuration, **kwargs) -> Self:
        """This class methods copies the fields it needs from another config-like dataclass."""
        init_args = cls.dict_from_dataclass(config, **kwargs)
        return cls(**init_args)

    @classmethod
    def from_dict(cls, config: dict, **kwargs) -> Self:
        """This class methods copies the fields it needs from another config-like dataclass."""
        if not isinstance(config, dict):
            raise TypeError(
                f"Expected a dictionary, received a {type(config)!r} instead."
            )
        config = dict(config)
        config.update(kwargs)
        field_names = [field.name for field in fields(cls) if field.init]
        init_args = {key: value for key, value in config.items() if key in field_names}
        return cls(**init_args)

    @classmethod
    def dict_from_dataclass(cls, config: Configuration, **kwargs) -> Dict:
        """This class methods copies the fields it needs from another config-like dataclass."""
        init_args: Dict[str, Any] = {}
        field_names = []
        for config_field in fields(cls):
            if not config_field.init:
                continue
            field_name = config_field.name
            field_names.append(config_field.name)
            if not hasattr(config, field_name):
                continue
            init_args[field_name] = getattr(config, field_name)
        init_args.update(kwargs)
        return init_args

    def __eq__(self, other):
        return astuple(self) == astuple(other)


@dataclass(frozen=True)
class PieceIdentifier(Configuration):
    """Identifies one piece and, in combination with a Configuration, information pertaining to it."""

    piece_id: PieceID


@dataclass(frozen=True)
class PieceStackIdentifier(Configuration):
    """Identifies several pieces and, in combination with a Configuration, information pertaining to them."""

    piece_index: PieceIndex


@dataclass(frozen=True)
class ConfiguredObjectMixin(ABC):
    """"""

    _config_type: ClassVar[Type[Configuration]]
    _default_config_type: ClassVar[Type[Configuration]]
    _id_type: ClassVar[Type[Configuration]]
    _enum_type: ClassVar[Type[Enum]]

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__

    @property
    def config(self) -> Configuration:
        return self._config_type.from_dataclass(self)

    @classmethod
    @property
    def dtype(cls) -> Union[Enum, str]:
        """Name of the class as enum member (if cls._enum_type is define, string otherwise)."""
        if hasattr(cls, "_enum_type"):
            return cls._enum_type(cls.name)
        return cls.name

    @property
    def ID(self) -> Configuration:
        return self._id_type.from_dataclass(self)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        identifier: Optional[Configuration] = None,
        **kwargs,
    ) -> Self:
        """Create a Feature from a dataframe and a :obj:`Configuration`. The required identifiers can be given either
        as :obj:`FeatureIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        cfg_kwargs = cls._config_type.dict_from_dataclass(config, **kwargs)
        if identifier is not None and not hasattr(cls, "_id_type"):
            warnings.warn(
                f"{cls.name} objects need no identifier since their configuration makes them unique."
            )
        return cls(**cfg_kwargs, identifier=identifier)

    @classmethod
    def from_default(
        cls,
        identifier: Optional[Configuration] = None,
        **kwargs,
    ) -> Self:
        """Create an instance from the default :obj:`Configuration`, which can be modified using keyword arguments."""
        config = cls.get_default_config(**kwargs)
        return cls.from_config(config=config, identifier=identifier, **kwargs)

    @classmethod
    def from_id(cls, config_id: Configuration, **kwargs) -> Self:
        id_kwargs = cls._id_type.dict_from_dataclass(config_id)
        id_kwargs.update(kwargs)
        return cls(**id_kwargs)

    @classmethod
    def get_default_config(cls, **kwargs) -> Configuration:
        if not isinstance(cls.dtype, str):
            kwargs["dtype"] = cls.dtype
        return cls._default_config_type.from_dict(kwargs)


@dataclass(frozen=True)
class ConfiguredDataframe(ConfiguredObjectMixin, WrappedDataframe, Data):
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
class StackConfig(Configuration):
    configuration: Configuration


@dataclass(frozen=True)
class DefaultStackConfig(StackConfig):
    configuration: Configuration


@dataclass(frozen=True)
class StackID(StackConfig):
    identifier: PieceStackIdentifier

    @property
    def piece_index(self) -> PieceIndex:
        return self.identifier.piece_index


@dataclass(frozen=True)
class Stack(StackID, ConfiguredDataframe):
    _config_type: ClassVar[Type[Configuration]] = StackConfig
    _default_config_type: ClassVar[Type[Configuration]] = DefaultStackConfig
    _id_type: ClassVar[Type[Configuration]] = StackID

    @property
    def dtype(self) -> Enum:
        return self.configuration.dtype

    @classmethod
    def from_list(
        cls,
        list_of_dataframes: List[Union[SomeDataframe, ConfiguredDataframe]],
        configuration: Optional[Union[FacetConfig, FeatureConfig]] = None,
        identifier: Optional[PieceStackIdentifier] = None,
    ) -> Self:
        if len(list_of_dataframes) == 0:
            raise ValueError("Cannot create empty stack.")
        first_element = list_of_dataframes[0]
        if configuration is None:
            try:
                configuration = first_element.config
            except Exception:
                raise ValueError(
                    f"{type(first_element)!r} is not a configured object and no configuration was given."
                )
        if identifier is None:
            try:
                piece_index = PieceIndex([df.piece_id for df in list_of_dataframes])
            except Exception:
                piece_ids = []
                for df in list_of_dataframes:
                    try:
                        piece_ids.append(df.piece_id)
                    except Exception:
                        raise ValueError(
                            f"{type(df)!r} does not have a piece_id and no identifier was given."
                        )
                    id_types = set(type(piece_id) for piece_id in piece_ids)
                    raise ValueError(
                        f"Unable to create PieceIndex from types {id_types}"
                    )
            identifier = PieceStackIdentifier(piece_index=piece_index)
        else:
            n_ids, n_dfs = len(identifier.piece_index), len(list_of_dataframes)
            if n_ids == n_dfs:
                pass  # ToDo: verify given IDs against those of the ConfigurerDataframes or SomeDataframe.index
            elif n_ids > n_dfs:
                raise ValueError(
                    f"Given identifier has {n_ids} PieceIDs but the given list has only {n_dfs} frames."
                )
        concat_method = configuration.concat_method
        concatenated_frames = concat_method(list_of_dataframes)
        return cls.from_df(
            df=concatenated_frames, configuration=configuration, identifier=identifier
        )

    @classmethod
    def from_df(
        cls,
        df: Union[SomeDataframe, ConfiguredDataframe],
        configuration: Optional[Union[FacetConfig, FeatureConfig]] = None,
        identifier: Optional[PieceStackIdentifier] = None,
        **kwargs,
    ) -> Self:
        if configuration is None:
            try:
                configuration = df.config
            except AttributeError:
                pass
                raise ValueError(
                    f"{type(df)!r} is not a configured object and no configuration was given."
                )
        if identifier is None:
            if hasattr(df, "identifier") and isinstance(df, ConfiguredDataframe):
                identifier = df.identifier
            else:
                idx = df.index
                n_levels = idx.nlevels
                if n_levels < 2:
                    raise ValueError(
                        "The given DataFrame has less than two levels and no identifier was given."
                    )
                while idx.nlevels > 2:
                    idx = idx.droplevel(-1)
                piece_index = PieceIndex(idx.unique())
                identifier = PieceStackIdentifier(piece_index=piece_index)
        return cls.from_default(
            df=df, configuration=configuration, identifier=identifier, **kwargs
        )

    def get_piece(self, piece_id: PieceID):
        df = self.df.loc[piece_id,]
        constructor = typestring2type(self.dtype)
        identifier = PieceIdentifier(piece_id=piece_id)
        return constructor.from_config(
            config=self.configuration,
            df=df,
            identifier=identifier,
        )


def typestrings2types(
    typestrings: Union[Union[str, Enum], Collection[Union[str, Enum]]]
) -> Tuple[type]:
    """Turns one or several names of classes contained in this module into a
    tuple of references to these classes."""
    if isinstance(typestrings, (str, Enum)):
        typestrings = [typestrings]
    result = [typestring2type(typestring) for typestring in typestrings]
    return tuple(result)


def typestring2type(typestring: Union[str, Enum]) -> type:
    if isinstance(typestring, Enum):
        typestring = typestring.value
    d_types = Data._registry
    ps_types = PipelineStep._registry
    if typestring in d_types:
        return d_types[typestring]
    elif typestring in ps_types:
        return ps_types[typestring]
    raise KeyError(
        f"Typestring '{typestring}' does not correspond to a known subclass of PipelineStep or Data:\n"
        f"{ps_types}\n{d_types}"
    )

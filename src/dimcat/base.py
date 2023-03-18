from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import astuple, dataclass, fields
from enum import Enum
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, Union

from dimcat.dtypes import WrappedDataframe
from dimcat.dtypes.base import SomeDataframe, logger
from typing_extensions import Self


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
    def dict_from_dataclass(cls, config: Configuration, **kwargs) -> Dict:
        """This class methods copies the fields it needs from another config-like dataclass."""
        init_args: Dict[str, Any] = {}
        field_names = []
        for config_field in fields(cls):
            field_names.append(config_field.name)
            value = getattr(config, config_field.name, None)
            if value is None:
                continue
            init_args[config_field.name] = value
        if any(kw not in field_names for kw in kwargs.keys()):
            foreign = [kw for kw in kwargs.keys() if kw not in field_names]
            plural = (
                f"s '{', '.join(foreign)}'" if len(foreign) > 1 else f" '{foreign[0]}'"
            )
            logger.warning(f"Keyword argument{plural} not a field of {cls.name}.")
        init_args.update(kwargs)
        return init_args

    def __eq__(self, other):
        return astuple(self) == astuple(other)


@dataclass(frozen=True)
class ConfiguredObjectMixin(ABC):
    """"""

    _config_type: ClassVar[Type[Configuration]]
    _default_config_type: ClassVar[Type[Configuration]]
    _id_config_type: ClassVar[Type[Configuration]]
    _id_type: ClassVar[Type[Configuration]]
    _enum_type: ClassVar[Type[Enum]]

    @classmethod
    @property
    def name(cls) -> str:
        return cls.__name__

    @property
    def config(self) -> Configuration:
        return self._config_type.from_dataclass(self)

    @property
    def identifier(self) -> Configuration:
        return self._id_type.from_dataclass(self)

    @classmethod
    @property
    def dtype(cls) -> Union[Enum, str]:
        """Name of the class as enum member."""
        if hasattr(cls, "_enum_type"):
            return cls._enum_type(cls.name)
        return cls.name

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
        cfg_kwargs = cls._config_type.dict_from_dataclass(config)
        if identifier is not None:
            if hasattr(cls, "_id_type"):
                warnings.warn(
                    f"{cls.name} objects need no identifier since their configuration makes them unique."
                )
            else:
                id_kwargs = cls._id_config_type.dict_from_dataclass(identifier)
                cfg_kwargs.update(id_kwargs)
        cfg_kwargs.update(kwargs)
        if cfg_kwargs["dtype"] != cls.dtype:
            cfg_class = config.__class__.__name__
            raise TypeError(
                f"Cannot initiate {cls.name} with {cfg_class}.dtype={config.dtype!r}."
            )
        return cls(**cfg_kwargs)

    @classmethod
    def from_default(
        cls,
        identifiers: Optional[Configuration] = None,
        **kwargs,
    ) -> Self:
        """Create an instance from the default :obj:`Configuration`, which can be modified using keyword arguments."""
        if len(kwargs) == 0:
            config = cls.get_default_config()
            return cls.from_config(config=config, identifier=identifiers)
        if cls._id_config_type == Configuration:
            # this class takes no special identifiers, so the kwargs need no splitting
            config = cls.get_default_config(**kwargs)
            return cls.from_config(config=config, identifier=identifiers)
        # split id_kwargs from cfg_kwargs
        cfg_field_names = [fld.name for fld in fields(cls._config_type)]
        cfg_kwargs = {kw: arg for kw, arg in kwargs.items() if kw in cfg_field_names}
        config = cls.get_default_config(**cfg_kwargs)
        if len(kwargs) > len(cfg_kwargs):
            id_kwargs = {
                kw: arg for kw, arg in kwargs.items() if kw not in cfg_field_names
            }
            return cls.from_config(config=config, identifier=identifiers, **id_kwargs)
        return cls.from_config(config=config, identifier=identifiers)

    @classmethod
    def from_id(cls, config_id: Configuration, **kwargs) -> Self:
        id_kwargs = cls._id_type.dict_from_dataclass(config_id)
        id_kwargs.update(kwargs)
        return cls(**id_kwargs)

    @classmethod
    def get_default_config(cls, **kwargs) -> Configuration:
        if not isinstance(cls.dtype, str):
            kwargs["dtype"] = cls.dtype
        return cls._default_config_type(**kwargs)


@dataclass(frozen=True)
class ConfiguredDataframe(ConfiguredObjectMixin, WrappedDataframe):
    @classmethod
    def from_df(
        cls,
        df: SomeDataframe,
        identifiers: Optional[Configuration] = None,
        **kwargs,
    ) -> Self:
        """Create a Feature from a dataframe and a :obj:`Configuration`. The required identifiers can be given either
        as :obj:`FeatureIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        return cls.from_default(df=df, identifiers=identifiers, **kwargs)

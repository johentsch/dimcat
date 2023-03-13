from __future__ import annotations

import logging
import warnings
from abc import ABC
from dataclasses import astuple, dataclass, fields
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Generic,
    Iterable,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeAlias,
    TypeVar,
    Union,
    _GenericAlias,
    get_args,
    overload,
)

import modin.pandas as mpd
import numpy as np
import pandas as pd
from dimcat.base import Data
from scipy.stats import entropy
from typing_extensions import Self

PathLike: TypeAlias = Union[str, Path]


logger = logging.getLogger(__name__)
T_co = TypeVar("T_co", covariant=True)
C = TypeVar("C")  # convertible
Out = TypeVar("Out")  # output


class PieceID(NamedTuple):
    corpus: str
    piece: str


GroupID: TypeAlias = tuple
SliceID: TypeAlias = Tuple[str, str, pd.Interval]
SomeID: TypeAlias = Union[PieceID, SliceID]


class AnalyzerName(str, Enum):
    PitchClassVectors = "PitchClassVectors"


class TypedSequence(Sequence[T_co]):
    """A TypedSequence behaves like a list in many aspects but with the difference that it
    imposes one particular data type on its elements.

    If it is instantiated without a converter, the type will be inferred from the first element:

        >>> A = TypedSequence([[1], '2', {7}])
        TypedSequence[list]([[1], ['2'], [7]])

    However, this only works if ``type(a)`` yields a constructor, that works on all elements,
    otherwise a converter needs to be passed:

        >>> converter = lambda e: (e,)
        >>> B = TypedSequence([1, 2, 3], converter)
        TypedSequence[tuple]([(1,), (2,), (3,)])

    TypedSequences can be nested, i.e. have other TypedSequences as elements. The base class,
    however, does not enforce equal type parameters on them:

        >>> C = TypedSequence([A, B, [1, 2.1, 3.9]])
        TypedSequence[TypedSequence]([TypedSequence[list]([[1], ['2'], [7]]),
                                      TypedSequence[tuple]([(1,), (2,), (3,)]),
                                      TypedSequence[int]([1, 2, 3])])

    The module as a few example of parametrized subtypes. Notably, subtypes can register
    themselves as default type for instantiating a TypedSequence with a particular first value.
    This is useful for downcasting to a subclass that has the fitting converter pre-defined.
    For example, ``PieceIndex`` is defined as

        class PieceIndex(TypedSequence[PieceID], register_for=[PieceID]):

    where ``register_for=List[Type]`` makes sure any TypedSequence instantiated
    **without a custom converter** and with a first element of type ``Type`` will be
    cast to that subclass.
    """

    _type_parameter: Optional[T_co] = None
    """Stores the value of the parametrized type."""
    _type2subclass: Dict[Type[T_co], Type[TypedSequence]] = {}
    """Registry of all subclasses that are defined with ``register_for=[type,...]``. Whenever a TypedSequence is
    initiated without converter, the __new__() method looks at the type of the first value (if any) and, if it
    is contained in the registry, creates an object from the pertinent subclass.
    """

    def __len__(self):
        return len(self.values)

    def __getitem__(self, int_or_slice):
        if isinstance(int_or_slice, (int, slice)):
            return self.values[int_or_slice]
        raise KeyError(f"{self.name} cannot be subscripted with {int_or_slice}")

    def to_series(self):
        try:
            S = pd.Series(self.values, dtype=self.dtype)
        except Exception:
            S = pd.Series(self.values, dtype=object)
        return S

    def map(
        self,
        func: Callable[[T_co], Out],
    ) -> TypedSequence[Out]:
        try:
            values = list(map(func, self.values))
        except Exception as e:
            raise TypeError(f"Mapping {func} onto {self.name} failed with:\n'{e}'")
        sequential = TypedSequence(values=values)
        return sequential

    def filtered_by_condition(
        self,
        condition: Callable[[T_co], bool],
    ) -> Iterator[T_co]:
        yield from (x for x in self.values if condition(x))

    def __init_subclass__(cls, register_for=None, **kwargs):
        """Registers every subclass under the class variable :attr:`_registry`"""
        super().__init_subclass__(**kwargs)
        if register_for is not None:
            for dtype in register_for:
                if dtype in cls._type2subclass:
                    raise KeyError(
                        f"Type {dtype} had already been registered by {cls._type2subclass[dtype]}."
                    )
                cls._type2subclass[dtype] = cls
            logger.debug(
                f"{cls}: TypedSequence will default to a [{cls}] if first value is {register_for}."
            )
        # The following two lines make the value of T available for all parametrized subclasses.
        # Thanks to Paweł Rubin via https://stackoverflow.com/a/71720366
        cls._type_parameter = get_args(cls.__orig_bases__[0])[0]
        logger.debug(f"{cls}._type_parameter = {get_args(cls.__orig_bases__[0])[0]}")

    def __new__(
        cls,
        values: Sequence[Union[T_co, C]],
        converter: Optional[Callable[[C], T_co]] = None,
        **kwargs,
    ):
        if not isinstance(values, (Sequence, np.ndarray, pd.Series, WrappedSeries)):
            raise TypeError(
                f"{cls.__name__}: The first argument needs to be a Sequence, not {type(values)}."
            )
        nonempty = len(values) > 0
        if converter is None and nonempty:
            first_type = type(values[0])
            if (
                first_type in TypedSequence._type2subclass
                and cls.__name__ == "TypedSequence"
            ):
                new_object_type = TypedSequence._type2subclass[first_type]
                logger.debug(
                    f"Creating {new_object_type} because {first_type} is in {TypedSequence._type2subclass.keys()}"
                )
                return super().__new__(new_object_type)
            logger.debug(
                f"Creating {cls} because {first_type} is not in {TypedSequence._type2subclass.keys()}"
            )
        return super().__new__(cls)

    @overload
    def __init__(self, values: Sequence[T_co], converter: Literal[None], **kwargs):
        ...

    @overload
    def __init__(self, values: Sequence[C], converter: Callable[[C], T_co], **kwargs):
        ...

    def __init__(
        self,
        values: Sequence[Union[T_co, C]],
        converter: Optional[Callable[[C], T_co]] = None,
        **kwargs,
    ):
        """Sequence object that converts all elements to the same data type.

        If no converter is passed, it is inferred in two different ways:
        a) If self is a subclass of TypedSequence that has been parametrized with class T,
           T is used as a converter. That is, it needs to work as a constructor, which is
           tested using callable(T). Otherwise:
        b) As a fallback, the type of the first value is used for the entire sequence.

        Args:
            values:
                The values you want to create the sequence from. If one of them cannot be converted,
                a TypeError will be thrown.
            converter: A callable that converts values of all expected/allowed types to T_co.
        """
        logger.debug(
            f"{self.__class__.__name__}(values={list(values)}, converter={converter})"
        )
        self._values: List[T_co] = []
        self._converter: Optional[Callable[[C], T_co]] = None
        self.converter = converter
        self.values = values

    @property
    def converter(self) -> Optional[Callable[[Any], T_co]]:
        if self._converter is not None:
            return self._converter
        if not isinstance(self._type_parameter, (TypeVar, _GenericAlias)):
            return self._type_parameter

    @converter.setter
    def converter(self, converter: Optional[Callable[[Any], T_co]]):
        self._converter = converter

    @property
    def dtype(self) -> Optional[Type]:
        if self._type_parameter is None or isinstance(
            self._type_parameter, (TypeVar, _GenericAlias)
        ):
            if len(self.values) > 0:
                return type(self.values[0])
            else:
                return self.converter
        return self._type_parameter

    @property
    def name(self) -> str:
        name = self.__class__.__name__
        if self.dtype is None:
            name += "[None]"
        else:
            name += f"[{self.dtype.__name__}]"
        return name

    @property
    def values(self) -> List[T_co]:
        return list(self._values)

    @values.setter
    def values(self, values: Sequence[Union[T_co, C]]):
        self._values = [self.convert(val) for val in values]

    def convert(self, value: Union[T_co, C]) -> T_co:
        if self.converter is None:
            # this should happen only once in the object's lifetime
            if all(
                (
                    self._type_parameter is not None,
                    callable(self._type_parameter),
                    not isinstance(self._type_parameter, TypeVar),
                )
            ):
                self.converter = self._type_parameter
            else:
                self.converter = type(value)
        try:
            return self.converter(value)
        except Exception as e:
            raise TypeError(
                f"Conversion {self.converter.__name__}({value}) -> {self.dtype} failed with:\n'{e}'"
            )

    @overload
    def append(self, value: T_co, convert: Literal[False]) -> None:
        ...

    @overload
    def append(self, value: Union[T_co, C], convert: Literal[True]) -> None:
        ...

    def append(self, value: Union[T_co, C], convert: bool = False) -> None:
        if convert:
            self._values.append(self.convert(value))
        else:
            try:
                type_check = isinstance(value, self.dtype)
            except Exception as e:
                if len(self._values) == 0:
                    try:
                        converted_value = self.convert(value)
                    except Exception:
                        raise ValueError(
                            f"The exact dtype of this empty sequence is not yet defined and "
                            f"{value} cannot be converted with {self.converter}."
                        )
                    value_type, converted_type = type(value), type(converted_value)
                    if issubclass(value_type, converted_type):
                        logger.debug(
                            f"First value {value} of type {value_type} is compatible with the type "
                            f"yielded by the converter {self.converter}."
                        )
                        self._values.append(value)
                    else:
                        raise TypeError(
                            f"This sequence is empty but the first value to be appended, {value} "
                            f"has type {value_type}, which seems to be incompatible with the type "
                            f"{converted_type} yielded by the converter {self.converter}. Try setting "
                            f"convert=True."
                        )
                else:
                    raise TypeError(
                        f"Checking the type of {value} against {self.dtype} failed with\n{e}"
                    )
            if type_check:
                self._values.append(value)
            else:
                raise TypeError(
                    f"Cannot append {value} to {self.name}. Try setting convert=True."
                )

    @overload
    def extend(self, values: Iterable[T_co], convert: Literal[False]) -> None:
        ...

    @overload
    def extend(self, values: Iterable[Union[T_co, C]], convert: Literal[True]) -> None:
        ...

    def extend(self, values: Iterable[Union[T_co, C]], convert: bool = False) -> None:
        for value in values:
            self.append(value=value, convert=convert)

    def unique(self) -> TypedSequence[T_co]:
        unique_values = (
            self.to_series().unique()
        )  # keeps order of first occurrence, unlike using set()
        return TypedSequence(unique_values)

    def count(self) -> pd.Series:
        """Count the occurrences of objects in the sequence"""
        return self.to_series().value_counts()

    def mean(self) -> float:
        return self.to_series().mean()

    def probability(self) -> pd.Series:
        return self.to_series().value_counts(normalize=True)

    def entropy(self) -> float:
        """
        The Shannon entropy (information entropy), the expected/average surprisal based on its probability distrib.
        """
        # mean_entropy = self.event_entropy().mean()
        p = self.probability()
        distr_entropy = entropy(p, base=2)
        return distr_entropy

    def surprisal(self) -> pd.Series:
        """The self entropy, information content, surprisal"""
        probs = self.probability()
        self_entropy = -np.log(probs)
        series = pd.Series(data=self_entropy, name="surprisal")
        return series

    def __eq__(self, other) -> bool:
        """Considered as equal when 'other' is a Sequence containing the same values."""
        if isinstance(other, Sequence):
            if len(self._values) != len(other):
                return False
            return all(a == b for a, b in zip(self.values, other))
        return False

    def __repr__(self):
        return f"{self.name}({self.values})"

    def __str__(self):
        return f"{self.name}({self.values})"


class DataBackend(str, Enum):
    PANDAS = "pandas"
    MODIN = "modin"


SomeDataframe: TypeAlias = Union[pd.DataFrame, mpd.DataFrame]
SomeSeries: TypeAlias = Union[pd.Series, mpd.Series]
S = TypeVar("D", bound=SomeDataframe)
S = TypeVar("S", bound=SomeSeries)


@dataclass(frozen=True)
class WrappedSeries(Generic[S]):
    """Wrapper around a Series."""

    series: S

    @classmethod
    def from_series(cls, series: S, **kwargs):
        """Subclasses can implement transformational logic."""
        instance = cls(series=series, **kwargs)
        return instance

    def __getitem__(self, int_or_slice_or_mask):
        if isinstance(int_or_slice_or_mask, (int, slice)):
            return self.series.iloc[int_or_slice_or_mask]
        if isinstance(int_or_slice_or_mask, pd.Series):
            return self.series[int_or_slice_or_mask]
        raise KeyError(f"{self.name} cannot be subscripted with {int_or_slice_or_mask}")

    def __getattr__(self, item):
        """Enable using IndexSequence like a Series."""
        return getattr(self.series, item)

    def __len__(self) -> int:
        return len(self.series.index)


@dataclass(frozen=True)
class WrappedDataframe(Generic[S]):
    """Wrapper around a DataFrame."""

    df: S

    @classmethod
    def from_df(cls, df: S, **kwargs):
        """Subclasses can implement transformational logic."""
        instance = cls(df=df, **kwargs)
        return instance

    def get_feature(self, key: str) -> WrappedSeries[S]:
        """In its basic form, get one of the columns as a :obj:`TypedSequence`.
        Subclasses may offer additional features, such as transformed columns or subsets of the table.
        """
        series: SomeSeries = self.df[key]
        sequential_data = TypedSequence(series)
        return sequential_data

    def __getattr__(self, item):
        """Enable using TabularData like a DataFrame."""
        return getattr(self.df, item)

    def __len__(self) -> int:
        return len(self.df.index)


SomeFeature: TypeAlias = Union[WrappedDataframe, WrappedSeries]


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
    def dtype(cls) -> Enum:
        """Name of the class as enum member."""
        return cls._enum_type(cls.name)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        identifiers: Optional[Configuration] = None,
        **kwargs,
    ) -> Self:
        """Create a Feature from a dataframe and a :obj:`Configuration`. The required identifiers can be given either
        as :obj:`FeatureIdentifiers`, or as keyword arguments. In addition, keyword arguments can be used to override
        values in the given configuration.
        """
        cfg_kwargs = cls._config_type.dict_from_dataclass(config)
        if identifiers is not None:
            if cls._id_config_type == Configuration:
                warnings.warn(
                    f"{cls.name} objects need no identifiers since their configuration makes them unique."
                )
            else:
                id_kwargs = cls._id_config_type.dict_from_dataclass(identifiers)
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
            return cls.from_config(config=config, identifiers=identifiers)
        if cls._id_config_type == Configuration:
            # this class takes no special identifiers, so the kwargs need no splitting
            config = cls.get_default_config(**kwargs)
            return cls.from_config(config=config, identifiers=identifiers)
        # split id_kwargs from cfg_kwargs
        cfg_field_names = [fld.name for fld in fields(cls._config_type)]
        cfg_kwargs = {kw: arg for kw, arg in kwargs.items() if kw in cfg_field_names}
        config = cls.get_default_config(**cfg_kwargs)
        if len(kwargs) > len(cfg_kwargs):
            id_kwargs = {
                kw: arg for kw, arg in kwargs.items() if kw not in cfg_field_names
            }
            return cls.from_config(config=config, identifiers=identifiers, **id_kwargs)
        return cls.from_config(config=config, identifiers=identifiers)

    @classmethod
    def from_id(cls, config_id: Configuration, **kwargs) -> Self:
        id_kwargs = cls._id_type.dict_from_dataclass(config_id)
        id_kwargs.update(kwargs)
        return cls(**id_kwargs)

    @classmethod
    def get_default_config(cls, **kwargs) -> Configuration:
        kwargs["dtype"] = cls.dtype
        return cls._default_config_type(**kwargs)


class SentinelEnum(str, Enum):
    ConfiguredDataframe = "ConfiguredDataframe"


class ConfiguredDataframe(ConfiguredObjectMixin, WrappedDataframe):
    _enum_type: ClassVar[Type[Enum]] = SentinelEnum

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

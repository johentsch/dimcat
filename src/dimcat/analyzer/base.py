"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
from __future__ import annotations

import logging
from abc import ABC
from dataclasses import asdict, dataclass
from enum import Enum
from typing import (
    Any,
    Callable,
    ClassVar,
    Dict,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import pandas as pd
from dimcat.base import (
    Configuration,
    ConfiguredDataframe,
    ConfiguredObjectMixin,
    ConfiguredSeries,
    PieceIdentifier,
    PieceStackIdentifier,
    PipelineStep,
    Stack,
    typestrings2types,
)
from dimcat.data import AnalyzedData, Dataset
from dimcat.data.facet import Feature, FeatureName, StackedFeature
from dimcat.dtypes import PieceID, PieceIndex
from dimcat.dtypes.base import SomeFeature, SomeSeries, WrappedSeries
from typing_extensions import Self

logger = logging.getLogger(__name__)


def _typestring2type(typestring: [str]) -> Type:
    """Returns by name a member of the current scope."""
    return globals()[typestring]


R = TypeVar("R")


class AnalyzerName(str, Enum):
    """Identifies the available analyzers."""

    @classmethod
    def make_tuple(
        cls, facets: Iterable[Union[AnalyzerName, str]]
    ) -> Tuple[AnalyzerName]:
        return tuple(cls(c) for c in facets)

    Counter = "Counter"
    TPCrange = "TPCrange"
    PitchClassVectors = "PitchClassVectors"
    ChordSymbolUnigrams = "ChordSymbolUnigrams"
    ChordSymbolBigrams = "ChordSymbolBigrams"


class ResultName(str, Enum):
    """Identifies the available analyzers."""

    Result = "Result"
    RowResult = "RowResult"
    StackedResult = "StackedResult"


class DispatchStrategy(str, Enum):
    GROUPBY_APPLY = "GROUPBY_APPLY"
    ITER_STACK = "ITER_STACK"


class UnitOfAnalysis(str, Enum):
    SLICE = "SLICE"
    PIECE = "PIECE"
    GROUP = "GROUP"


class Orientation(str, Enum):
    WIDE = "WIDE"
    LONG = "LONG"


@dataclass(frozen=True)
class AnalyzerConfig(Configuration):
    analyzed_feature: FeatureName
    dtype: AnalyzerName
    strategy: DispatchStrategy
    smallest_unit: UnitOfAnalysis
    orientation: Orientation
    fill_na: Any


@dataclass(frozen=True)
class DefaultAnalyzerConfig(AnalyzerConfig):
    analyzed_feature: FeatureName
    dtype: AnalyzerName
    strategy: DispatchStrategy = DispatchStrategy.GROUPBY_APPLY
    smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE
    orientation: Orientation = Orientation.WIDE
    fill_na: Any = None


@dataclass(frozen=True)
class Analyzer(AnalyzerConfig, ConfiguredObjectMixin, PipelineStep):
    """Analyzers are PipelineSteps that process data and store the results in Data.processed.
    The base class performs no analysis, instantiating it serves mere testing purpose.
    """

    _config_type: ClassVar[Type[Configuration]] = AnalyzerConfig
    _default_config_type: ClassVar[Type[Configuration]] = DefaultAnalyzerConfig
    _id_type: ClassVar[Type[Configuration]] = AnalyzerConfig
    _enum_type: ClassVar[Type[Enum]] = AnalyzerName
    _result_type: ResultName = ResultName.RowResult

    assert_all: ClassVar[Tuple[str]] = tuple()
    """Each of these :obj:`PipelineSteps <.PipelineStep>` needs to be matched by at least one PipelineStep previously
     applied to the :obj:`.Dataset`, otherwise :meth:`process_data` raises a ValueError."""

    # assert_previous_step: ClassVar[Tuple[str]] = tuple()
    # """Analyzer.process_data() raises ValueError if last :obj:`PipelineStep` applied to the
    # :obj:`_Dataset` does not match any of these types."""

    excluded_steps: ClassVar[Tuple[str]] = tuple()
    """:meth:`process_data` raises ValueError if any of the previous :obj:`PipelineStep` applied to the
    :obj:`.Dataset` matches one of these types."""

    @staticmethod
    def aggregate(result_a: R, result_b: R) -> R:
        """Static method that combines two results of :meth:`compute`.

        This needs to be equivalent to calling self.compute on the concatenation of the respective data resulting
        in the two arguments."""
        pass

    @staticmethod
    def compute(feature: SomeFeature, **kwargs) -> Any:
        """Static method that performs the actual computation."""
        return feature

    def groupby_apply(
        self, feature: SomeFeature, groupby: SomeSeries = None, **kwargs
    ) -> SomeFeature:
        """Static method that performs the computation on a groupby. The value of ``groupby`` needs to be
        a Series of the same length as ``feature`` or otherwise work as positional argument to feature.groupby().
        """
        if groupby is None:
            return feature.groupby(level=[0, 1]).apply(
                self.compute, **asdict(self.config)
            )
        return feature.groupby(groupby).apply(self.compute, **asdict(self.config))

    def dispatch(self, dataset: Dataset) -> ResultMixin:
        """The logic how and to what the compute method is applied, based on the config and the Dataset."""
        _result_type: Type[ResultMixin] = _typestring2type(self._result_type)
        result_config = DefaultResultConfig(
            dataset_before=dataset, analyzer=self, dtype=self._result_type
        )
        if self.strategy == DispatchStrategy.ITER_STACK:  # more cases to follow
            results, piece_ids = [], []
            for feature in dataset.iter_feature(self.analyzed_feature):
                piece_ids.append(feature.identifier.piece_id)
                eligible, message = self.check(feature)
                if not eligible:
                    logger.info(f"{feature.identifier} not eligible: {message}")
                    continue
                raw_result = self.compute(
                    feature=self.pre_process(feature), **asdict(self.config)
                )
                results.append(
                    _result_type.from_raw(
                        result=raw_result,
                        config=result_config,
                        identifier=feature.identifier,
                    )
                )
            identifier = PieceStackIdentifier(piece_index=PieceIndex(piece_ids))
            return StackedResult.from_list(
                results, configuration=result_config, identifier=identifier
            )
        if self.strategy == DispatchStrategy.GROUPBY_APPLY:
            stacked_feature = self.pre_process(
                dataset.get_feature(self.analyzed_feature)
            )
            results = self.groupby_apply(stacked_feature)
            return StackedResult.from_df(df=results, configuration=result_config)
        raise ValueError(f"Unknown dispatch strategy '{self.strategy!r}'")

    @classmethod
    def _check_asserted_pipeline_steps(cls, dataset: Dataset):
        """Returns None if the check passes.

        Raises:
            ValueError: If one of the asserted PipelineSteps has not previously been applied to the Dataset.
        """
        if len(cls.assert_all) == 0:
            return True
        assert_steps = typestrings2types(cls.assert_all)
        missing = []
        for step in assert_steps:
            if not any(
                isinstance(previous_step, step)
                for previous_step in dataset.pipeline_steps
            ):
                missing.append(step)
        if len(missing) > 0:
            missing_names = ", ".join(m.__name__ for m in missing)
            raise ValueError(
                f"Applying a {cls.name} requires previous application of: {missing_names}."
            )

    @classmethod
    def _check_excluded_pipeline_steps(cls, dataset: Dataset):
        """Returns None if the check passes.

        Raises:
            ValueError: If any of the PipelineSteps applied to the Dataset matches one of the ones excluded.
        """
        if len(cls.excluded_steps) == 0:
            return
        excluded_steps = typestrings2types(cls.excluded_steps)
        excluded = []
        for step in excluded_steps:
            if any(
                isinstance(previous_step, step)
                for previous_step in dataset.pipeline_steps
            ):
                excluded.append(step)
        if len(excluded) > 0:
            excluded_names = ", ".join(e.__name__ for e in excluded)
            raise ValueError(f"{cls.name} cannot be applied after {excluded_names}.")

    def process_data(self, dataset: Dataset) -> AnalyzedData:
        """Returns an :obj:`AnalyzedData` copy of the Dataset with the added analysis result."""
        self._check_asserted_pipeline_steps(dataset)
        self._check_excluded_pipeline_steps(dataset)
        new_dataset = AnalyzedData(dataset)
        stacked_result = self.dispatch(dataset)
        stacked_result = self.post_process(stacked_result)
        new_dataset.set_result(stacked_result)
        return new_dataset

    def pre_process(
        self, feature: Union[Feature, StackedFeature]
    ) -> Union[Feature, StackedFeature]:
        """Whatever needs to be done before analyzing the feature, e.g. transforming it based on
        the config. The method needs to work both on a Feature and a StackedFeature.
        """
        return feature

    def post_process(self, result):
        """Whatever needs to be done after analyzing the data before passing it to the dataset."""
        if self.orientation == Orientation.WIDE:
            return result.unstack(fill_value=self.fill_na)
        else:
            return result


@dataclass(frozen=True)
class ResultConfig(Configuration):
    dataset_before: Dataset
    analyzer: Analyzer
    dtype: ResultName
    concat_method: Callable[[Dict[PieceID, Result], Sequence[str]], Result]


@dataclass(frozen=True)
class DefaultResultConfig(ResultConfig):
    dataset_before: Dataset
    analyzer: Analyzer
    dtype: ResultName = ResultName.StackedResult
    concat_method: Callable[[Dict[PieceID, Result], Sequence[str]], Result] = pd.concat


@dataclass(frozen=True)
class ResultID(ResultConfig):
    """Fields serving to identify one particular result."""

    identifier: Union[PieceStackIdentifier, PieceIdentifier]


@dataclass(frozen=True)
class ResultMixin(ABC):
    """Mixin"""

    _config_type: ClassVar[Type[ResultConfig]] = ResultConfig
    _default_config_type: ClassVar[Type[DefaultResultConfig]] = DefaultResultConfig
    _id_type: ClassVar[Type[ResultID]] = ResultID
    _enum_type: ClassVar[Type[ResultName]] = ResultName


@dataclass(frozen=True)
class Result(ResultID, ResultMixin):
    pass


@dataclass(frozen=True)
class RowResult(Result, ConfiguredSeries):
    """A result taking form of a series that can be treated as a row when stacking multiple results."""

    @classmethod
    def get_default_config(cls, **kwargs) -> DefaultResultConfig:
        kwargs["dtype"] = cls.dtype
        return cls._default_config_type.from_dict(kwargs)

    @classmethod
    def from_raw(
        cls,
        result: Union[SomeSeries, WrappedSeries],
        config: Configuration,
        identifier: Optional[Configuration],
        **kwargs,
    ) -> Self:
        cfg_kwargs = cls._config_type.dict_from_dataclass(config, **kwargs)
        return cls.from_series(series=result, identifier=identifier, **cfg_kwargs)


@dataclass(frozen=True)
class StackedResult(Stack, ConfiguredDataframe, ResultMixin):
    pass
    # @classmethod
    # def from_list_of_series(
    #         cls,
    #         list_of_series: List[Union[SomeSeries, WrappedSeries, ConfiguredSeries]],
    #         configuration: Optional[Union[ResultConfig]],
    #         identifier: Optional[PieceStackIdentifier] = None,
    # ) -> Self:
    #     rows = list(map(lambda S: S.to_frame().T, list_of_series))
    #     return super().from_list(list_of_dataframes=rows,
    #                              configuration=configuration,
    #                              identifier=identifier)


#
# @dataclass
# class Result(Data):
#     """Represents the result of an Analyzer processing a :class:`_Dataset`"""
#
#     def __init__(
#         self,
#         analyzer: "Analyzer",  # noqa: F821
#         dataset_before: Dataset,
#         dataset_after: AnalyzedData = None,
#         **kwargs,
#     ):
#         super().__init__(**kwargs)
#         self.analyzer = analyzer
#         self.dataset_before = dataset_before
#         self.dataset_after = dataset_after
#         self.config: dict = {}
#         self.result_dict: dict = {}
#         self._concatenated_results = None
#
#     def _concat_results(
#         self,
#         index_result_dict: Optional[dict] = None,
#         level_names: Optional[Union[Tuple[str], str]] = None,
#     ) -> SomeDataframe:
#         config2piece_results: Dict[
#             Configuration, Dict[PieceID, WrappedDataframe]
#         ] = defaultdict(dict)
#         for piece_id, piece_result in self.result_dict.items():
#             config2piece_results[piece_result.config][piece_id] = piece_result
#         if len(config2piece_results) > 1:
#             raise NotImplementedError(
#                 f"Currently, results with diverging configs cannot be concatenated:\n"
#                 f"{set(config2piece_results.keys())}"
#             )
#         concatenated_per_config = []
#         for config, piece_results in config2piece_results.items():
#             concatenated_per_config.append(
#                 config.concat_method(piece_results, names=["corpus", "piece"])
#             )
#         result = concatenated_per_config[0]
#         return result
#
#     def get_results(self):
#         return self._concat_results()
#
#     def get_group_results(self):
#         group_results = dict(self.iter_group_results())
#         level_names = tuple(self.dataset_after.index_levels["groups"])
#         if len(level_names) == 0:
#             level_names = "group"
#         return self._concat_results(group_results, level_names=level_names)
#
#     def _aggregate_results_by_ids(self, indices: Iterable[SomeID]):
#         group_results = [
#             self.result_dict[idx] for idx in indices if idx in self.result_dict
#         ]
#         if len(group_results) == 0:
#             return
#         aggregated = reduce(self.analyzer.aggregate, group_results)
#         return aggregated
#
#     def _get_aggregated_result_for_group(self, idx: GroupID):
#         indices = self.dataset_after.grouped_indices[idx]
#         return self._aggregate_results_by_ids(indices)
#
#     def items(self):
#         yield from self.result_dict.items()
#
#     def iter_results(self):
#         yield from self.result_dict.values()
#
#     def iter_group_results(self):
#         if isinstance(self.dataset_after, GroupedData):
#             for group, indices in self.dataset_after.iter_grouped_indices():
#                 aggregated = self._aggregate_results_by_ids(indices)
#                 if aggregated is None:
#                     logger.warning(
#                         f"{self.analyzer.name} yielded no result for group {group}"
#                     )
#                     continue
#                 yield group, aggregated
#         else:
#             aggregated = self._aggregate_results_by_ids(self.iter_results())
#             yield self.dataset_before.name, aggregated
#
#     def __copy__(self):
#         new_obj = self.__class__(
#             analyzer=self.analyzer,
#             dataset_before=self.dataset_before,
#             dataset_after=self.dataset_after,
#         )
#         for k, v in self.__dict__.items():
#             if k not in ["analyzer", "dataset_before", "dataset_after"]:
#                 setattr(new_obj, k, copy.copy(v))
#         return new_obj
#
#     def __deepcopy__(self, memodict={}):
#         new_obj = self.__class__(
#             analyzer=self.analyzer,
#             dataset_before=self.dataset_before,
#             dataset_after=self.dataset_after,
#         )
#         for k, v in self.__dict__.items():
#             if k not in ["analyzer", "dataset_before", "dataset_after"]:
#                 setattr(new_obj, k, copy.deepcopy(v, memodict))
#         return new_obj
#
#     def __setitem__(self, key, value):
#         self.result_dict[key] = value
#
#     def __getitem__(self, item):
#         if item in self.result_dict:
#             return self.result_dict[item]
#         return self._get_aggregated_result_for_group[item]
#
#     def __len__(self):
#         return len(self.result_dict)
#
#     def __repr__(self):
#         name = f"{self.analyzer.name} of {self.dataset_before.name}"
#         name += "\n" + "-" * len(name)
#         n_results = f"{len(self)} results"
#         if len(self.config) > 0:
#             config = pretty_dict(
#                 self.config, heading_key="config", heading_value="value"
#             )
#         else:
#             config = ""
#         return "\n\n".join((name, n_results, config))
#
#     def _repr_html_(self):
#         return self._concat_results().to_html()


if __name__ == "__main__":
    a = Analyzer(dtype="Analyzer", analyzed_feature="Harmonies", _result_type="Result")
    print(a.config)
    dataset = Dataset()
    dataset.load("~/corelli")
    analyzed = a.process_data(dataset)
    print(analyzed.get_results())

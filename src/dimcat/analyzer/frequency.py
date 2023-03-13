from dataclasses import dataclass
from typing import ClassVar, Type

from dimcat.analyzer.base import (
    Analyzer,
    AnalyzerName,
    DispatchStrategy,
    ResultName,
    UnitOfAnalysis,
)
from dimcat.data.facet import FeatureName
from dimcat.dtypes.base import Configuration, WrappedSeries


@dataclass(frozen=True)
class CounterConfig(Configuration):
    analyzed_feature: FeatureName
    result_type: ResultName
    dtype: AnalyzerName
    strategy: DispatchStrategy
    smallest_unit: UnitOfAnalysis


@dataclass(frozen=True)
class DefaultCounterConfig(CounterConfig):
    analyzed_feature: FeatureName
    result_type: ResultName = ResultName.Result
    dtype: AnalyzerName = AnalyzerName.Counter
    strategy: DispatchStrategy = DispatchStrategy.GROUPBY_APPLY
    smallest_unit: UnitOfAnalysis = UnitOfAnalysis.SLICE


@dataclass(frozen=True)
class CounterID(CounterConfig):
    """Fields serving to identify one particular counter."""

    pass


@dataclass(frozen=True)
class Counter(Analyzer, CounterID):
    _config_type: ClassVar[Type[CounterConfig]] = CounterConfig
    _default_config_type: ClassVar[Type[DefaultCounterConfig]] = DefaultCounterConfig
    _id_type: ClassVar[Type[CounterID]] = CounterID

    @staticmethod
    def compute(feature: WrappedSeries, **kwargs) -> WrappedSeries:
        return feature.value_counts()


if __name__ == "__main__":
    from dimcat import Dataset

    a = Counter.from_default(
        analyzed_feature=FeatureName.TPC, strategy=DispatchStrategy.ITER_STACK
    )
    print(a.identifier)
    dataset = Dataset()
    dataset.load("~/corelli")
    analyzed = a.process_data(dataset)
    print(analyzed.get_results())

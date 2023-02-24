from dataclasses import dataclass
from typing import ClassVar, Type

from dimcat.analyzer.base import Analyzer, AnalyzerName, ResultName
from dimcat.data.facet import FeatureName
from dimcat.dtypes import Configuration
from dimcat.dtypes.base import WrappedSeries


@dataclass(frozen=True)
class CounterConfig(Configuration):
    analyzed_feature: FeatureName
    result_type: ResultName
    dtype: AnalyzerName


@dataclass(frozen=True)
class DefaultCounterConfig(CounterConfig):
    analyzed_feature: FeatureName
    result_type: ResultName = ResultName.Result
    dtype: AnalyzerName = AnalyzerName.Counter


@dataclass(frozen=True)
class CounterID(CounterConfig):
    """Fields serving to identify one particular counter."""

    pass


@dataclass(frozen=True)
class Counter(Analyzer, CounterID):
    config_type: ClassVar[Type[CounterConfig]] = CounterConfig
    default_config_type: ClassVar[Type[DefaultCounterConfig]] = DefaultCounterConfig
    id_type: ClassVar[Type[CounterID]] = CounterID

    @staticmethod
    def compute(feature: WrappedSeries, **kwargs) -> WrappedSeries:
        return feature.value_counts()


if __name__ == "__main__":
    from dimcat import Dataset

    a = Counter.from_default(analyzed_feature=FeatureName.TPC)
    print(a.identifier)
    dataset = Dataset()
    dataset.load("~/corelli")
    analyzed = a.process_data(dataset)
    print(analyzed.get_results())

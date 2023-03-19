from dataclasses import dataclass
from typing import ClassVar, Type, Union

from dimcat.analyzer.base import (
    Analyzer,
    AnalyzerConfig,
    AnalyzerName,
    DefaultAnalyzerConfig,
    DispatchStrategy,
)
from dimcat.base import Stack
from dimcat.data.facet import Feature, FeatureName, StackedFeature
from dimcat.dtypes.base import WrappedSeries


@dataclass(frozen=True)
class CounterConfig(AnalyzerConfig):
    pass


@dataclass(frozen=True)
class DefaultCounterConfig(CounterConfig, DefaultAnalyzerConfig):
    dtype: AnalyzerName = AnalyzerName.Counter


@dataclass(frozen=True)
class Counter(CounterConfig, Analyzer):
    _config_type: ClassVar[Type[CounterConfig]] = CounterConfig
    _default_config_type: ClassVar[Type[DefaultCounterConfig]] = DefaultCounterConfig
    _id_type: ClassVar[Type[CounterConfig]] = CounterConfig

    @staticmethod
    def compute(feature: WrappedSeries, **kwargs) -> WrappedSeries:
        return feature.value_counts()

    def pre_process(
        self, feature: Union[Feature, StackedFeature]
    ) -> Union[Feature, StackedFeature]:
        if isinstance(feature, Stack):
            n_columns = feature.configuration.n_columns
        else:
            n_columns = feature.n_columns
        if n_columns == 1:
            processed = feature.iloc[:, -1]
        else:
            columns = feature.iloc[:, -n_columns:]
            processed = list(map(tuple, columns.values))
        return processed

    def post_process(self, result):
        renamed = result.rename("absolute_count")
        return super().post_process(renamed)


if __name__ == "__main__":
    from dimcat import Dataset

    a = Counter.from_default(
        analyzed_feature=FeatureName.TPC, strategy=DispatchStrategy.ITER_STACK
    )
    print(a.ID)
    dataset = Dataset()
    dataset.load("~/corelli")
    analyzed = a.process_data(dataset)
    print(analyzed.get_results())

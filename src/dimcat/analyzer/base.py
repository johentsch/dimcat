"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
from __future__ import annotations

import copy
import logging
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from enum import Enum
from functools import reduce
from typing import Any, ClassVar, Dict, Iterable, Optional, Tuple, Type, TypeVar, Union

from dimcat.base import Data, PipelineStep
from dimcat.data import AnalyzedData, Dataset, GroupedData
from dimcat.data.facet import FeatureName
from dimcat.dtypes import Configuration, GroupID, PieceID, SomeID, WrappedDataframe
from dimcat.dtypes.base import SomeDataframe, SomeFeature
from dimcat.utils import typestrings2types
from ms3 import pretty_dict
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


@dataclass(frozen=True)
class AnalyzerConfig(Configuration):
    dtype: AnalyzerName
    analyzed_feature: FeatureName
    result_type: ResultName


@dataclass(frozen=True)
class DefaultAnalyzerConfig(AnalyzerConfig):
    dtype: AnalyzerName
    analyzed_feature: FeatureName
    result_type: ResultName


@dataclass(frozen=True)
class AnalyzerID(AnalyzerConfig):
    """Fields serving to identify one particular analyzer."""

    pass


@dataclass(frozen=True)
class Analyzer(PipelineStep, AnalyzerID):
    """Analyzers are PipelineSteps that process data and store the results in Data.processed.
    The base class performs no analysis, instantiating it serves mere testing purpose.
    """

    assert_all: ClassVar[Tuple[str]] = tuple()
    """Each of these :obj:`PipelineSteps <.PipelineStep>` needs to be matched by at least one PipelineStep previously
     applied to the :obj:`.Dataset`, otherwise :meth:`process_data` raises a ValueError."""

    # assert_previous_step: ClassVar[Tuple[str]] = tuple()
    # """Analyzer.process_data() raises ValueError if last :obj:`PipelineStep` applied to the
    # :obj:`_Dataset` does not match any of these types."""

    excluded_steps: ClassVar[Tuple[str]] = tuple()
    """:meth:`process_data` raises ValueError if any of the previous :obj:`PipelineStep` applied to the
    :obj:`.Dataset` matches one of these types."""

    config_type: ClassVar[Type[AnalyzerConfig]] = AnalyzerConfig
    default_config_type: ClassVar[Type[DefaultAnalyzerConfig]] = DefaultAnalyzerConfig
    id_type: ClassVar[Type[AnalyzerID]] = AnalyzerID

    @property
    def config(self) -> AnalyzerConfig:
        return self.config_type.from_dataclass(self)

    @property
    def identifier(self) -> AnalyzerID:
        return self.id_type.from_dataclass(self)

    @classmethod
    @property
    def dtype(cls) -> AnalyzerName:
        """Name of the class as enum member."""
        return AnalyzerName(cls.name)

    @staticmethod
    def aggregate(result_a: R, result_b: R) -> R:
        """Static method that combines two results of :meth:`compute`.

        This needs to be equivalent to calling self.compute on the concatenation of the respective data resulting
        in the two arguments."""
        pass

    @staticmethod
    def compute(feature: SomeFeature, **kwargs) -> SomeFeature:
        """Static method that performs the actual computation takes place."""
        return feature

    def dispatch(self, dataset: Dataset) -> Result:
        """The logic how and to what the compute method is applied, based on the config and the Dataset."""
        result_type = _typestring2type(self.result_type)
        result_object = result_type(analyzer=self, dataset_before=dataset)
        config_dict = asdict(self.config)
        if True:  # more cases to follow
            for feature in dataset.iter_facet(self.analyzed_feature):
                eligible, message = self.check(feature)
                if not eligible:
                    logger.info(f"{feature.identifier} not eligible: {message}")
                    continue
                result_object[feature.identifier] = self.compute(
                    feature=feature, **config_dict
                )
        return result_object

    @classmethod
    def from_config(
        cls,
        config: AnalyzerConfig,
        identifiers: Any = None,
        **kwargs,
    ) -> Self:
        """"""
        cfg_kwargs = cls.config_type.dict_from_dataclass(config)
        cfg_kwargs.update(kwargs)
        if identifiers is not None:
            logger.warning(
                "Analyzers currently do not come with particular identifiers."
            )
        if cfg_kwargs["dtype"] != cls.dtype:
            cfg_class = config.__class__.__name__
            raise TypeError(
                f"Cannot initiate {cls.name} with {cfg_class}.dtype={config.dtype}."
            )
        return cls(**cfg_kwargs)

    @classmethod
    def from_default(
        cls,
        identifiers: Any = None,
        **kwargs,
    ) -> Self:
        """"""
        if len(kwargs) > 0:
            cfg_field_names = [fld.name for fld in fields(cls.config_type)]
            cfg_kwargs = {
                kw: arg for kw, arg in kwargs.items() if kw in cfg_field_names
            }
            config = cls.get_default_config(**cfg_kwargs)
            if len(kwargs) > len(cfg_kwargs):
                id_kwargs = {
                    kw: arg for kw, arg in kwargs.items() if kw not in cfg_field_names
                }
                return cls.from_config(
                    config=config, identifiers=identifiers, **id_kwargs
                )
            return cls.from_config(config=config, identifiers=identifiers)
        else:
            config = cls.get_default_config()
            cls.from_config(config=config, identifiers=identifiers)

    @classmethod
    def from_id(cls, identifier: AnalyzerID):
        kwargs = cls.id_type.dict_from_dataclass(identifier)
        return cls(**kwargs)

    @classmethod
    def get_default_config(cls, **kwargs) -> DefaultAnalyzerConfig:
        return cls.default_config_type(dtype=cls.dtype, **kwargs)

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
        result_object = self.dispatch(dataset)
        result_object = self.post_process(result_object)
        new_dataset.set_result(result_object)
        return new_dataset

    def post_process(self, processed):
        """Whatever needs to be done after analyzing the data before passing it to the dataset."""
        return processed


@dataclass(frozen=True)
class ResultConfig(Configuration):
    dataset_before: Dataset
    analyzer: Analyzer
    dtype: ResultName


@dataclass(frozen=True)
class DefaultResultConfig(ResultConfig):
    dataset_before: Dataset
    analyzer: Analyzer
    dtype: ResultName = ResultName.Result


@dataclass(frozen=True)
class ResultID(ResultConfig):
    """Fields serving to identify one particular analyzer."""

    pass


@dataclass
class Result(Data):
    """Represents the result of an Analyzer processing a :class:`_Dataset`"""

    def __init__(
        self,
        analyzer: "Analyzer",  # noqa: F821
        dataset_before: Dataset,
        dataset_after: AnalyzedData = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.analyzer = analyzer
        self.dataset_before = dataset_before
        self.dataset_after = dataset_after
        self.config: dict = {}
        self.result_dict: dict = {}
        self._concatenated_results = None

    def _concat_results(
        self,
        index_result_dict: Optional[dict] = None,
        level_names: Optional[Union[Tuple[str], str]] = None,
    ) -> SomeDataframe:
        config2piece_results: Dict[
            Configuration, Dict[PieceID, WrappedDataframe]
        ] = defaultdict(dict)
        for piece_id, piece_result in self.result_dict.items():
            config2piece_results[piece_result.config][piece_id] = piece_result
        if len(config2piece_results) > 1:
            raise NotImplementedError(
                f"Currently, results with diverging configs cannot be concatenated:\n"
                f"{set(config2piece_results.keys())}"
            )
        concatenated_per_config = []
        for config, piece_results in config2piece_results.items():
            concatenated_per_config.append(
                config.concat_method(piece_results, names=["corpus", "piece"])
            )
        result = concatenated_per_config[0]
        return result

    def get_results(self):
        return self._concat_results()

    def get_group_results(self):
        group_results = dict(self.iter_group_results())
        level_names = tuple(self.dataset_after.index_levels["groups"])
        if len(level_names) == 0:
            level_names = "group"
        return self._concat_results(group_results, level_names=level_names)

    def _aggregate_results_by_ids(self, indices: Iterable[SomeID]):
        group_results = [
            self.result_dict[idx] for idx in indices if idx in self.result_dict
        ]
        if len(group_results) == 0:
            return
        aggregated = reduce(self.analyzer.aggregate, group_results)
        return aggregated

    def _get_aggregated_result_for_group(self, idx: GroupID):
        indices = self.dataset_after.grouped_indices[idx]
        return self._aggregate_results_by_ids(indices)

    def items(self):
        yield from self.result_dict.items()

    def iter_results(self):
        yield from self.result_dict.values()

    def iter_group_results(self):
        if isinstance(self.dataset_after, GroupedData):
            for group, indices in self.dataset_after.iter_grouped_indices():
                aggregated = self._aggregate_results_by_ids(indices)
                if aggregated is None:
                    logger.warning(
                        f"{self.analyzer.name} yielded no result for group {group}"
                    )
                    continue
                yield group, aggregated
        else:
            aggregated = self._aggregate_results_by_ids(self.iter_results())
            yield self.dataset_before.name, aggregated

    def __copy__(self):
        new_obj = self.__class__(
            analyzer=self.analyzer,
            dataset_before=self.dataset_before,
            dataset_after=self.dataset_after,
        )
        for k, v in self.__dict__.items():
            if k not in ["analyzer", "dataset_before", "dataset_after"]:
                setattr(new_obj, k, copy.copy(v))
        return new_obj

    def __deepcopy__(self, memodict={}):
        new_obj = self.__class__(
            analyzer=self.analyzer,
            dataset_before=self.dataset_before,
            dataset_after=self.dataset_after,
        )
        for k, v in self.__dict__.items():
            if k not in ["analyzer", "dataset_before", "dataset_after"]:
                setattr(new_obj, k, copy.deepcopy(v, memodict))
        return new_obj

    def __setitem__(self, key, value):
        self.result_dict[key] = value

    def __getitem__(self, item):
        if item in self.result_dict:
            return self.result_dict[item]
        return self._get_aggregated_result_for_group[item]

    def __len__(self):
        return len(self.result_dict)

    def __repr__(self):
        name = f"{self.analyzer.name} of {self.dataset_before.name}"
        name += "\n" + "-" * len(name)
        n_results = f"{len(self)} results"
        if len(self.config) > 0:
            config = pretty_dict(
                self.config, heading_key="config", heading_value="value"
            )
        else:
            config = ""
        return "\n\n".join((name, n_results, config))

    def _repr_html_(self):
        return self._concat_results().to_html()


if __name__ == "__main__":
    a = Analyzer(dtype="Analyzer", analyzed_feature="Harmonies", result_type="Result")
    print(a.config)
    dataset = Dataset()
    dataset.load("~/corelli")
    analyzed = a.process_data(dataset)
    print(analyzed.get_results())

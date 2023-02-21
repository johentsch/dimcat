"""Analyzers are PipelineSteps that process data and store the results in Data.processed."""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from enum import Enum
from typing import ClassVar, Iterable, Tuple, Type, TypeVar, Union

import dimcat.data as data_module
from dimcat.base import PipelineStep
from dimcat.data import AnalyzedData, Dataset
from dimcat.dtypes import Configuration
from dimcat.utils import typestrings2types

logger = logging.getLogger(__name__)


def _typestring2type(typestring: str) -> Type:
    return getattr(data_module, typestring)


R = TypeVar("R")


class AnalyzerName(str, Enum):
    """Identifies the various types of data facets and makes accessible their default configs and TabularData."""

    @classmethod
    def make_tuple(
        cls, facets: Iterable[Union[AnalyzerName, str]]
    ) -> Tuple[AnalyzerName]:
        return tuple(cls(c) for c in facets)

    TPCrange = "TPCrange"
    PitchClassVectors = "PitchClassVectors"
    ChordSymbolUnigrams = "ChordSymbolUnigrams"
    ChordSymbolBigrams = "ChordSymbolBigrams"


@dataclass(frozen=True)
class AnalyzerConfig(Configuration):
    dtype: AnalyzerName
    analyzed_aspect: str
    result_type: str


@dataclass(frozen=True)
class DefaultAnalyzerConfig(AnalyzerConfig):
    dtype: AnalyzerName
    analyzed_aspect: str
    result_type: str


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

    @property
    def config(self) -> AnalyzerConfig:
        return AnalyzerConfig.from_dataclass(self)

    @property
    def identifier(self) -> AnalyzerID:
        return AnalyzerID.from_dataclass(self)

    @staticmethod
    def aggregate(result_a: R, result_b: R) -> R:
        """Static method that combines two results of :meth:`compute`.

        This needs to be equivalent to calling self.compute on the concatenation of the respective data resulting
        in the two arguments."""
        pass

    @staticmethod
    def compute(aspect, **kwargs) -> aspect:
        """Static method that performs the actual computation takes place."""
        return aspect

    def dispatch(self, dataset: Dataset) -> Result:
        """The logic how and to what the compute method is applied, based on the config and the Dataset."""
        result_type = _typestring2type(self.result_type)
        result_object = result_type(analyzer=self, dataset_before=dataset)
        config_dict = asdict(self.config)
        if True:  # more cases to follow
            for aspect in dataset.iter_facet(self.analyzed_aspect):
                eligible, message = self.check(aspect)
                if not eligible:
                    logger.info(f"{aspect.identifier} not eligible: {message}")
                    continue
                result_object[aspect.piece_id] = self.compute(
                    aspect=aspect, **config_dict
                )
        return result_object

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


if __name__ == "__main__":
    a = Analyzer(dtype="Analyzer", analyzed_aspect="Harmonies", result_type="Result")
    dataset = Dataset()
    dataset.load("~/corelli")
    analyzed = a.process_data(dataset)
    print(analyzed.get_results())

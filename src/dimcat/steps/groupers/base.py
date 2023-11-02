import logging
from collections import defaultdict
from numbers import Number
from typing import (
    Dict,
    Hashable,
    Iterator,
    List,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
)

import marshmallow as mm
import pandas as pd
from dimcat.base import is_subclass_of
from dimcat.data.datasets.base import Dataset
from dimcat.data.datasets.processed import GroupedDataset
from dimcat.data.resources.dc import DimcatIndex, DimcatResource, PieceIndex
from dimcat.data.resources.features import Feature
from dimcat.dc_exceptions import GrouperNotSetUpError, ResourceNotProcessableError
from dimcat.steps.base import FeatureProcessingStep
from dimcat.utils import check_name, get_middle_composition_year
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Grouper(FeatureProcessingStep):
    # inherited from PipelineStep:
    new_dataset_type = GroupedDataset
    new_resource_type = None  # same as input
    applicable_to_empty_datasets = True
    # inherited from FeatureProcessingStep:
    allowed_features = None  # any
    output_package_name = None  # transform 'features'
    requires_at_least_one_feature = False

    class Schema(FeatureProcessingStep.Schema):
        level_name = mm.fields.Str()

    def __init__(self, level_name: str = "grouper", **kwargs):
        super().__init__(**kwargs)
        self._level_name: str = None
        self.level_name = level_name

    @property
    def level_name(self) -> str:
        return self._level_name

    @level_name.setter
    def level_name(self, level_name: str):
        check_name(level_name)
        self._level_name = level_name

    def apply_grouper(self, resource: Feature) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        return pd.concat([resource.df], keys=[self.level_name], names=[self.level_name])

    def _make_new_resource(self, resource: Feature) -> Feature:
        """Apply the grouper to a Feature."""
        if self.level_name in resource.get_level_names():
            self.logger.debug(
                f"Resource {resource.resource_name!r} already has a level named {self.level_name!r}."
            )
            return resource
        result_constructor = self._get_new_resource_type(resource)
        results = self.apply_grouper(resource)
        result_name = self.resource_name_factory(resource)
        return result_constructor.from_dataframe(
            df=results,
            resource_name=result_name,
        )

    def _iter_resources(self, dataset: Dataset) -> Iterator[Tuple[str, DimcatResource]]:
        """Iterate over all resources in the dataset's OutputCatalog."""
        return dataset.outputs.iter_resources()

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Apply this PipelineStep to a :class:`Dataset` and return a copy containing the output(s)."""
        new_dataset = self._make_new_dataset(dataset)
        self.fit_to_dataset(new_dataset)
        new_dataset._pipeline.add_step(self)
        package_name_resource_iterator = self._iter_resources(new_dataset)
        processed_resources = defaultdict(list)
        for package_name, resource in package_name_resource_iterator:
            try:
                new_resource = self.process_resource(resource)
            except ResourceNotProcessableError as e:
                self.logger.warning(
                    f"Resource {resource.resource_name!r} could not be grouped and is not included in "
                    f"the new Dataset due to the following error: {e!r}"
                )
                continue
            processed_resources[package_name].append(new_resource)
        for package_name, resources in processed_resources.items():
            new_package = self._make_new_package(package_name)
            new_package.extend(resources)
            n_processed = len(resources)
            if new_package.n_resources < n_processed:
                if new_package.n_resources == 0:
                    self.logger.warning(
                        f"None of the {n_processed} {package_name} were successfully transformed."
                    )
                else:
                    self.logger.warning(
                        f"Transformation was successful only on {new_package.n_resources} of the "
                        f"{n_processed} features."
                    )
            new_dataset.outputs.replace_package(new_package)
        return new_dataset

    def _post_process_result(self, result: DimcatResource) -> DimcatResource:
        """Change the default_groupby value of the returned Feature."""
        result.update_default_groupby(self.level_name)
        return result


class CorpusGrouper(Grouper):
    def __init__(self, level_name: str = "corpus", **kwargs):
        super().__init__(level_name=level_name, **kwargs)


class CustomPieceGrouper(Grouper):
    class Schema(Grouper.Schema):
        grouped_pieces = mm.fields.Nested(DimcatIndex.Schema)

        @mm.pre_load
        def deal_with_dict(self, data, **kwargs):
            if isinstance(data["grouped_pieces"], MutableMapping):
                if "dtype" not in data["grouped_pieces"] or not is_subclass_of(
                    data["grouped_pieces"]["dtype"], DimcatIndex
                ):
                    # dealing with a manually compiled DimcatConfig where grouped_pieces are a grouping dict
                    grouped_pieces = DimcatIndex.from_grouping(data["grouped_pieces"])
                    data["grouped_pieces"] = grouped_pieces.to_dict()
            return data

    @classmethod
    def from_grouping(
        cls,
        piece_groups: Dict[Hashable, List[tuple]],
        level_names: Sequence[str] = ("piece_group", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a CustomPieceGrouper from a dictionary of piece groups.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names:
            Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set True to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        grouped_pieces = PieceIndex.from_grouping(
            grouping=piece_groups,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )
        return cls(level_name=grouped_pieces.names[0], grouped_pieces=grouped_pieces)

    def __init__(
        self,
        level_name: str = "piece_group",
        grouped_pieces: DimcatIndex | pd.MultiIndex = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, **kwargs)
        self._grouped_pieces: Optional[DimcatIndex] = None
        if grouped_pieces is not None:
            self.grouped_pieces = grouped_pieces

    @property
    def grouped_pieces(self) -> DimcatIndex:
        if self._grouped_pieces is None:
            return DimcatIndex.from_tuples([], (self.level_name, "corpus", "piece"))
        return self._grouped_pieces

    @grouped_pieces.setter
    def grouped_pieces(self, grouped_pieces: DimcatIndex):
        if isinstance(grouped_pieces, pd.Index):
            grouped_pieces = DimcatIndex(grouped_pieces)
        elif isinstance(grouped_pieces, dict):
            raise TypeError(
                f"Use {self.name}.from_dict() to create a {self.name}from a dictionary."
            )
        elif not isinstance(grouped_pieces, DimcatIndex):
            raise TypeError(f"Expected DimcatIndex, got {type(grouped_pieces)}")
        if grouped_pieces.names[-1] != "piece":
            raise ValueError(
                f"Expected last level to to be named 'piece', not {grouped_pieces.names[-1]}"
            )
        self._grouped_pieces = grouped_pieces

    def apply_grouper(self, resource: Feature) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        return resource.align_with_grouping(self.grouped_pieces)

    def check_resource(self, resource: DimcatResource) -> None:
        if len(self.grouped_pieces) == 0:
            raise GrouperNotSetUpError(self.dtype)
        super().check_resource(resource)


class YearGrouper(CustomPieceGrouper):
    @classmethod
    def from_grouping(
        cls,
        piece_groups: Dict[Number, List[tuple]],
        level_names: Sequence[str] = ("middle_composition_year", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a YearGrouper from a dictionary of piece groups.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        level_names:
            Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set True to disable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        return super().from_grouping(
            piece_groups=piece_groups,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )

    def __init__(
        self,
        level_name: str = "middle_composition_year",
        grouped_pieces: DimcatIndex | pd.MultiIndex = None,
        **kwargs,
    ):
        super().__init__(level_name=level_name, grouped_pieces=grouped_pieces, **kwargs)

    def fit_to_dataset(self, dataset: Dataset) -> None:
        metadata = dataset.get_metadata()
        sorted_composition_years = get_middle_composition_year(metadata).sort_values()
        grouping = sorted_composition_years.groupby(
            sorted_composition_years, sort=True
        ).groups
        group_index = DimcatIndex.from_grouping(
            grouping, ("middle_composition_year", "corpus", "piece")
        )
        self.grouped_pieces = group_index

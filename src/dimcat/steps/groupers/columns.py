from typing import Optional

import marshmallow as mm
import pandas as pd
from dimcat.data.resources import DimcatResource, Feature, FeatureName
from dimcat.steps.groupers.base import Grouper


class ColumnGrouper(Grouper):
    class Schema(Grouper.Schema):
        grouped_column = mm.fields.Str()

    def __init__(
        self,
        grouped_column: str,
        level_name: Optional[str] = None,
        **kwargs,
    ):
        if level_name is None:
            level_name = grouped_column
        super().__init__(level_name=level_name, **kwargs)
        self.grouped_column: str = grouped_column

    def check_resource(self, resource: DimcatResource) -> None:
        super().check_resource(resource)
        if self.grouped_column not in resource.df.columns:
            raise ValueError(
                f"Expected {self.grouped_column} column in {resource.name}"
            )

    def apply_grouper(self, resource: Feature) -> pd.DataFrame:
        """Apply the grouper to a Feature."""
        resource_df = resource.df
        grouped_df = pd.concat(
            dict(resource_df.groupby(self.grouped_column).__iter__()),
            names=[self.level_name],
        )
        # this is equivalent to but slightly faster than:
        # grouped_df = resource_df.groupby(self.grouped_column).apply(lambda df: df) <= index level would need renaming
        return grouped_df


class MeasureGrouper(ColumnGrouper):
    def __init__(
        self,
        grouped_column: str = "mn",
        level_name: str = "measure",
        **kwargs,
    ):
        super().__init__(grouped_column=grouped_column, level_name=level_name, **kwargs)


class ModeGrouper(ColumnGrouper):
    allowed_features = (FeatureName.HarmonyLabels, FeatureName.KeyAnnotations)

    def __init__(
        self,
        grouped_column: str = "localkey_mode",
        level_name: str = "mode",
        **kwargs,
    ):
        super().__init__(grouped_column=grouped_column, level_name=level_name, **kwargs)

from typing import Iterator, List, Literal, Tuple

import pandas as pd
from dimcat.analyzer.base import Analyzer
from dimcat.data import AnalyzedGroupedDataset, AnalyzedSlicedDataset
from dimcat.dtypes import SomeID


class SliceInfoAnalyzer(Analyzer):
    """"""

    assert_all = ["Slicer"]

    def check(self, df):
        if len(df) == 0:
            return False, "DataFrame is empty."
        return True, ""

    def data_iterator(
        self, data: AnalyzedSlicedDataset
    ) -> Iterator[Tuple[Literal["all_slices"], pd.DataFrame]]:
        yield from [("all_slices", data.get_slice_info())]


class GroupedSliceInfoAnalyzer(SliceInfoAnalyzer):
    """"""

    assert_all = ["Slicer", "Grouper"]

    def data_iterator(
        self, data: AnalyzedGroupedDataset
    ) -> Iterator[Tuple[SomeID, pd.DataFrame]]:
        yield from data.iter_grouped_slice_info()


class LocalKeySequence(GroupedSliceInfoAnalyzer):
    """"""

    assert_all = ["LocalKeySlicer", "PieceGrouper"]

    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        self.level_names["processed"] = ["localkeys"]
        self.group2pandas = None

    @staticmethod
    def aggregate(result_a: pd.Series, result_b: pd.Series) -> pd.Series:
        return pd.concat(result_a, result_b)

    @staticmethod
    def compute(slice_info: pd.DataFrame) -> pd.Series:
        return slice_info.localkey


class LocalKeyUnique(SliceInfoAnalyzer):
    """"""

    assert_all = ["LocalKeySlicer"]

    def __init__(self, **kwargs):
        """"""
        super().__init__(**kwargs)
        self.level_names["processed"] = ["localkeys"]
        self.group2pandas = None

    @staticmethod
    def aggregate(result_a: List[str], result_b: List[str]) -> List[str]:
        set_a = set(result_a)
        set_a.update(result_b)
        return sorted(set_a)

    @staticmethod
    def compute(slice_info: pd.DataFrame) -> List[str]:
        return list(slice_info.localkey.unique())

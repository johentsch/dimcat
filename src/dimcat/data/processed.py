from __future__ import annotations

from collections import defaultdict
from functools import lru_cache
from typing import (
    TYPE_CHECKING,
    Collection,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import ms3
import pandas as pd
from dimcat.base import Data
from dimcat.data.base import Dataset, logger
from dimcat.dtypes import GroupID, PieceID, SliceID, SomeID
from dimcat.dtypes.base import SomeDataframe
from dimcat.utils import clean_index_levels, typestrings2types
from ms3._typing import ScoreFacet

if TYPE_CHECKING:
    from dimcat.analyzer.base import Result


class _ProcessedDataMixin(Data):
    """Base class for types of processed :obj:`_Dataset` objects.
    Processed datatypes are created by passing a _Dataset object. The new object will be a copy of the Data with the
    :attr:`prefix` prepended. Subclasses should have an __init__() method that calls super().__init__() and then
    adds additional fields.
    """

    assert_types: Union[str, Collection[str]] = ["Dataset"]
    """Objects raise TypeError upon instantiation if the passed data are not of one of these types."""
    excluded_types: Union[str, Collection[str]] = []
    """Objects raise TypeError upon instantiation if the passed data are of one of these types."""
    type_mapping: Dict[Union[str, Collection[str]], str] = {}
    """{Input type(s) -> Output type}. __new__() picks the first 'value' where the input Data are of type 'key'.
    Objects raise TypeError if nothing matches. object or Data can be used as fallback/default key.
    """

    def __new__(cls, data: Data, **kwargs):
        """Depending on the type of ``data`` (currently only :class:`Dataset` is implemented),
        the new object is turned into the Dataset subtype that corresponds to the performed processing step.

        This method uses the class properties :attr:`assert_types` and :attr:`excluded_types` to determine if the
        input Dataset can actually undergo the current type of processing. Then it uses the class property
        :attr:`type_mapping` to determine the type of the new object to be created.


        Args:
            data: Dataset to be converted into a processed subtype.
            **kwargs:
        """
        assert_types = typestrings2types(cls.assert_types)
        if not isinstance(data, assert_types):
            raise TypeError(
                f"{cls.__name__} objects can only be created from {cls.assert_types} ({assert_types}), "
                f"not '{type(data).__name__}'"
            )
        excluded_types = typestrings2types(cls.excluded_types)
        if isinstance(data, excluded_types):
            raise TypeError(
                f"{cls.__name__} objects cannot be created from '{type(data).__name__}' because it is among the "
                f"excluded_types {cls.excluded_types}."
            )
        type_mapping = {
            typestrings2types(input_type): typestrings2types(output_type)[0]
            for input_type, output_type in cls.type_mapping.items()
        }
        new_obj_type = None
        for input_type, output_type in type_mapping.items():
            if isinstance(data, input_type):
                new_obj_type = output_type
                break
        if new_obj_type is None:
            raise TypeError(
                f"{cls.__name__} no output type defined for '{type(data)}', only for {list(type_mapping.keys())}."
            )
        obj = super().__new__(new_obj_type)
        # obj.__init__(data=data, **kwargs)
        return obj

    def __init__(self, data: Data, **kwargs):
        super().__init__(data=data, **kwargs)


class AnalyzedData(_ProcessedDataMixin):
    """A type of Data object that contains the results of an Analyzer and knows how to plot it."""

    type_mapping = {
        (
            "AnalyzedGroupedSlicedDataset",
            "GroupedSlicedDataset",
        ): "AnalyzedGroupedSlicedDataset",
        ("AnalyzedSlicedDataset", "SlicedDataset"): "AnalyzedSlicedDataset",
        ("AnalyzedGroupedDataset", "GroupedDataset"): "AnalyzedGroupedDataset",
        "Dataset": "AnalyzedDataset",
    }

    def __new__(
        cls, data: Data, **kwargs
    ) -> Union[
        "AnalyzedDataset",
        "AnalyzedGroupedDataset",
        "AnalyzedSlicedDataset",
        "AnalyzedGroupedSlicedDataset",
    ]:
        return super().__new__(cls, data=data, **kwargs)

    def __init__(self, data: Data, **kwargs):
        super().__init__(data=data, **kwargs)
        self.result: Optional[Result] = None
        """Analyzers store their result here using :meth:`set_result`."""

    def set_result(self, result: Result):
        self.result = result

    def get_results(self) -> SomeDataframe:
        return self.result.get_results()

    def get_group_results(self) -> pd.DataFrame:
        return self.result.get_group_results()

    def iter_results(self):
        yield from self.result.iter_results()

    def iter_group_results(self):
        yield from self.result.iter_group_results()

    # @overload
    # def iter(
    #     self, as_pandas: bool = Literal[False], ignore_groups: bool = Literal[False]
    # ) -> Iterator[Tuple[GroupID, Union[Dict[ID, Any], Any]]]:
    #     ...
    #
    # @overload
    # def iter(
    #     self, as_pandas: bool = Literal[True], ignore_groups: bool = Literal[False]
    # ) -> Iterator[Tuple[GroupID, Union[Pandas, Any]]]:
    #     ...
    #
    # @overload
    # def iter(
    #     self, as_pandas: bool = Literal[False], ignore_groups: bool = Literal[True]
    # ) -> Iterator[Union[Tuple[ID, Any], Any]]:
    #     ...
    #
    # @overload
    # def iter(
    #     self, as_pandas: bool = Literal[True], ignore_groups: bool = Literal[True]
    # ) -> Iterator[Union[Pandas, Any]]:
    #     ...
    #
    # def iter(
    #     self, as_pandas: bool = True, ignore_groups: bool = False
    # ) -> Iterator[
    #     Union[
    #         Tuple[GroupID, Union[Dict[ID, Any], Any]],
    #         Tuple[GroupID, Union[Pandas, Any]],
    #         Union[Tuple[ID, Any], Any],
    #         Union[Pandas, Any],
    #     ]
    # ]:
    #     """Iterate through :attr:`processed` data.
    #
    #     Args:
    #         as_pandas:
    #             Setting this value to False corresponds to iterating through .processed.items(),
    #             where keys are group IDs and values are results for Analyzers that compute
    #             one result per group, or {ID -> result} dicts for Analyzers that compute
    #             one result per item per group. The default value (True) has no effect in the first case,
    #             but in the second case, the dictionary will be converted to a Series if the conversion method is
    #             set in :attr:`group2pandas`.
    #         ignore_groups:
    #             If set to True, the iteration loop is flattened and does not include group identifiers. If as_pandas
    #             is False (default), and the applied Analyzer computes one {ID -> result} dict per group,
    #             this will correspond to iterating through the (ID, result) tuples for all groups.
    #
    #     Yields:
    #         The result of the last applied Analyzer for each group or for each item of each group.
    #     """
    #     if ignore_groups and not as_pandas:
    #         raise ValueError(
    #             "If you set 'as_dict' and 'ignore_groups' are in conflict, choose one or use _.get()."
    #         )
    #     for group, result in self.result.items():
    #         if ignore_groups:
    #             if self.group2pandas is None:
    #                 yield result
    #             elif as_pandas:
    #                 yield self.convert_group2pandas(result)
    #             else:
    #                 yield from result.items()
    #         else:
    #             if as_pandas and self.group2pandas is not None:
    #                 yield group, self.convert_group2pandas(result)
    #             else:
    #                 yield group, result


class GroupedData(_ProcessedDataMixin):
    """A type of Data object that behaves like its predecessor but returns and iterates through groups."""

    type_mapping = {
        (
            "AnalyzedGroupedSlicedDataset",
            "AnalyzedSlicedDataset",
        ): "AnalyzedGroupedSlicedDataset",
        "GroupedSlicedDataset": "GroupedSlicedDataset",
        ("AnalyzedGroupedDataset", "AnalyzedDataset"): "AnalyzedGroupedDataset",
        "SlicedDataset": "GroupedSlicedDataset",
        "Dataset": "GroupedDataset",
    }

    def __new__(
        cls, data: Data, **kwargs
    ) -> Union[
        "GroupedDataset",
        "GroupedSlicedDataset",
        "AnalyzedGroupedDataset",
        "AnalyzedGroupedSlicedDataset",
    ]:
        return super().__new__(cls, data=data, **kwargs)

    def __init__(self, data: Data, **kwargs):
        logger.debug(f"{type(self).__name__} -> before {super()}.__init__()")
        super().__init__(data=data, **kwargs)
        logger.debug(f"{type(self).__name__} -> after {super()}.__init__()")
        if not hasattr(self, "grouped_indices"):
            if hasattr(data, "grouped_indices"):
                self.grouped_indices = data.grouped_indices
            else:
                self.grouped_indices: Dict[GroupID, List[SomeID]] = {(): self.indices}
                """{group_key -> indices} dictionary of indices (IDs) which serve for accessing individual pieces of
                data and associated metadata. An index is a ('corpus_name', 'piece_name') tuple ("ID")
                that can have a third element identifying a segment/chunk of a piece.
                The group_keys are an empty tuple by default; with every applied Grouper,
                the length of all group_keys grows by one and the number of group_keys grows or stays the same."""

    def iter_grouped_indices(self) -> Iterator[Tuple[str, List[SomeID]]]:
        """Iterate through groups of indices as defined by the previously applied Groupers.

        Yields
        -------
        :obj:`tuple` of :obj:`str`
            A tuple of keys reflecting the group hierarchy
        :obj:`list` of :obj:`tuple`
            A list of IDs belonging to the same group.
        """
        if len(self.indices) == 0:
            raise ValueError("No data has been loaded.")
        if any(len(index_list) == 0 for index_list in self.grouped_indices.values()):
            logger.warning("Data object contains empty groups.")
        yield from self.grouped_indices.items()

    def iter_grouped_slice_info(self) -> Iterator[Tuple[tuple, pd.DataFrame]]:
        """Iterate through concatenated slice_info DataFrame for each group."""
        for group, index_group in self.iter_grouped_indices():
            group_info = {ix: self.slice_info[ix] for ix in index_group}
            group_df = pd.concat(group_info.values(), keys=group_info.keys(), axis=1).T
            group_df.index = self._rename_multiindex_levels(
                group_df.index, self.index_levels["indices"]
            )
            yield group, group_df


class SlicedData(_ProcessedDataMixin):
    """A type of Data object that contains the slicing information created by a Slicer. It slices all requested
    facets based on that information.
    """

    excluded_types = ["AnalyzedData", "SlicedData"]
    type_mapping = {
        "GroupedDataset": "GroupedSlicedDataset",
        "Dataset": "SlicedDataset",
    }

    def __new__(
        cls, data: Data, **kwargs
    ) -> Union["SlicedDataset", "GroupedSlicedDataset"]:
        return super().__new__(cls, data=data, **kwargs)

    def __init__(self, data: Data, **kwargs):
        logger.debug(f"{type(self).__name__} -> before {super()}.__init__()")
        super().__init__(data=data, **kwargs)
        logger.debug(f"{type(self).__name__} -> after {super()}.__init__()")
        if not hasattr(self, "sliced"):
            self.sliced = {}
            """Dict for sliced data facets."""
        if not hasattr(self, "slice_info"):
            self.slice_info = {}
            """Dict holding metadata of slices (e.g. the localkey of a segment)."""

    def get_slice(self, index, what):
        if what in self.sliced and index in self.sliced[what]:
            return self.sliced[what][index]

    def get_slice_info(self) -> pd.DataFrame:
        """Concatenates slice_info Series and returns them as a DataFrame."""
        if len(self.slice_info) == 0:
            logger.info("No slices available.")
            return pd.DataFrame()
        concatenated_info = pd.concat(
            self.slice_info.values(), keys=self.slice_info.keys(), axis=1
        ).T
        concatenated_info.index.rename(self.index_levels["indices"], inplace=True)
        return concatenated_info

    def iter_slice_info(self) -> Iterator[Tuple[SliceID, pd.Series]]:
        """Iterate through concatenated slice_info Series for each group."""
        yield from self.slice_info.items()


class AnalyzedDataset(AnalyzedData, Dataset):
    pass


class GroupedDataset(GroupedData, Dataset):
    def iter_grouped_facet(
        self,
        what: ScoreFacet,
    ) -> Iterator[Tuple[GroupID, pd.DataFrame]]:
        """Iterate through one concatenated facet DataFrame per group.

        Args:
            what: Which type of facet to retrieve.

        Yields:
            Group index.
            Facet DataFrame.
        """
        for group, index_group in self.iter_grouped_indices():
            result = {}
            missing_id = []
            for index in index_group:
                df = self.get_item(index, what=what)
                if df is None or len(df.index) == 0:
                    missing_id.append(index)
                    continue
                result[index] = df
            n_results = len(result)
            if len(missing_id) > 0:
                if n_results == 0:
                    pass
                    # logger.info(f"No '{what}' available for {group}.")
                else:
                    logger.info(
                        f"Group {group} is missing '{what}' for the following indices:\n{missing_id}"
                    )
            if n_results == 0:
                continue
            if n_results == 1:
                # workaround necessary because of nasty "cannot handle overlapping indices;
                # use IntervalIndex.get_indexer_non_unique" error
                result["empty"] = pd.DataFrame()
            result = pd.concat(
                result.values(),
                keys=result.keys(),
                names=self.index_levels["indices"] + ["interval"],
            )
            yield group, result

    def set_indices(
        self, new_indices: Union[List[PieceID], Dict[SomeID, List[PieceID]]]
    ) -> None:
        """Replace :attr:`indices` with a new list of IDs and update the :attr:`grouped_indices` accordingly.

        Args:
            new_indices:
                The new list of IDs or an {old_id -> [new_id]} dictionary to replace the IDs with a list of new IDs.
        """
        id2group = defaultdict(lambda: ())
        if len(self.piece_index) > 0:
            id2group.update(
                {
                    ID: group
                    for group, group_ids in self.iter_grouped_indices()
                    for ID in group_ids
                }
            )
        new_grouped_indices = defaultdict(list)
        if isinstance(new_indices, dict):
            for old_id, new_ids in new_indices.items():
                old_group = id2group[old_id]
                new_grouped_indices[old_group].extend(new_ids)
        else:
            for new_id in new_indices:
                old_group = id2group[new_id]
                new_grouped_indices[old_group].append(new_id)
        self.grouped_indices = {
            k: new_grouped_indices[k] for k in sorted(new_grouped_indices.keys())
        }
        new_indices = sum(new_grouped_indices.values(), [])
        self.indices = sorted(new_indices)

    def set_grouped_indices(self, grouped_indices: Dict[GroupID, List[SomeID]]):
        self.grouped_indices = grouped_indices


class SlicedDataset(SlicedData, Dataset):
    @lru_cache()
    def get_item(self, ID: SliceID, what: ScoreFacet) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame pertaining to the facet ``what`` of the piece ``index``. If
        the facet has been sliced before, the sliced DataFrame is returned.

        Args:
            ID: (corpus, fname) or (corpus, fname, interval)
            what: What facet to retrieve.

        Returns:
            DataFrame representing an entire score facet, or a chunk (slice) of it.
        """
        match ID:
            case (_, _, _):
                return self.get_slice(ID, what)
            case (corpus, piece):
                return super().get_item(ID=PieceID(corpus, piece), what=what)

    def iter_facet(self, what: ScoreFacet) -> Iterator[Tuple[SliceID, pd.DataFrame]]:
        """Iterate through facet DataFrames.

        Args:
            what: Which type of facet to retrieve.

        Yields:
            Index tuple.
            Facet DataFrame.
        """
        if not self.slice_facet_if_necessary(what):
            logger.info(f"No sliced {what} available.")
            raise StopIteration
        yield from super().iter_facet(what=what)

    def slice_facet_if_necessary(self, what):
        """

        Parameters
        ----------
        what : :obj:`str`
            Facet for which to create slices if necessary

        Returns
        -------
        :obj:`bool`
            True if slices are available or not needed, False otherwise.
        """
        if not hasattr(self, "slice_info"):
            # no slicer applied
            return True
        if len(self.slice_info) == 0:
            # applying slicer did not yield any slices
            return True
        if what in self.sliced:
            # already sliced
            return True
        self.sliced[what] = {}
        facet_ids = defaultdict(list)
        for corpus, fname, interval in self.slice_info.keys():
            facet_ids[(corpus, fname)].append(interval)
        for id, intervals in facet_ids.items():
            facet_df = self.get_item(id, what)
            if facet_df is None or len(facet_df.index) == 0:
                continue
            sliced = ms3.overlapping_chunk_per_interval(facet_df, intervals)
            self.sliced[what].update(
                {id + (iv,): chunk for iv, chunk in sliced.items()}
            )
        if len(self.sliced[what]) == 0:
            del self.sliced[what]
            return False
        return True


class AnalyzedGroupedDataset(AnalyzedDataset, GroupedDataset):
    assert_types = ["GroupedDataset", "AnalyzedDataset"]
    type_mapping = {
        (
            "AnalyzedGroupedDataset",
            "AnalyzedDataset",
            "GroupedDataset",
        ): "AnalyzedGroupedDataset",
    }


class AnalyzedSlicedDataset(AnalyzedDataset, SlicedDataset):
    assert_types = ["SlicedDataset", "AnalyzedDataset"]
    excluded_types = []
    type_mapping = {
        (
            "AnalyzedSlicedDataset",
            "AnalyzedDataset",
            "SlicedDataset",
        ): "AnalyzedSlicedDataset",
    }
    pass


class GroupedSlicedDataset(GroupedDataset, SlicedDataset):
    assert_types = ["SlicedDataset", "GroupedDataset"]
    excluded_types = ["AnalyzedData"]
    type_mapping = {
        (
            "SlicedDataset",
            "GroupedDataset",
            "GroupedSlicedDataset",
        ): "GroupedSlicedDataset",
    }

    def get_slice_info(self, ignore_groups=False) -> pd.DataFrame:
        """Concatenates slice_info Series and returns them as a DataFrame."""
        group_dfs = {}
        for group, index_group in self.iter_grouped_indices():
            group_info = {ix: self.slice_info[ix] for ix in index_group}
            group_dfs[group] = pd.concat(
                group_info.values(), keys=group_info.keys(), axis=1
            ).T
        concatenated_info = pd.concat(group_dfs.values(), keys=group_dfs.keys())
        concatenated_info.index = self._rename_multiindex_levels(
            concatenated_info.index,
            self.index_levels["groups"] + self.index_levels["indices"],
        )
        return clean_index_levels(concatenated_info)

    # def iter_facet(self, what, unfold=False, concatenate=False, ignore_groups=False):
    #     """Iterate through groups of potentially sliced facet DataFrames.
    #
    #     Parameters
    #     ----------
    #     what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
    #             'cadences', 'chords', 'measures', 'rests'}
    #         What facet to retrieve.
    #     unfold : :obj:`bool`, optional
    #         Pass True if you need repeats to be unfolded.
    #     concatenate : :obj:`bool`, optional
    #         By default, the returned dict contains one DataFrame per ID in the group.
    #         Pass True to instead concatenate the DataFrames. Then, the dict will contain only
    #         one entry where the key is a tuple containing all IDs and the value is a DataFrame,
    #         the components of which can be distinguished using its MultiIndex.
    #     ignore_groups : :obj:`bool`, False
    #         If set to True, the iteration loop is flattened and yields (index, facet_df) pairs directly. Clashes
    #         with the setting concatenate=True which concatenates facets per group.
    #
    #     Yields
    #     ------
    #     :obj:`tuple`
    #         Group identifier
    #     :obj:`dict` or :obj:`pandas.DataFrame`
    #         Default: {ID -> DataFrame}.
    #         If concatenate=True: DataFrame with MultiIndex identifying ID, and (eventual) interval.
    #     """
    #     if not self.slice_facet_if_necessary(what, unfold):
    #         logger.info(f"No sliced {what} available.")
    #         raise StopIteration
    #     if sum((concatenate, ignore_groups)) > 1:
    #         raise ValueError(
    #             "Arguments 'concatenate' and 'ignore_groups' are in conflict, choose one "
    #             "or use the method get_facet()."
    #         )
    #     for group, index_group in self.iter_grouped_indices():
    #         result = {}
    #         missing_id = []
    #         for index in index_group:
    #             df = self.get_item(index, what=what, unfold=unfold)
    #             if df is None:
    #                 continue
    #             elif ignore_groups:
    #                 yield index, df
    #             if len(df.index) == 0:
    #                 missing_id.append(index)
    #             result[index] = df
    #         if ignore_groups:
    #             continue
    #         n_results = len(result)
    #         if len(missing_id) > 0:
    #             if n_results == 0:
    #                 pass
    #                 # logger.info(f"No '{what}' available for {group}.")
    #             else:
    #                 logger.info(
    #                     f"Group {group} is missing '{what}' for the following indices:\n"
    #                     f"{missing_id}"
    #                 )
    #         if n_results == 0:
    #             continue
    #         if concatenate:
    #             if n_results == 1:
    #                 # workaround necessary because of nasty "cannot handle overlapping indices;
    #                 # use IntervalIndex.get_indexer_non_unique" error
    #                 result["empty"] = pd.DataFrame()
    #             result = pd.concat(
    #                 result.values(),
    #                 keys=result.keys(),
    #                 names=self.index_levels["indices"] + ["interval"],
    #             )
    #             result = {tuple(index_group): result}
    #
    #         yield group, result


class AnalyzedGroupedSlicedDataset(AnalyzedSlicedDataset, GroupedSlicedDataset):
    assert_types = [
        "GroupedSlicedDataset",
        "AnalyzedGroupedDataset",
        "AnalyzedSlicedDataset",
    ]
    type_mapping = {
        (
            "GroupedSlicedDataset",
            "AnalyzedGroupedDataset",
            "AnalyzedSlicedDataset",
            "AnalyzedGroupedSlicedDataset",
        ): "AnalyzedGroupedSlicedDataset",
    }
    pass

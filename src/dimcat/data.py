"""Class hierarchy for data types."""
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from functools import lru_cache
from typing import List, Union

import pandas as pd
from ms3 import Parse, overlapping_chunk_per_interval

from .utils import clean_index_levels


class Data(ABC):
    """
    Subclasses are the dtypes that this library uses. Every PipelineStep accepts one or several
    dtypes.

    The initializer can set parameters influencing how the contained data will look and is able
    to create an object from an existing Data object to enable type conversion.
    """

    def __init__(self, data=None, **kwargs):
        """

        Parameters
        ----------
        data : Data
            Convert a given Data object if possible.
        kwargs
            All keyword arguments are passed to load().
        """
        self._data = None
        """Protected attribute for storing and internally accessing the loaded data."""

        self.indices = {}
        """Indices for accessing individual pieces of data and associated metadata."""

        self.processed = {}
        """Subclasses store processed data here."""

        self.pipeline_steps = []
        """The sequence of applied PipelineSteps that has led to the current state in reverse
        order (first element was applied last)."""

        self.index_levels = {
            "indices": ["corpus", "fname"],
            "slicer": [],
            "groups": [],
            "processed": [],
        }
        """Keeps track of index level names. Also used for automatic naming of output files."""

        self.sliced = {}
        """Dict for sliced data facets."""

        self.slice_info = {}
        """Dict holding metadata of slices (e.g. the localkey of a segment)."""

        self.group2pandas = "group2dataframe"

        if data is not None:
            self.data = data

    @property
    @abstractmethod
    def data(self):
        """Get the data field in its raw form."""
        return self._data

    @data.setter
    @abstractmethod
    def data(self, data_object):
        """Implement setting the _data field after performing type check."""
        if data_object is not None:
            raise NotImplementedError
        self._data = data_object

    def copy(self):
        return self.__class__(data=self)

    def get(self):
        """Get all processed data at once."""
        return self.processed

    def iter(self):
        """Iterate through processed data."""
        for tup in self.processed.items():
            yield tup

    @abstractmethod
    def iter_facet(self, what):
        """Iterate through (potentially grouped) pieces of data."""
        for index_group in self.iter_groups():
            yield [self.get_item(index) for index in index_group]

    @abstractmethod
    def get_item(self, index, what):
        """Get an individual piece of data."""

    def iter_groups(self):
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
        if any(len(index_list) == 0 for index_list in self.indices.values()):
            print("Data object contains empty groups.")
        yield from self.indices.items()

    def track_pipeline(
        self,
        pipeline_step,
        group2pandas=None,
        indices=None,
        processed=None,
        grouper=None,
        slicer=None,
    ):
        """Keep track of the applied pipeline_steps and update index level names and group2pandas
        conversion method.

        Parameters
        ----------
        pipeline_step : :obj:`PipelineStep`
        group2pandas : :obj:`str`, optional
        indices : :obj:`str`, optional
        processed : :obj:`str`, optional
        grouper : :obj:`str`, optional
        slicer : :obj:`str`, optional
        """
        self.pipeline_steps = [pipeline_step] + self.pipeline_steps
        if processed is not None:
            if isinstance(processed, str):
                processed = [processed]
            self.index_levels["processed"] = processed
        if indices is not None:
            if indices == "IDs":
                # once_per_group == True
                self.index_levels["indices"] = ["IDs"]
            elif len(self.index_levels["indices"]) == 2:
                self.index_levels["indices"] = self.index_levels["indices"] + [indices]
            else:
                self.index_levels["indices"][2] = indices
            assert 1 <= len(self.index_levels["indices"]) < 4
        if group2pandas is not None:
            self.group2pandas = group2pandas
        if grouper is not None:
            self.index_levels["groups"] = self.index_levels["groups"] + [grouper]
        if slicer is not None:
            self.index_levels["slicer"] = [slicer]

    @abstractmethod
    def load(self):
        """Load data into memory."""

    def group_of_values2series(self, group_dict) -> pd.Series:
        """Converts an {ID -> processing_result} dict into a Series."""
        series = pd.Series(group_dict, name=self.index_levels["processed"][0])
        series.index = self._rename_multiindex_levels(
            series.index, self.index_levels["indices"]
        )
        return series

    def group_of_series2series(self, group_dict) -> pd.Series:
        """Converts an {ID -> processing_result} dict into a Series."""
        lengths = [len(S) for S in group_dict.values()]
        if 0 in lengths:
            group_dict = {k: v for k, v in group_dict.items() if len(v) > 0}
            if len(group_dict) == 0:
                print("Group contained only empty Series")
                return pd.Series()
            else:
                n_empty = lengths.count(0)
                print(f"Had to remove {n_empty} empty Series before concatenation.")
        if len(group_dict) == 1 and list(group_dict.keys())[0] == "group_ids":
            series = list(group_dict.values())[0]
            series.index = self._rename_multiindex_levels(
                series.index, self.index_levels["processed"]
            )
        else:
            series = pd.concat(group_dict.values(), keys=group_dict.keys())
            series.index = self._rename_multiindex_levels(
                series.index,
                self.index_levels["indices"] + self.index_levels["processed"],
            )
        return series

    def group2dataframe(self, group_dict) -> pd.DataFrame:
        """Converts an {ID -> processing_result} dict into a DataFrame."""
        try:
            df = pd.concat(group_dict.values(), keys=group_dict.keys())
        except (TypeError, ValueError):
            print(group_dict)
            raise
        df.index = self._rename_multiindex_levels(
            df.index, self.index_levels["indices"] + self.index_levels["processed"]
        )
        return df

    def group2dataframe_unstacked(self, group_dict):
        return self.group2dataframe(group_dict).unstack()

    def _rename_multiindex_levels(self, multiindex: pd.MultiIndex, index_level_names):
        """Renames the index levels based on the _.index_levels dict."""
        try:
            n_levels = multiindex.nlevels
            if n_levels == 1:
                return multiindex.rename(index_level_names[0])
            n_names = len(index_level_names)
            if n_names < n_levels:
                levels = list(range(len(index_level_names)))
                # The level parameter makes sure that, when n names are given, only the first n levels are being
                # renamed. However, this will lead to unexpected behaviour if index levels are named by an integer
                # that does not correspond to the position of another index level, e.g. ('level0_name', 0, 1)
                return multiindex.rename(index_level_names, level=levels)
            elif n_names > n_levels:
                return multiindex.rename(index_level_names[:n_levels])
            return multiindex.rename(index_level_names)
        except (TypeError, ValueError) as e:
            print(
                f"Failed to rename MultiIndex levels {multiindex.names} to {index_level_names}: '{e}'"
            )
            print(multiindex[:10])
            print(f"self.index_levels: {self.index_levels}")
        # TODO: This method should include a call to clean_multiindex_levels and make use of self.index_levels
        return multiindex


def remove_corpus_from_ids(result):
    """Called when group contains corpus and removes redundant repetition from indices."""
    if isinstance(result, dict):
        without_corpus = {}
        for key, v in result.items():
            if isinstance(key[0], str):
                without_corpus[key[1:]] = v
            else:
                new_key = tuple(k[1:] for k in key)
                without_corpus[new_key] = v
        return without_corpus
    print(result)
    return result.droplevel(0)


class Corpus(Data):
    """Essentially a wrapper for a ms3.Parse object."""

    def __init__(self, data=None, **kwargs):
        """

        Parameters
        ----------
        data : Data
            Convert a given Data object into a Corpus if possible.
        kwargs
            All keyword arguments are passed to load().
        """
        super().__init__()
        self.pieces = {}
        """
        IDs and metadata of those pieces that have not been filtered out.::

            {(corpus, fname) ->
                {
                 'metadata' -> {key->value},
                 'matched_files' -> [namedtuple]
                }
            }

        """
        if data is None:
            self._data = Parse()
        else:
            self.data = data
        if len(kwargs) > 0:
            self.load(**kwargs)

    @property
    def data(self):
        """Get the data field in its raw form."""
        return self._data

    @data.setter
    def data(self, data_object):
        """Check if the assigned object is suitable for conversion."""
        if not isinstance(data_object, Corpus):
            raise TypeError(
                f"{data_object.__class__} could not be converted to a Corpus."
            )
        self._data = data_object._data
        self.pieces = deepcopy(data_object.pieces)
        self.indices = deepcopy(data_object.indices)
        self.processed = deepcopy(data_object.processed)
        self.sliced = deepcopy(data_object.sliced)
        self.slice_info = deepcopy(data_object.slice_info)
        self.pipeline_steps = list(data_object.pipeline_steps)
        self.index_levels = deepcopy(data_object.index_levels)
        self.group2pandas = data_object.group2pandas

    @property
    def is_grouped(self) -> bool:
        groups = list(self.indices.keys())
        return len(groups) != 1 or groups[0] != ()

    def get(self, as_dict=False):
        """Uses _.iter() to get all processed data at once.

        Parameters
        ----------
        as_dict : :obj:`bool`, optional
            By default, the result is a pandas DataFrame or Series where the first levels
            display the groups. Pass True to obtain a nested {group -> {id -> processed}}
            dictionary instead.

        Returns
        -------
        :obj:`pandas.DataFrame` or :obj:`pandas.Series` or :obj:`dict`
        """
        if len(self.processed) == 0:
            print("No data has been processed so far.")
            return
        results = {group: result for group, result in self.iter(as_dict=as_dict)}
        if as_dict:
            return results
        # default: concatenate to a single pandas object
        if len(results) == 1 and () in results:
            pandas_obj = pd.concat(results.values())
        else:
            try:
                pandas_obj = pd.concat(
                    results.values(),
                    keys=results.keys(),
                    names=self.index_levels["groups"],
                )
            except ValueError:
                print(self.index_levels["groups"])
                print(results.keys())
                raise
        return clean_index_levels(pandas_obj)

    def get_facet(self, what, unfold=False):
        """Uses _.iter_facet() to collect and concatenate all DataFrames for a particular facet.

        Parameters
        ----------
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.
        unfold : :obj:`bool`, optional
            Pass True if you need repeats to be unfolded.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        group_dfs = {
            group: df
            for group, dfs in self.iter_facet(
                what=what, unfold=unfold, concatenate=True
            )
            for df in dfs.values()
        }
        if len(group_dfs) == 1:
            return list(group_dfs.values())[0]
        concatenated_groups = pd.concat(
            group_dfs.values(), keys=group_dfs.keys(), names=self.index_levels["groups"]
        )
        return clean_index_levels(concatenated_groups)

    def convert_group2pandas(self, result_dict) -> Union[pd.Series, pd.DataFrame]:
        """Converts the {ID -> processing_result} dict using the method specified in _.group2pandas."""
        converters = {
            "group_of_values2series": self.group_of_values2series,
            "group_of_series2series": self.group_of_series2series,
            "group2dataframe": self.group2dataframe,
            "group2dataframe_unstacked": self.group2dataframe_unstacked,
        }
        converter = converters[self.group2pandas]
        pandas_obj = converter(result_dict)
        return clean_index_levels(pandas_obj)

    def iter(self, as_dict=False, ignore_groups=False):
        """Iterate through processed data.

        Parameters
        ----------
        as_dict : :obj:`bool`, optional
            By default, the IDs and processed data belonging to the same group are concatenated
            into a single pandas object (Series or DataFrame).

        Yields
        ------
        :obj:`tuple`
            Group identifier. Empty if no grouper has been applied previously.
        :obj:`pandas.DataFrame` or :obj:`pandas.Series` or :obj:`dict`
            Processed data for one particular group. Whether it is a pandas object or dict depends
            on ``as_dict``. Whether it is a Series or DataFrame depends on the previously applied
            Pipeline.
        ignore_groups : :obj:`bool`, False
            If set to True, the iteration loop is flattened and yields (index, result) pairs directly.
        """
        if sum((as_dict, ignore_groups)) > 1:
            raise ValueError(
                "Arguments 'as_dict' and 'ignore_groups' are in conflict, choose one."
            )
        for group, result in self.processed.items():
            if ignore_groups:
                yield from result.items()
            elif as_dict:
                yield group, result
            else:
                if ignore_groups:
                    for tup in result.items():
                        yield tup
                else:
                    yield group, self.convert_group2pandas(result)

    def load(
        self,
        directory: List[str] = None,
        parse_tsv: bool = True,
        parse_scores: bool = False,
        ms: str = None,
    ):
        """
        Load and parse all of the desired raw data and metadata.

        Parameters
        ----------
        directory : str
            The path to all the data to load.
        parse_tsv : :obj:`bool`, optional
            By default, all detected TSV files are parsed and their type is inferred.
            Pass False to prevent parsing TSV files.
        parse_scores : :obj:`bool`, optional
            By default, detected scores files are not parsed to save time.
            Call super().__init__(parse_scores=True) if an analyzer needs
            to access parsed MuseScore XML.
        ms : :obj:`str`, optional
            If you pass the path to your local MuseScore 3 installation, ms3 will attempt to parse
            musicXML, MuseScore 2, and other formats by temporarily converting them. If you're
            using the standard path, you may try 'auto', or 'win' for Windows, 'mac' for MacOS,
            or 'mscore' for Linux. In case you do not pass the 'file_re' and the MuseScore
            executable is detected, all convertible files are automatically selected, otherwise
            only those that can be parsed without conversion.
        """
        if ms is not None:
            self.data.ms = ms
        if directory is not None:
            if isinstance(directory, str):
                directory = [directory]
            for d in directory:
                self.data.add_dir(
                    directory=d,
                )
        if parse_tsv:
            self.data.parse_tsv()
        if parse_scores:
            self.data.parse_mscx()
        if len(self.data._parsed_tsv) == 0 and len(self.data._parsed_mscx) == 0:
            print("No files have been parsed for analysis.")
        else:
            self.get_indices()

    def get_indices(self):
        """Fills self.pieces with metadata and IDs for all loaded data. This resets previously
        applied groupings."""
        self.pieces = {}
        self.indices = {}
        # self.group_labels = {}
        for key in self.data.keys():
            view = self.data[key]
            for metadata, (fname, matched_files) in zip(
                view.metadata().to_dict(orient="records"),
                view.detect_ids_by_fname(parsed_only=True).items(),
            ):
                assert (
                    fname == metadata["fnames"]
                ), f"metadata() and pieces() do not correspond for key {key}, fname {fname}."
                piece_info = {}
                piece_info["metadata"] = metadata
                piece_info["matched_files"] = matched_files
                ID = (key, fname)
                self.pieces[ID] = piece_info
        self.indices[()] = list(self.pieces.keys())

    def slice_facet_if_necessary(self, what, unfold):
        """

        Parameters
        ----------
        what : :obj:`str`
            Facet for which to create slices if necessary
        unfold : :obj:`bool`
            Whether repeats should be unfolded.

        Returns
        -------
        :obj:`bool`
            True if slices are available or not needed, False otherwise.
        """
        if len(self.slice_info) == 0:
            # no slicer applied
            return True
        if what in self.sliced:
            # already sliced
            return True
        self.sliced[what] = {}
        facet_ids = defaultdict(list)
        for corpus, fname, interval in self.slice_info.keys():
            facet_ids[(corpus, fname)].append(interval)
        for id, intervals in facet_ids.items():
            facet_df = self.get_item(id, what, unfold)
            if facet_df is None or len(facet_df.index) == 0:
                continue
            sliced = overlapping_chunk_per_interval(facet_df, intervals)
            self.sliced[what].update(
                {id + (iv,): chunk for iv, chunk in sliced.items()}
            )
        if len(self.sliced[what]) == 0:
            del self.sliced[what]
            return False
        return True

    def iter_facet(self, what, unfold=False, concatenate=False, ignore_groups=False):
        """Iterate through groups of potentially sliced facet DataFrames.

        Parameters
        ----------
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.
        unfold : :obj:`bool`, optional
            Pass True if you need repeats to be unfolded.
        concatenate : :obj:`bool`, optional
            By default, the returned dict contains one DataFrame per ID in the group.
            Pass True to instead concatenate the DataFrames. Then, the dict will contain only
            one entry where the key is a tuple containing all IDs and the value is a DataFrame,
            the components of which can be distinguished using its MultiIndex.
        ignore_groups : :obj:`bool`, False
            If set to True, the iteration loop is flattened and yields (index, facet_df) pairs directly. Clashes
            with the setting concatenate=True which concatenates facets per group.

        Yields
        ------
        :obj:`tuple`
            Group identifier
        :obj:`dict` or :obj:`pandas.DataFrame`
            Default: {ID -> DataFrame}.
            If concatenate=True: DataFrame with MultiIndex identifying ID, and (eventual) interval.
        """
        if not self.slice_facet_if_necessary(what, unfold):
            print(f"No sliced {what} available.")
            raise StopIteration
        if sum((concatenate, ignore_groups)) > 1:
            raise ValueError(
                "Arguments 'concatenate' and 'ignore_groups' are in conflict, choose one "
                "or use the method get_facet()."
            )
        for group, index_group in self.iter_groups():
            result = {}
            missing_id = []
            for index in index_group:
                df = self.get_item(index, what, unfold)
                if df is None:
                    continue
                elif ignore_groups:
                    yield index, df
                if len(df.index) == 0:
                    missing_id.append(index)
                result[index] = df
            if ignore_groups:
                continue
            n_results = len(result)
            if len(missing_id) > 0:
                if n_results == 0:
                    pass
                    # print(f"No '{what}' available for {group}.")
                else:
                    print(
                        f"Group {group} is missing '{what}' for the following indices:\n"
                        f"{missing_id}"
                    )
            if n_results == 0:
                continue
            if concatenate:
                if n_results == 1:
                    # workaround necessary because of nasty "cannot handle overlapping indices;
                    # use IntervalIndex.get_indexer_non_unique" error
                    result["empty"] = pd.DataFrame()
                result = pd.concat(
                    result.values(),
                    keys=result.keys(),
                    names=self.index_levels["indices"] + ["interval"],
                )
                result = {tuple(index_group): result}

            yield group, result

    def get_previous_pipeline_step(self, idx=0, of_type=None):
        """Retrieve one of the previously applied PipelineSteps, either by index or by type.

        Parameters
        ----------
        idx : :obj:`int`, optional
            List index used if ``of_type`` is None. Defaults to 0, which is the PipeLine step
            most recently applied.
        of_type : :obj:`PipelineStep`, optional
            Return the most recently applied PipelineStep of this type.

        Returns
        -------
        :obj:`PipelineStep`
        """
        if of_type is None:
            n_previous_steps = len(self.pipeline_steps)
            try:
                return self.pipeline_steps[idx]
            except IndexError:
                print(f"Invalid index idx={idx} for list of length {n_previous_steps}")
                raise
        try:
            return next(
                step for step in self.pipeline_steps if isinstance(step, of_type)
            )
        except StopIteration:
            raise StopIteration(
                f"Previously applied PipelineSteps do not include any {of_type}: {self.pipeline_steps}"
            )

    def get_slice(self, index, what):
        if what in self.sliced and index in self.sliced[what]:
            return self.sliced[what][index]

    def get_slice_info(self) -> pd.DataFrame:
        """Concatenates slice_info Series and returns them as a DataFrame."""
        concatenated_info = pd.concat(
            self.slice_info.values(), keys=self.slice_info.keys(), axis=1
        ).T
        concatenated_info.index.rename(self.index_levels["indices"], inplace=True)
        return concatenated_info

    @lru_cache()
    def get_item(self, index, what, unfold=False, multiindex=False):
        """Retrieve a DataFrame pertaining to the facet ``what`` of the piece ``index``. If
        the facet has been sliced before, the sliced DataFrame is returned.

        Parameters
        ----------
        index : :obj:`tuple`
            (corpus, fname) or (corpus, fname, interval)
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.
        unfold : :obj:`bool`, optional
            Pass True if you need repeats to be unfolded.
        multiindex : :obj:`bool`, optional
            By default, has one level, which is a :obj:``pandas.IntervalIndex``. Pass True to
            prepend ``index`` as an additional index level.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        n_index_elements = len(index)
        if n_index_elements == 2:
            corpus, fname = index
            try:
                df = self.data[corpus][fname].get_dataframe(
                    what, unfold, interval_index=True
                )
                if not isinstance(df.index, pd.IntervalIndex):
                    print(f"'{what}' of {index} does not come with an IntervalIndex")
                    df = pd.DataFrame()
            except FileNotFoundError:
                # print(f"No {what} available for {index}. Returning empty DataFrame.")
                df = pd.DataFrame()
        elif n_index_elements == 3:
            df = self.get_slice(index, what)
        else:
            raise NotImplementedError(
                f"'{index}': Indices can currently include 2 or 3 elements."
            )
        if df is None:
            return
        assert (
            df.index.nlevels == 1
        ), f"Retrieved DataFrame has {df.index.nlevels}, not 1"
        if multiindex:
            df = pd.concat([df], keys=[index[:2]], names=["corpus", "fname"])
        return df

"""Class hierarchy for data types."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

import pandas as pd
from dimcat.base import Data, PipelineStep
from dimcat.data.loader import infer_data_loader
from dimcat.dtypes import PieceID, PieceIndex, PLoader, PPiece, SomeID
from dimcat.utils.functions import clean_index_levels
from ms3._typing import ScoreFacet
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Dataset(Data):
    """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` accept."""

    # region Initialization

    def __init__(self, dataset: Optional[Dataset] = None, **kwargs):
        """The central

        Args:
            dataset: Instantiate from this Dataset by copying its fields, empty fields otherwise.
            **kwargs: Dataset is cooperative and calls super().__init__(data=dataset, **kwargs)
        """
        super().__init__(data=dataset, **kwargs)

        self.loaders: List[PLoader] = []
        """Stores the various loaders which, together, are responsible the original,
        unprocessed :attr:`pieces` mapping."""

        self._pieces: Dict[PieceID, PPiece] = {}
        """References to the individual pieces contained in the data. The exact type depends on the type of data.
        Controlled through the property :attr:`pieces` which returns a copy and cannot be modified directly.
        """

        self.piece_index: PieceIndex = PieceIndex([])
        """List of PieceIDs used for accessing individual pieces of data and
        associated metadata. A PieceID is a ('corpus', 'piece') NamedTuple."""

        self.pipeline_steps: List[PipelineStep] = []
        """The sequence of applied PipelineSteps that has led to the current state."""

        if dataset is not None:
            # If subclasses have a different logic of copying fields, they can override these methods
            self._init_piece_index_from_dataset(dataset, **kwargs)
            self._init_pieces_from_dataset(dataset, **kwargs)
            self._init_pipeline_steps_from_dataset(dataset, **kwargs)
            self._init_loaders_from_dataset(dataset, **kwargs)

    def _init_piece_index_from_dataset(self, dataset: Dataset, **kwargs):
        self.piece_index = PieceIndex(dataset.piece_index)

    def _init_pieces_from_dataset(self, dataset: Dataset, **kwargs):
        self._pieces = {PID: dataset.get_piece(PID) for PID in self.piece_index}

    def _init_pipeline_steps_from_dataset(self, dataset: Dataset, **kwargs):
        self.pipeline_steps = list(dataset.pipeline_steps)

    def _init_loaders_from_dataset(self, dataset: Dataset, **kwargs):
        self.loaders = list(dataset.loaders)

    def copy(self, **kwargs) -> Self:
        """Return a copy of the Dataset."""
        return self.__class__(dataset=self, **kwargs)

    # endregion Initialization

    # region Properties

    @property
    def n_indices(self) -> int:
        """Number of pieces currently selected. Different from n_pieces."""
        return len(self.piece_index)

    @property
    def n_pieces(self) -> int:
        """Number of pieces that this dataset has access to. For a Dataset that has been initialized as a subset,
        this number would increase with a call to :meth:`reset_indices`.
        """
        return len(self._pieces)

    @property
    def pieces(self) -> Dict[PieceID, PPiece]:
        """References to the individual pieces contained in the dataset. The exact type depends on the type of data."""
        return dict(self._pieces)

    # endregion Properties

    # region Loaders

    def _retrieve_pieces_from_loader(self, loader: PLoader):
        """Add the Piece objects that the given loader yields to :attr:`pieces` and add their IDs to the
        :attr:`self.piece_index`.
        """
        for PID, piece in loader.iter_pieces():
            self._set_piece(piece_id=PID, piece_obj=piece)

    def _set_piece(self, piece_id: PieceID, piece_obj: PPiece) -> None:
        """Add a piece to the dataset.

        Raises:
            ValueError if ``piece_id`` is already present.
        """
        if piece_id in self._pieces:
            raise ValueError(
                f"Dataset already contains a piece with PieceID {piece_id}."
            )
        if piece_id in self.piece_index:
            raise ValueError(
                f"Dataset.piece_index already contains PieceID {piece_id} although it "
                f"had not been present in .pieces"
            )
        self._pieces[piece_id] = piece_obj
        self.piece_index.append(piece_id)

    def attach_loader(self, loader: PLoader):
        """Add an initialized loader to :attr:`loaders` and add the pieces it yields to :attr:`pieces`."""
        self.loaders.append(loader)
        self._retrieve_pieces_from_loader(loader)

    def load(
        self,
        directory: Optional[Union[str, List[str]]],
        loader: Optional[Type[PLoader]] = None,
        **kwargs,
    ) -> None:
        """Add to the dataset all pieces that ``loader`` creates from ``directory``.
        If no loader class is specified, DiMCAT will try to infer it.

        Args:
            directory: The path(s) to all the data to load.
            loader:
                Loader class to be initialized with the keyword arguments ``directory`` and ``**kwargs``.
                If none is specified, DiMCAT will call :func:`infer_data_loader` on each directory.
            **kwargs:
                Keyword arguments that the specified loader class accepts. If the loader class is to be inferred,
                only arguments specified in :meth:`.dtypes.PLoader.__init__` are safe.
        """
        if isinstance(directory, str):
            directory = [directory]
        _loader = loader
        for d in directory:
            if loader is None:
                _loader = infer_data_loader(d)
            loader_object = _loader(directory=d, **kwargs)
            self.attach_loader(loader_object)

    def reset_pieces(self):
        """Iterate through all attached loaders and"""
        self.piece_index = PieceIndex([])
        for loader in self.loaders:
            self._retrieve_pieces_from_loader(loader)

    # endregion Loaders

    # region Data access

    def get_piece(self, PID: Union[PieceID, Tuple[str, str]]) -> PPiece:
        """Get a Piece object by its ('corpus', 'piece') PieceID"""
        piece = self._pieces.get(PID)
        if piece is None:
            raise KeyError(f"ID not found in .pieces: {PID}")
        return piece

    def iter_pieces(self) -> Iterator[Tuple[PieceID, PPiece]]:
        for PID in self.piece_index:
            yield PID, self.get_piece(PID)

    def get_facet(self, what: ScoreFacet) -> pd.DataFrame:
        """Uses _.iter_facet() to collect and concatenate all DataFrames for a particular facet.

        Parameters
        ----------
        what : {'form_labels', 'events', 'expanded', 'notes_and_rests', 'notes', 'labels',
                'cadences', 'chords', 'measures', 'rests'}
            What facet to retrieve.

        Returns
        -------
        :obj:`pandas.DataFrame`
        """
        dfs = {idx: df for idx, df in self.iter_facet(what=what)}
        if len(dfs) == 1:
            return list(dfs.values())[0]
        concatenated_groups = pd.concat(
            dfs.values(), keys=dfs.keys(), names=self.index_levels["indices"]
        )
        return clean_index_levels(concatenated_groups)

    @lru_cache()
    def get_item(self, ID: PieceID, what: ScoreFacet) -> Optional[pd.DataFrame]:
        """Retrieve a DataFrame pertaining to the facet ``what`` of the piece ``index``.

        Args:
            ID: (corpus, fname) or (corpus, fname, interval)
            what: What facet to retrieve.

        Returns:
            DataFrame representing an entire score facet, or a chunk (slice) of it.
        """

        file, df = self._pieces[ID].get_facet(what, interval_index=True)
        logger.debug(f"Retrieved {what} from {file}.")
        if df is not None and not isinstance(df.index, pd.IntervalIndex):
            logger.info(f"'{what}' of {ID} does not come with an IntervalIndex")
            df = None
        if df is None:
            return
        assert (
            df.index.nlevels == 1
        ), f"Retrieved DataFrame has {df.index.nlevels}, not 1"
        return df

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
                logger.info(
                    f"Invalid index idx={idx} for list of length {n_previous_steps}"
                )
                raise
        try:
            return next(
                step for step in self.pipeline_steps if isinstance(step, of_type)
            )
        except StopIteration:
            raise StopIteration(
                f"Previously applied PipelineSteps do not include any {of_type}: {self.pipeline_steps}"
            )

    def iter_facet(self, what: ScoreFacet) -> Iterator[Tuple[SomeID, pd.DataFrame]]:
        """Iterate through facet DataFrames.

        Args:
            what: Which type of facet to retrieve.

        Yields:
            Index tuple.
            Facet DataFrame.
        """
        for idx, piece in self.iter_pieces():
            _, df = piece.get_parsed(facet=what, interval_index=True)
            if df is None or len(df.index) == 0:
                logger.info(f"{idx} has no {what}.")
                continue
            yield idx, df

    def set_indices(
        self, new_indices: Union[List[SomeID], Dict[SomeID, List[SomeID]]]
    ) -> None:
        """Replace :attr:`indices` with a new list of IDs.

        Args:
            new_indices:
                The new list IDs or a dictionary of several lists of IDs. The latter is useful for re-grouping
                freshly sliced IDs of a :class:`GroupedDataset`.
        """
        if isinstance(new_indices, dict):
            new_indices = sum(new_indices.values(), [])
        self.piece_index = new_indices

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
                logger.info("Group contained only empty Series")
                return pd.Series()
            else:
                n_empty = lengths.count(0)
                logger.info(
                    f"Had to remove {n_empty} empty Series before concatenation."
                )
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
            logger.info(group_dict)
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
            logger.info(
                f"Failed to rename MultiIndex levels {multiindex.names} to {index_level_names}: '{e}'"
            )
            logger.info(multiindex[:10])
            logger.info(f"self.index_levels: {self.index_levels}")
        # TODO: This method should include a call to clean_multiindex_levels and make use of self.index_levels
        return multiindex

    def __str__(self):
        return str(self.piece_index)

    def __repr__(self):
        return str(self.piece_index)


def remove_corpus_from_ids(result):
    """Called when group contains corpus_names and removes redundant repetition from indices."""
    if isinstance(result, dict):
        without_corpus = {}
        for key, v in result.items():
            if isinstance(key[0], str):
                without_corpus[key[1:]] = v
            else:
                new_key = tuple(k[1:] for k in key)
                without_corpus[new_key] = v
        return without_corpus
    logger.info(result)
    return result.droplevel(0)

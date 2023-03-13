"""Class hierarchy for data types."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

import pandas as pd
from dimcat.base import Data, PipelineStep
from dimcat.data.facet import (
    Available,
    DefaultFeatureConfig,
    Facet,
    FacetConfig,
    FacetID,
    FacetName,
    FeatureConfig,
    FeatureID,
    FeatureName,
    StackedFacet,
    StackedFacetConfig,
    StackedFacetID,
    TabularFeature,
    feature_config2facet_config,
    get_stacked_facet_class,
    str2feature_name,
)
from dimcat.data.loader import PLoader, StackedFacetLoader, infer_data_loader
from dimcat.data.piece import PPiece
from dimcat.dtypes import Configuration, PieceID, PieceIndex
from dimcat.dtypes.base import SomeFeature
from IPython.display import display
from typing_extensions import Self

logger = logging.getLogger(__name__)


class Dataset(Data):
    """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of."""

    # region Initialization

    def __init__(self, data: Optional[Dataset] = None, **kwargs):
        """The central type of object that all :obj:`PipelineSteps <.PipelineStep>` process and return a copy of.

        Args:
            data: Instantiate from this Dataset by copying its fields, empty fields otherwise.
            **kwargs: Dataset is cooperative and calls super().__init__(data=dataset, **kwargs)
        """
        super().__init__(data=data, **kwargs)

        self.loaders: List[PLoader] = []
        """Stores the various loaders which, together, are responsible the original,
        unprocessed :attr:`pieces` mapping."""

        self._pieces: Dict[PieceID, PPiece] = {}
        """References to the individual pieces contained in the data. The exact type depends on the type of data.
        Controlled through the property :attr:`pieces` which returns a copy and cannot be modified directly.
        """

        self._cache: Dict[Configuration, SomeFeature] = {}

        self.piece_index: PieceIndex = PieceIndex([])
        """List of PieceIDs used for accessing individual pieces of data and
        associated metadata. A PieceID is a ('corpus', 'piece') NamedTuple."""

        self.pipeline_steps: List[PipelineStep] = []
        """The sequence of applied PipelineSteps that has led to the current state."""

        if data is not None:
            # If subclasses have a different logic of copying fields, they can override these methods
            self._init_piece_index_from_dataset(data)
            self._init_pieces_from_dataset(data)
            self._init_pipeline_steps_from_dataset(data)
            self._init_loaders_from_dataset(data)
        if len(kwargs) > 0:
            self.load(**kwargs)

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
        for piece_obj in loader.iter_pieces():
            self._set_piece(piece_obj=piece_obj)

    def _set_piece(self, piece_obj: PPiece) -> None:
        """Add a piece to the dataset.

        Raises:
            ValueError if ``piece_id`` is already present.
        """
        PID = piece_obj.piece_id
        if PID in self._pieces:
            raise ValueError(f"Dataset already contains a piece with PieceID {PID}.")
        if PID in self.piece_index:
            raise ValueError(
                f"Dataset.piece_index already contains PieceID {PID} although it "
                f"had not been present in .pieces"
            )
        self._pieces[PID] = piece_obj
        self.piece_index.append(PID)

    def set_loader(self, loader: PLoader):
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
            self.set_loader(loader_object)

    def reset_pieces(self):
        """Iterate through all attached loaders and"""
        self.piece_index = PieceIndex([])
        for loader in self.loaders:
            self._retrieve_pieces_from_loader(loader)

    # endregion Loaders

    # region Data access

    def available_facets(
        self, min_availability: Optional[Available] = None
    ) -> Dict[PieceID, Dict[FacetName, Available]]:
        availability = {
            piece.piece_id: piece.get_available_facets(
                min_availability=min_availability
            )
            for piece in self.iter_pieces()
        }
        return availability

    def get_piece(self, PID: Union[PieceID, Tuple[str, str]]) -> PPiece:
        """Get a Piece object by its ('corpus', 'piece') PieceID"""
        piece = self._pieces.get(PID)
        if piece is None:
            raise KeyError(f"ID not found in .pieces: {PID}")
        return piece

    def iter_pieces(self) -> Iterator[PPiece]:
        for PID in self.piece_index:
            yield self.get_piece(PID)

    def get_facet(self, facet: Union[FacetName, Configuration]) -> StackedFacet:
        """Retrieve the facet from all selected pieces, stacked.

        Args:
            facet:

        Returns:

        """
        if isinstance(facet, Configuration):
            if isinstance(facet, (StackedFacetID, FacetID)):
                raise NotImplementedError("Not accepting IDs as of now, only configs.")
            stacked_facet_config = StackedFacetConfig.from_dataclass(facet)
        else:
            facet_class = get_stacked_facet_class(facet)
            stacked_facet_config = facet_class.get_default_config()
        if stacked_facet_config in self._cache:
            return self._cache[stacked_facet_config]
        config2stacked_facet_objects: Dict[StackedFacetConfig] = defaultdict(list)
        for loader in self.loaders:
            stacked_facet = loader.get_facet(facet)
            config = StackedFacetConfig.from_dataclass(stacked_facet)
            config2stacked_facet_objects[config].append(stacked_facet)
        if len(config2stacked_facet_objects) > 1:
            raise NotImplementedError(
                f"Currently, facets with diverging configs cannot be concatenated:\n"
                f"{set(config2stacked_facet_objects.keys())}"
            )
        concatenated_per_config = []
        for config, stacked_facet_objects in config2stacked_facet_objects.items():
            stacked_facet_dfs = pd.concat(stacked_facet_objects)
            piece_ids = []
            for sfo in stacked_facet_objects:
                piece_ids.extend(sfo.piece_index)
            piece_index = PieceIndex(piece_ids)
            facet_constructor = get_stacked_facet_class(config.dtype)
            stacked_facet = facet_constructor.from_df(
                df=stacked_facet_dfs, piece_index=piece_index, file_path=None
            )
            concatenated_per_config.append(stacked_facet)

        result = concatenated_per_config[0]
        stacked_facet_config = StackedFacetConfig.from_dataclass(result)
        self._cache[stacked_facet_config] = result
        return result

    def get_feature(self, feature: Union[FeatureName, Configuration]) -> TabularFeature:
        if isinstance(feature, Configuration):
            if isinstance(feature, FeatureID):
                raise NotImplementedError("Not accepting IDs as of now, only configs.")
            feature_config = FeatureConfig.from_dataclass(feature)
            feature_name = feature_config.dtype
            facet_argument = feature_config2facet_config(feature_config)
        else:
            feature_name = str2feature_name(feature)
            feature_config = DefaultFeatureConfig(feature_name=feature_name)
            facet_argument = feature_name.facet
        stacked_facet = self.get_facet(facet_argument)
        return stacked_facet.get_feature(feature_config)

    def iter_facet(self, facet: Union[FacetName, FacetConfig]) -> Iterator[Facet]:
        """Iterate through :obj:`Facet` objects."""
        for piece_obj in self.iter_pieces():
            facet_obj = piece_obj.get_facet(facet=facet)
            yield facet_obj

    def iter_feature(
        self, feature: Union[FeatureName, Configuration]
    ) -> Iterator[TabularFeature]:
        stacked_feature = self.get_feature(feature)
        for piece_id, df in stacked_feature.groupby(level=[0, 1]):
            pass

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
        pipeline_steps = list(reversed(self.pipeline_steps))
        if of_type is None:
            n_previous_steps = len(pipeline_steps)
            try:
                return pipeline_steps[idx]
            except IndexError:
                logger.info(
                    f"Invalid index idx={idx} for list of length {n_previous_steps}"
                )
                raise
        try:
            return next(step for step in pipeline_steps if isinstance(step, of_type))
        except StopIteration:
            raise StopIteration(
                f"Previously applied PipelineSteps do not include any {of_type}: {self.pipeline_steps}"
            )

    # endregion Data access

    # region Display

    def show_available_facets(self, min_availability: Optional[Available] = None):
        available = self.available_facets(min_availability=min_availability)
        available_df = pd.DataFrame.from_dict(available, orient="index")
        if available_df.isna().any().any():
            available_df = available_df.fillna(Available.EXTERNALLY).astype(int)
        display(available_df)

    # endregion Display

    # region Dunder methods

    def __str__(self):
        return str(self.show_available_facets())

    def __repr__(self):
        return str(self.show_available_facets())

    # endregion Dunder methods


if __name__ == "__main__":
    from dimcat.data.loader import DcmlLoader
    from ms3 import assert_dfs_equal

    loader1 = DcmlLoader("~/corelli")
    loader2 = StackedFacetLoader("~/corelli")
    dataset1 = Dataset()
    dataset2 = Dataset()
    dataset1.set_loader(loader1)
    dataset2.set_loader(loader2)
    print(dataset1.piece_index == dataset2.piece_index)
    print(f"DS1:\n{dataset1.get_facet('Notes')}")
    print(f"DS2:\n{dataset2.get_facet('Notes')}")
    assert_dfs_equal(dataset1.get_facet("Notes").df, dataset2.get_facet("Notes").df)

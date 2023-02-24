"""Class hierarchy for data types."""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Iterator, List, Optional, Tuple, Type, Union

import pandas as pd
from dimcat.base import Data, PipelineStep
from dimcat.data.facet import Available, Facet, FacetConfig, FacetName, FeatureName
from dimcat.data.loader import PLoader, infer_data_loader
from dimcat.data.piece import PPiece
from dimcat.dtypes import PieceID, PieceIndex
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

        self.piece_index: PieceIndex = PieceIndex([])
        """List of PieceIDs used for accessing individual pieces of data and
        associated metadata. A PieceID is a ('corpus', 'piece') NamedTuple."""

        self.pipeline_steps: List[PipelineStep] = []
        """The sequence of applied PipelineSteps that has led to the current state."""

        if data is not None:
            # If subclasses have a different logic of copying fields, they can override these methods
            self._init_piece_index_from_dataset(data, **kwargs)
            self._init_pieces_from_dataset(data, **kwargs)
            self._init_pipeline_steps_from_dataset(data, **kwargs)
            self._init_loaders_from_dataset(data, **kwargs)

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

    def get_facet(self, facet: Union[FacetName, FacetConfig]) -> Facet:
        """Retrieve the concatenated facet for all selected pieces.

        Args:
            facet:

        Returns:

        """
        config2facet_objects: Dict[FacetConfig, Dict[PieceID, Facet]] = defaultdict(
            dict
        )
        for facet_obj in self.iter_facet(facet=facet):
            facet_config = FacetConfig.from_dataclass(facet_obj)
            config2facet_objects[facet_config][facet_obj.piece_id] = facet_obj
        if len(config2facet_objects) > 1:
            raise NotImplementedError(
                f"Currently, facets with diverging configs cannot be concatenated:\n"
                f"{set(config2facet_objects.keys())}"
            )
        concatenated_per_config = []
        for config, facet_objects in config2facet_objects.items():
            concatenated_per_config.append(
                config.concat_method(facet_objects, names=["corpus", "piece"])
            )
        result = concatenated_per_config[0]
        return result

    def iter_facet(self, facet: Union[FacetName, FacetConfig]) -> Iterator[Facet]:
        """Iterate through :obj:`Facet` objects."""
        if facet is FeatureName.TPC:
            feature = FeatureName.TPC
            facet = FacetName.Notes
            for piece_obj in self.iter_pieces():
                facet_obj = piece_obj.get_facet(facet=facet)
                yield facet_obj.get_feature(feature)
        else:
            for piece_obj in self.iter_pieces():
                facet_obj = piece_obj.get_facet(facet=facet)
                yield facet_obj

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
        return str(self.piece_index)

    def __repr__(self):
        return str(self.piece_index)

    # endregion Dunder methods


if __name__ == "__main__":
    from dimcat.data.loader import DcmlLoader

    loader = DcmlLoader("~/corelli")
    dataset = Dataset()
    dataset.set_loader(loader)
    print(dataset.piece_index)
    print(dataset.get_facet("Notes"))

from __future__ import annotations

import os
import warnings
from functools import cache
from pprint import pformat
from typing import Dict, Generic, Iterable, List, Optional, Sequence

import frictionless as fl
import marshmallow as mm
import ms3
import pandas as pd
from dimcat.base import get_setting
from dimcat.data.base import Data
from dimcat.data.resource.base import (
    IX,
    D,
    Resource,
    ResourceStatus,
    SomeDataframe,
    SomeIndex,
)
from dimcat.data.resource.utils import (
    align_with_grouping,
    ensure_level_named_piece,
    infer_schema_from_df,
    load_fl_resource,
    load_index_from_fl_resource,
    make_boolean_mask_from_set_of_tuples,
    make_index_from_grouping_dict,
    make_tsv_resource,
)
from dimcat.dc_exceptions import (
    BasePathNotDefinedError,
    FilePathNotDefinedError,
    PotentiallyUnrelatedDescriptorError,
    ResourceIsFrozenError,
)
from dimcat.dc_warnings import PotentiallyUnrelatedDescriptorUserWarning
from dimcat.utils import check_name
from frictionless import FrictionlessException
from typing_extensions import Self


class DimcatResource(Resource, Generic[D]):
    """Data object wrapping a dataframe. The dataframe's metadata are stored as a :obj:`frictionless.Resource`, that
    can be used for serialization and (lazy) deserialization.

    Every serialization of a DimcatResource (e.g. to store it as a config) requires that the dataframe was either
    originally read from disk or, otherwise, that it be stored to disk. The behaviour depends on whether the resource
    is part of a package or not.

    Standalone resource (rare case)
    -------------------------------

    If the resource is not part of a package, serializing it results in two files on disk:

    - the dataframe stored as ``<basepath>/<name>.tsv``
    - the frictionless descriptor ``<basepath>/<name>.resource.json``

    where ``<name>`` defaults to ``resource_name`` unless ``filepath`` is specified. The serialization has the shape

    .. code-block:: python

        {
            "dtype": "DimcatResource",
            "resource": "<name>.resource.json",
            "basepath": "<basepath>"
        }

    A standalone resource can be instantiated in the following ways:

    - ``DimcatResource()``: Creates an empty DimcatResource for setting the .df attribute later. If no ``basepath``
      is specified, the current working directory is used if the resource is to be serialized.
    - ``DimcatResource.from_descriptor(descriptor_path)``: The frictionless descriptor is loaded from disk.
      Its directory is used as ``basepath``. ``descriptor_path`` is expected to end in "resource.[json|yaml]".
    - ``DimcatResource.from_dataframe(df=df, resource_name, basepath)``: Creates a new DimcatResource from a dataframe.
      If ``basepath`` is not specified, the current working directory is used if the resource is to be serialized.
    - ``DimcatResource.from_resource(resource=DimcatResource)``: Creates a DimcatResource from an existing one
      by copying the fields it specifies.

    Resource in a package (common case)
    -----------------------------------

    A DimcatResource knows that it is part of a package if its ``filepath`` ends on ``.zip``. In that case, the
    DimcatPackage will take care of the serialization and not store an individual resource descriptor.
    """

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict | Resource,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        """Create a DimcatResource by loading its frictionless descriptor is loaded from disk.
        The descriptor's directory is used as ``basepath``. ``descriptor_path`` is expected to end in
        ``.resource.json``.

        Args:
            descriptor: Descriptor corresponding to a frictionless resource descriptor.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end either in resource.json or resource.yaml.
            basepath: Where the file would be serialized.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        return super().from_descriptor(
            descriptor=descriptor,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )

    @classmethod
    def from_descriptor_path(
        cls,
        descriptor_path: str,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        """Create a Resource from a frictionless descriptor file on disk.

        Args:
            descriptor_path: Absolute path where the JSON/YAML descriptor is located.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).

        """
        return super().from_descriptor_path(
            descriptor_path=descriptor_path,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: D,
        resource_name: str,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        """Create a DimcatResource from a dataframe, specifying its name and, optionally, at what path it is to be
        serialized.

        Args:
            df: Dataframe to create the resource from.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            basepath: Where the file would be serialized. If ``resource`` is a filepath, its directory is used.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        new_object = cls(
            basepath=basepath,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )
        assert (
            new_object.resource.path is not None
        ), f"""{cls.name}(
    basepath={basepath},
    descriptor_filename={descriptor_filename},
    auto_validate={auto_validate},
    default_groupby={default_groupby},
)
"""
        if resource_name is not None:
            new_object.resource_name = resource_name
        new_object._df = df
        assert (
            new_object.resource.path is not None
        ), f"""{cls.name}(
    basepath={basepath},
    descriptor_filename={descriptor_filename},
    auto_validate={auto_validate},
    default_groupby={default_groupby},
)
"""
        return new_object

    @classmethod
    def from_filepath(
        cls,
        filepath: str,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
        basepath: Optional[str] = None,
        **kwargs: Optional[bool],
    ) -> Self:
        """Create a Resource from a file on disk, be it a JSON/YAML resource descriptor, or a simple path resource.

        Args:
            filepath: Path pointing to a resource descriptor or a simple path resource.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end either in resource.json or resource.yaml.
            auto_validate:
                By default, the Resource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
            basepath:
                Basepath to use for the resource. If None, the folder of the ``filepath`` is used.
        """
        return super().from_filepath(
            filepath=filepath,
            resource_name=resource_name,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_index(
        cls,
        index: DimcatIndex | SomeIndex,
        resource_name: str,
        basepath: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        if isinstance(index, DimcatIndex):
            index = index.index
        dataframe = pd.DataFrame(index=index)
        return cls.from_dataframe(
            df=dataframe,
            resource_name=resource_name,
            descriptor_filename=descriptor_filename,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            basepath=basepath,
        )

    @classmethod
    def from_resource(
        cls,
        resource: Resource,
        descriptor_filename: Optional[str] = None,
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: Optional[bool] = None,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        """Create a DimcatResource from an existing :obj:`Resource`, specifying its name and,
        optionally, at what path it is to be serialized.

        Args:
            resource: An existing :obj:`frictionless.Resource` or a filepath.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            basepath: Where the file would be serialized. If ``resource`` is a filepath, its directory is used.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        if not isinstance(resource, Resource):
            raise TypeError(f"Expected a Resource, got {type(resource)!r}.")
        new_object = super().from_resource(
            resource=resource,
            descriptor_filename=descriptor_filename,
            resource_name=resource_name,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )
        # copy additional fields
        for attr in ("_df", "_status", "_corpus_name"):
            if (
                hasattr(resource, attr)
                and (value := getattr(resource, attr)) is not None
            ):
                setattr(new_object, attr, value)
        return new_object

    @classmethod
    def from_resource_path(
        cls,
        resource_path: str,
        resource_name: Optional[str] = None,
        descriptor_filename: Optional[str] = None,
        **kwargs,
    ) -> Self:
        if not resource_path.endswith(".tsv"):
            fname, fext = os.path.splitext(os.path.basename(resource_path))
            raise NotImplementedError(
                f"{fname}: Don't know how to load {fext} files yet."
                f"Either load the resource yourself and use {cls.name}.from_dataframe() or, if you "
                f"want to get a simple path resource, use Resource.from_resource_path() (not "
                f"DimcatResource)."
            )
        df = ms3.load_tsv(resource_path)
        return cls.from_dataframe(
            df=df,
            resource_name=resource_name,
            descriptor_filename=descriptor_filename,
            **kwargs,
        )

    class Schema(Resource.Schema):
        auto_validate = mm.fields.Boolean(metadata={"expose": False})
        default_groupby = mm.fields.List(
            mm.fields.String(), allow_none=True, metadata={"expose": False}
        )

        # @mm.post_load
        # def init_object(self, data, **kwargs):
        #     if "resource" not in data or data["resource"] is None:
        #         return super().init_object(data, **kwargs)
        #     if isinstance(data["resource"], str) and "descriptor_filename" not in data:
        #         if os.path.isabs(data["resource"]):
        #             if "basepath" in data:
        #                 filepath = make_rel_path(data["resource"], data["basepath"])
        #             else:
        #                 basepath, filepath = os.path.split(data["resource"])
        #                 data["basepath"] = basepath
        #         else:
        #             filepath = data["resource"]
        #         data["descriptor_filename"] = filepath
        #     if not isinstance(data["resource"], fl.Resource):
        #         data["resource"] = fl.Resource.from_descriptor(data["resource"])
        #     return super().init_object(data, **kwargs)

    def __init__(
        self,
        resource: fl.Resource = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        """

        Args:
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end either in resource.json or resource.yaml.
            basepath: Where the file would be serialized.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        self.logger.debug(
            f"""
DimcatResource.__init__(
    resource={resource!r},
    descriptor_filename={descriptor_filename!r},
    basepath={basepath!r},
    auto_validate={auto_validate!r},
    default_groupby={default_groupby!r},
)"""
        )
        self._df: D = None
        self.auto_validate = True if auto_validate else False  # catches None
        self._default_groupby: List[str] = []
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
        )
        if default_groupby is not None:
            self.default_groupby = default_groupby

        if self.auto_validate and self.status == ResourceStatus.DATAFRAME:
            _ = self.validate(raise_exception=True)

    def __dir__(self) -> List[str]:
        """Exposes the wrapped dataframe's properties and methods to the IDE."""
        elements = list(super().__dir__())
        if self.is_loaded:
            elements.extend(dir(self.df))
        else:
            # if not loaded, expose the field names from the descriptor
            elements.extend(self.field_names)
        return sorted(elements)

    def __getattr__(self, item):
        """Enables using DimcatResource just like the wrapped DataFrame."""
        msg = f"{self.name!r} object ({self._status!r}) has no attribute {item!r}."
        if not self.is_loaded:
            msg += " Try again after loading the dataframe into memory."
            raise AttributeError(msg)
        try:
            return getattr(self.df, item)
        except AttributeError:
            raise AttributeError(msg)

    def __getitem__(self, item):
        if self.is_loaded:
            try:
                return self.df[item]
            except Exception:
                raise KeyError(item)
        elif item in self.field_names:
            raise KeyError(
                f"Column {item!r} will be available after loading the dataframe into memory."
            )
        raise KeyError(item)

    def __len__(self) -> int:
        return len(self.df.index)

    def __repr__(self) -> str:
        return_str = f"{pformat(self.to_dict(), sort_dicts=False)}"
        return f"ResourceStatus={self.status.name}\n{return_str}"

    @property
    def column_schema(self) -> fl.Schema:
        return self._resource.schema

    @column_schema.setter
    def column_schema(self, new_schema: fl.Schema):
        if self.is_frozen:
            raise ResourceIsFrozenError(
                message="Cannot set schema on a resource whose valid descriptor has been written to disk."
            )
        self._resource.schema = new_schema
        if self.status < ResourceStatus.SCHEMA_ONLY:
            self._status = ResourceStatus.SCHEMA_ONLY
        elif self.status >= ResourceStatus.VALIDATED:
            self._status = ResourceStatus.DATAFRAME
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

    @property
    def default_groupby(self) -> List[str]:
        return list(self._default_groupby)

    @default_groupby.setter
    def default_groupby(self, default_groupby: str | List[str]) -> None:
        if default_groupby is None:
            raise ValueError("default_groupby cannot be None")
        if isinstance(default_groupby, str):
            default_groupby = [default_groupby]
        else:
            default_groupby = list(default_groupby)
        available_levels = self.get_level_names()
        missing = [level for level in default_groupby if level not in available_levels]
        if missing:
            raise ValueError(
                f"Invalid default_groupby: {missing!r} are not valid levels. "
                f"Available levels are: {available_levels!r}"
            )
        self._default_groupby = default_groupby

    @property
    def df(self) -> D:
        if self._df is not None:
            return self._df
        if self.is_frozen:
            return self.get_dataframe()
        raise RuntimeError(f"No dataframe accessible for this {self.name}:\n{self}")

    @df.setter
    def df(self, df: D):
        if self.descriptor_exists:
            raise PotentiallyUnrelatedDescriptorError(
                message=f"Cannot set dataframe on a resource whose valid descriptor has been written to disk. "
                f"Create a new resource via {self.name}.from_descriptor({self.get_descriptor_path()!r})."
            )
        if self.resource_exists:
            raise ResourceIsFrozenError(
                message=f"Cannot set dataframe on a resource {self.resource_name} that's pointing to an existing "
                f"resource {self.normpath}. "
            )
        if self.is_loaded:
            raise RuntimeError("This resource already includes a dataframe.")
        if isinstance(df, DimcatResource):
            df = df.df
        if isinstance(df, pd.Series):
            df = df.to_frame()
            self.logger.info(
                f"Got a series, converted it into a dataframe with column name {df.columns[0]}."
            )
        self._df = df
        if not self.column_schema.fields:
            try:
                self.column_schema = infer_schema_from_df(df)
            except FrictionlessException:
                self.logger.error(f"Could not infer schema from {type(df)}:\n{df}")
                raise
        self._status = ResourceStatus.DATAFRAME
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

    @property
    def field_names(self) -> List[str]:
        """The names of the fields in the resource's schema."""
        return self.column_schema.field_names

    @property
    def innerpath(self) -> str:
        """The innerpath is the resource_name plus the extension .tsv and is used as filename within a .zip archive."""
        if self.resource_name.endswith(".tsv"):
            return self.resource_name
        return self.resource_name + ".tsv"

    @property
    def is_empty(self) -> bool:
        """Whether this resource holds data available or not (yet)."""
        return self.status < ResourceStatus.DATAFRAME

    @property
    def is_loaded(self) -> bool:
        return self._df is not None or self.status in (
            ResourceStatus.STANDALONE_LOADED,
            ResourceStatus.PACKAGED_LOADED,
        )

    @property
    def is_valid(self) -> bool:
        """Returns the result of a previous validation or, if the resource has not been validated
        before, do it now. Importantly, this property assumes serialized resoures to be valid. If
        you want to actively validate the resource, use :meth:`validate` instead."""
        if self.is_serialized:
            return True
        return super().is_valid

    def align_with_grouping(
        self,
        grouping: DimcatIndex | pd.MultiIndex,
        sort_index=True,
    ) -> pd.DataFrame:
        """Aligns the resource with a grouping index. In the typical case, the grouping index will come with the levels
        ["<grouping_name>", "corpus", "piece"] and the result will be aligned such that every group contains the
        resource's sub-dataframes for the included pieces.
        """
        if isinstance(grouping, DimcatIndex):
            grouping = grouping.index
        if self.is_empty:
            self.logger.warning(f"Resource {self.name} is empty.")
            return pd.DataFrame(index=grouping)
        return align_with_grouping(self.df, grouping, sort_index=sort_index)

    def _get_current_status(self) -> ResourceStatus:
        if self.is_packaged:
            if self.is_loaded:
                return ResourceStatus.PACKAGED_LOADED
            else:
                return ResourceStatus.PACKAGED_NOT_LOADED
        match (self.is_serialized, self.descriptor_exists, self.is_loaded):
            case (True, True, True):
                return ResourceStatus.STANDALONE_LOADED
            case (True, True, False):
                return ResourceStatus.STANDALONE_NOT_LOADED
            case (True, False, True):
                return ResourceStatus.SERIALIZED
            case (True, False, False):
                warnings.warn(
                    f"The serialized data exists at {self.normpath!r} but no descriptor was found at "
                    f"{self.get_descriptor_path()!r}. You can create on using .store_descriptor(), set the "
                    f"descriptor_filename pointing to one (should be done upon instantiation), or, if this is supposed "
                    f"to be a PathResource only, it should not be instantiated as DimcatResource at all.",
                    RuntimeWarning,
                )
                return ResourceStatus.PATH_ONLY
            case (False, _, True):
                if self.descriptor_exists:
                    if not self.filepath:
                        raise RuntimeError(
                            f"The resource points to an existing descriptor at {self.get_descriptor_path()!r} but "
                            f"no filepath has been set. This should not have happened. Please consider filing an issue."
                        )
                    warnings.warn(
                        f"The resource is loaded and the there exists a descriptor at {self.get_descriptor_path()!r}, "
                        f"but the normpath {self.normpath} does not exist. This could signify a mismatch between the "
                        f"loaded dataframe and the data described by the descriptor which could result in data loss if "
                        f"the dataframe is serialized to disk, overwriting the descriptor that was actually describing "
                        f"something else.",
                        PotentiallyUnrelatedDescriptorUserWarning,
                    )
                if self._is_valid:  # using the property could trigger validation
                    return ResourceStatus.VALIDATED
                return ResourceStatus.DATAFRAME
            case _:
                if self.basepath and self.descriptor_exists:
                    warnings.warn(
                        f"The resource points to an existing descriptor at {self.get_descriptor_path()!r} but it "
                        f"hasn't been loaded. Please consider passing discriptor_filename="
                        f"{self.get_descriptor_filename()} when instantiating or using {self.name}"
                        f".from_descriptor_path(). If this is what you did, this warning likely stems from a bug, "
                        f"please consider filing an issue in this case.",
                        PotentiallyUnrelatedDescriptorUserWarning,
                    )
                if self.column_schema.fields:
                    return ResourceStatus.SCHEMA_ONLY
                return ResourceStatus.EMPTY

    @cache
    def get_dataframe(self) -> D:
        """
        Load the dataframe from disk based on the descriptor's normpath.

        Returns:
            The dataframe or DimcatResource.
        """
        dataframe = load_fl_resource(self._resource)
        if self.status == ResourceStatus.STANDALONE_NOT_LOADED:
            self._status = ResourceStatus.STANDALONE_LOADED
        elif self.status == ResourceStatus.PACKAGED_NOT_LOADED:
            self._status = ResourceStatus.PACKAGED_LOADED
        return dataframe

    def get_default_groupby(self) -> List[str]:
        """Returns the default index levels for grouping the resource."""
        if not self.default_groupby:
            return self.get_grouping_levels()
        return self.default_groupby

    def get_grouping_levels(self) -> List[str]:
        """Returns the levels of the grouping index."""
        return self.get_piece_index(max_levels=0).names

    def get_index(self) -> DimcatIndex:
        """Returns the index of the resource based on the ``primaryKey`` of the :obj:`frictionless.Schema`."""
        return DimcatIndex.from_resource(self)

    def get_level_names(self) -> List[str]:
        """Returns the level names of the resource's index."""
        return self.get_index().names

    def get_normpath(
        self,
        set_default_if_missing=False,
    ) -> str:
        try:
            return self.normpath
        except (BasePathNotDefinedError, FilePathNotDefinedError):
            return os.path.join(
                self.get_basepath(set_default_if_missing=set_default_if_missing),
                self.get_filepath(set_default_if_missing=set_default_if_missing),
            )

    def get_piece_index(self, max_levels: int = 2) -> PieceIndex:
        """Returns the :class:`PieceIndex` of the resource based on :attr:`get_index`. That is,
        an index of which the right-most level is unique and called `piece` and up to ``max_levels``
        additional index levels to its right.

        Args:
            max_levels: By default, the number of levels is limited to the default 2, ('corpus', 'piece').

        Returns:
            An index of the pieces described by the resource.
        """
        return PieceIndex.from_resource(self, max_levels=max_levels)

    def load(self, force_reload: bool = False) -> None:
        """Tries to load the data from disk into RAM. If successful, the .is_loaded property will be True.
        If the resource hadn't been loaded before, its .status property will be updated.
        """
        if not self.is_loaded or force_reload:
            _ = self.df

    def _make_empty_fl_resource(self):
        """Create an empty frictionless resource object with a minimal descriptor."""
        return make_tsv_resource()

    def set_basepath(
        self,
        basepath: str,
        reconcile: bool = False,
    ) -> None:
        super().set_basepath(
            basepath=basepath,
            reconcile=reconcile,
        )
        if self.auto_validate:
            _ = self.validate(raise_exception=True)

    def subselect(
        self,
        tuples: DimcatIndex | Iterable[tuple],
        levels: Optional[int | str | List[int | str]] = None,
    ) -> pd.DataFrame:
        """Returns a copy of a subselection of the dataframe based on the union of its index tuples (or subtuples)
        and the given tuples."""
        if self.is_empty:
            self.logger.warning("Resource is empty.")
            return self.copy()
        tuple_set = set(tuples)
        random_tuple = next(iter(tuple_set))
        if not isinstance(random_tuple, tuple):
            raise TypeError(
                f"Pass an iterable of tuples. A randomly selected element had type {type(random_tuple)!r}."
            )
        mask = make_boolean_mask_from_set_of_tuples(self.df.index, tuple_set, levels)
        return self.df[mask].copy()

    def store_dataframe(self, overwrite=False, validate: bool = True) -> None:
        """Stores the dataframe and its descriptor to disk based on the resource's configuration.

        Args:
            overwrite:
            validate:

        Raises:
            RuntimeError: If the resource is frozen or does not contain a dataframe or if the file exists already.
        """
        full_path = self.get_normpath(set_default_if_missing=True)
        if not overwrite and self.resource_exists:
            FileExistsError(
                f"Pass overwrite=True if you want to overwrite the existing {full_path}"
            )
        if self.status < ResourceStatus.DATAFRAME:
            raise RuntimeError(f"This {self.name} does not contain a dataframe.")
        ms3.write_tsv(self.df.reset_index(), full_path)
        self.logger.info(f"{self.name} serialized to {full_path}.")
        self.store_descriptor()
        if validate:
            report = self.validate(raise_exception=False)
            if report.valid:
                self.logger.info(f"Resource stored to {full_path} and validated.")
            else:
                errors = "\n".join(
                    str(err.message) for task in report.tasks for err in task.errors
                )
                msg = f"The resource did not validate after being stored to {full_path}:\n{errors}"
                if get_setting("never_store_unvalidated_data"):
                    os.remove(full_path)
                    self.logger.info(
                        msg
                        + "\nThe file was deleted because of the 'never_store_unvalidated_data' setting."
                    )
                self.logger.warning(msg)
        self._status = ResourceStatus.STANDALONE_LOADED

    def update_default_groupby(self, new_level_name: str) -> None:
        """Updates the value of :attr:`default_groupby` by prepending the new level name to it."""
        current_default = self.get_default_groupby()
        if current_default[0] == new_level_name:
            self.logger.debug(
                f"Default levels already start with {new_level_name!r}: {current_default}."
            )
            new_default_value = current_default
        else:
            new_default_value = [new_level_name] + current_default
            self.logger.debug(
                f"Updating default levels from {current_default} to {new_default_value}."
            )
        self.default_groupby = new_default_value

    def validate(
        self,
        raise_exception: bool = False,
        only_if_necessary: bool = False,
    ) -> Optional[fl.Report]:
        """Validate the resource's data against its descriptor.

        Args:
            raise_exception: (default False) Pass True to raise if the resource is not valid.
            only_if_necessary:
                (default False) Pass True to skip validation if the resource has already been validated or is
                assumed to be valid because it exists on disk.

        Returns:
            None if no validation took place (e.g. because resource is empty or ``only_if_necessary`` was True).
            Otherwise, frictionless report resulting from validating the data against the :attr:`column_schema`.

        Raises:
            FrictionlessException: If the resource is not valid and ``raise_exception`` is True.
        """
        if self.is_empty:
            self.logger.info("Nothing to validate.")
            return
        if only_if_necessary and (
            self._is_valid is not None or self.status >= ResourceStatus.VALIDATED
        ):
            self.logger.info("Already validated.")
            return
        if self.is_serialized:
            report = self._resource.validate()
        else:
            tmp_resource = fl.Resource(self.df)
            tmp_resource.schema = self.column_schema
            report = tmp_resource.validate()
        if report.valid:
            if self.status < ResourceStatus.VALIDATED:
                self._status = ResourceStatus.VALIDATED
        else:
            errors = [err.message for task in report.tasks for err in task.errors]
            if self.status == ResourceStatus.VALIDATED:
                self._status = ResourceStatus.DATAFRAME
            if get_setting("never_store_unvalidated_data") and raise_exception:
                raise fl.FrictionlessException("\n".join(errors))
        return report


class IndexField(mm.fields.Field):
    def _serialize(self, value, attr, obj, **kwargs):
        return value.to_list()


class DimcatIndex(Generic[IX], Data):
    """A wrapper around a :obj:`pandas.MultiIndex` that provides additional functionality such as keeping track of
    index levels and default groupings.

    A MultiIndex essentially is a Sequence of tuples where each tuple identifies dataframe row and includes one value
    per index level. Each index level has a name and can be seen as in individual :obj:`pandas.Index`. One important
    type of DimcatIndex is the PieceIndex which is a unique MultiIndex (that is, each tuple is unique) and where the
    last (i.e. right-most) level is named `piece`.

    NB: If you want to use the index in a dataframe constructor, use the actual, wrapped index object as in
    `pd.DataFrame(index=dc_index.index)`.
    """

    class Schema(Data.Schema):
        index = IndexField(required=True)
        names = mm.fields.List(mm.fields.Str(), required=True)

        @mm.post_load
        def init_object(self, data, **kwargs) -> pd.MultiIndex:
            return pd.MultiIndex.from_tuples(data["index"], names=data["names"])

    @classmethod
    def from_dataframe(cls, df: SomeDataframe) -> Self:
        """Create a DimcatIndex from a dataframe."""
        return cls.from_index(df.index)

    @classmethod
    def from_grouping(
        cls,
        grouping: Dict[str, List[tuple]],
        level_names: Sequence[str] = ("piece_group", "corpus", "piece"),
        sort: bool = False,
        raise_if_multiple_membership: bool = False,
    ) -> Self:
        """Creates a DimcatIndex from a dictionary of piece groups.

        Args:
        grouping: A dictionary where keys are group names and values are lists of index tuples.
        names: Names for the levels of the MultiIndex, i.e. one for the group level and one per level in the tuples.
        sort: By default the returned MultiIndex is not sorted. Set False to enable sorting.
        raise_if_multiple_membership: If True, raises a ValueError if a member is in multiple groups.
        """
        grouping = make_index_from_grouping_dict(
            grouping=grouping,
            level_names=level_names,
            sort=sort,
            raise_if_multiple_membership=raise_if_multiple_membership,
        )
        return cls.from_index(grouping, max_levels=0)

    @classmethod
    def from_index(cls, index: SomeIndex, **kwargs) -> Self:
        """Create a DimcatIndex from a dataframe index."""
        return cls(index)

    @classmethod
    def from_resource(
        cls,
        resource: DimcatResource | fl.Resource,
        index_col: Optional[int | str | List[int | str]] = None,
    ) -> Self:
        """Create a DimcatIndex from a frictionless Resource."""
        if isinstance(resource, DimcatResource):
            if resource.status < ResourceStatus.DATAFRAME:
                return cls()
            if resource.is_loaded:
                return cls(resource.df.index)
            fl_resource = resource.resource
        elif isinstance(resource, fl.Resource):
            fl_resource = resource
        else:
            raise TypeError(
                f"Expected DimcatResource or frictionless.Resource, got {type(resource)!r}."
            )
        # load only the index columns from the serialized resource
        index = load_index_from_fl_resource(fl_resource, index_col=index_col)
        return cls(index)

    @classmethod
    def from_tuples(
        cls,
        tuples: Iterable[tuple],
        names: Sequence[str],
    ) -> Self:
        list_of_tuples = list(tuples)
        if len(list_of_tuples) == 0:
            return cls()
        first_tuple = list_of_tuples[0]
        if not isinstance(first_tuple, tuple):
            raise ValueError(f"Expected tuples, got {type(first_tuple)!r}.")
        if len(first_tuple) != len(names):
            raise ValueError(
                f"Expected tuples of length {len(names)}, got {len(first_tuple)}."
            )
        multiindex = pd.MultiIndex.from_tuples(list_of_tuples, names=names)
        return cls(multiindex)

    def __init__(
        self,
        index: Optional[IX] = None,
        basepath: Optional[str] = None,
    ):
        super().__init__(basepath=basepath)
        if index is None:
            self._index = pd.MultiIndex.from_tuples([], names=["corpus", "piece"])
        elif isinstance(index, pd.Index):
            if None in index.names:
                raise ValueError("Index cannot have a None name: {index.names}.")
            for name in index.names:
                check_name(name)
            self._index = index.copy()
        else:
            raise TypeError(
                f"Expected None or pandas.(Multi)Index, got {type(index)!r}."
            )

    def __contains__(self, item):
        if isinstance(item, tuple):
            return item in set(self._index)
        if isinstance(item, Iterable):
            return set(item).issubset(set(self._index))
        return False

    def __eq__(self, other) -> bool:
        if isinstance(other, Iterable):
            return set(self) == set(other)
        return False

    def __getattr__(self, item):
        """Enables using DimcatIndex just like the wrapped Index object."""
        return getattr(self._index, item)

    def __getitem__(self, item):
        """Enables using DimcatIndex just like the wrapped Index object."""
        result = self._index[item]
        if isinstance(result, pd.Index):
            return self.__class__(result)
        return result

    def __hash__(self):
        return hash(set(self._index))

    def __iter__(self):
        return iter(self._index)

    def __len__(self) -> int:
        return len(self._index)

    def __repr__(self) -> str:
        return repr(self._index)

    def __str__(self) -> str:
        return str(self._index)

    @property
    def index(self) -> IX:
        return self._index

    @property
    def names(self) -> List[str]:
        return list(self._index.names)

    @property
    def piece_level_position(self) -> Optional[int]:
        """The position of the `piece` level in the index, or None if the index has no `piece` level."""
        return self.names.index("piece") if "piece" in self.names else None

    def copy(self) -> Self:
        return self.__class__(self._index.copy())

    def sample(self, n: int) -> Self:
        """Return a random sample of n elements."""
        as_series = self._index.to_series()
        sample = as_series.sample(n)
        as_index = pd.MultiIndex.from_tuples(sample, names=self.names)
        return self.__class__(as_index)

    def to_resource(self, **kwargs) -> DimcatResource:
        """Create a DimcatResource from this index."""
        return DimcatResource.from_index(self, **kwargs)


class PieceIndex(DimcatIndex[IX]):
    """A unique DimcatIndex where the last (i.e. right-most) level is named `piece`."""

    @classmethod
    def from_index(
        cls,
        index: DimcatIndex[IX] | IX,
        recognized_piece_columns: Optional[Iterable[str]] = None,
        max_levels: int = 2,
    ) -> Self:
        """Create a PieceIndex from another index."""
        if isinstance(index, DimcatIndex):
            index = index.index
        if len(index) == 0:
            return cls()
        index, piece_level_position = ensure_level_named_piece(
            index, recognized_piece_columns
        )
        level_names = index.names
        right_boundary = piece_level_position + 1
        drop_levels = level_names[right_boundary:]
        if max_levels > 0 and piece_level_position >= max_levels:
            drop_levels = level_names[: right_boundary - max_levels] + drop_levels
        if len(drop_levels) > 0:
            index = index.droplevel(drop_levels)
        return cls(index)

    @classmethod
    def from_resource(
        cls,
        resource: DimcatResource | fl.Resource,
        index_col: Optional[int | str | List[int | str]] = None,
        recognized_piece_columns: Optional[Iterable[str]] = None,
        max_levels: int = 2,
    ) -> Self:
        """Create a PieceIndex from a frictionless Resource."""
        index = DimcatIndex.from_resource(
            resource,
            index_col=index_col,
        )
        return cls.from_index(
            index,
            recognized_piece_columns=recognized_piece_columns,
            max_levels=max_levels,
        )

    @classmethod
    def from_tuples(
        cls,
        tuples: Iterable[tuple],
        names: Sequence[str] = ("corpus", "piece"),
    ) -> Self:
        return super().from_tuples(tuples, names)

    def __init__(self, index: Optional[IX] = None):
        if index is None:
            index = pd.MultiIndex.from_tuples([], name=("corpus", "piece"))
        else:
            index = index.drop_duplicates()
            assert (
                index.names[-1] == "piece"
            ), f"Expected last level to be named 'piece', got {index.names[-1]!r}."
        super().__init__(index)

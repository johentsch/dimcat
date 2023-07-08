from __future__ import annotations

import logging
import os
import warnings
import zipfile
from enum import IntEnum, auto
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, TypeAlias, TypeVar, Union

import frictionless as fl
import marshmallow as mm
import pandas as pd
from dimcat.base import get_class, get_setting, is_default_descriptor_path
from dimcat.data.base import Data
from dimcat.exceptions import (
    BasePathNotDefinedError,
    FilePathNotDefinedError,
    InvalidResourcePathError,
    ResourceIsFrozenError,
)
from dimcat.utils import make_valid_frictionless_name, replace_ext, resolve_path
from typing_extensions import Self

from .utils import (
    check_descriptor_filepath_argument,
    is_default_package_descriptor_path,
    make_fl_resource,
    make_rel_path,
    store_as_json_or_yaml,
)

try:
    import modin.pandas as mpd

    SomeDataframe: TypeAlias = Union[pd.DataFrame, mpd.DataFrame]
    SomeSeries: TypeAlias = Union[pd.Series, mpd.Series]
    SomeIndex: TypeAlias = Union[pd.Index, mpd.Index]
except ImportError:
    # DiMCAT has not been installed via dimcat[modin], hence the dependency is missing
    SomeDataframe: TypeAlias = pd.DataFrame
    SomeSeries: TypeAlias = pd.Series
    SomeIndex: TypeAlias = pd.Index

logger = logging.getLogger(__name__)

D = TypeVar("D", bound=SomeDataframe)
S = TypeVar("S", bound=SomeSeries)
IX = TypeVar("IX", bound=SomeIndex)

# region Resource


def reconcile_base_and_file(
    basepath: Optional[str],
    filepath: str,
) -> Tuple[str, str]:
    """The result is a tuple of an absolute basepath (or None) and a relative filepath."""
    assert filepath is not None, "filepath must not be None"
    if not basepath:
        if os.path.isabs(filepath):
            base, file = os.path.split(filepath)
        else:
            base, file = os.getcwd(), filepath
    else:
        if os.path.isabs(filepath):
            base = basepath
            file = make_rel_path(filepath, basepath)
        else:
            base = basepath
            file = filepath
    return resolve_path(base), file


class ResourceStatus(IntEnum):
    """Expresses the status of a DimcatResource with respect to it being described, valid, and serialized to disk,
    with or without its descriptor file. The enum members have increasing integer values starting with EMPTY == 0.
    Statuses > PATH_ONLY (1) are currently only relevant for DimcatResources. The current status is determined
    by the boolean state of the first three attributes in the table below:

    * is_serialized: True if the resource can be located physically on disk.
    * descriptor_exists: True if a descriptor file (JSON/YAML) is physically present on disk.
    * is_loaded: True if the resource is currently loaded into memory.

    The remaining attributes are derived from the first three and are not used to determine the current status:

    * assumed valid: True if the resource is assumed to be valid, which is the case for all serialized resources.
    * standalone: True if the resource is not part of a package. For "free" (not serialized) resources, it depends
      on the value :attr:`Resource.descriptor_filepath` (whether it corresponds to a package or resource descriptor).
    * empty: True if the resource is empty, i.e. it does not data. A DimcatResource that is PATH_ONLY is considered
      empty, whereas a Resource/PathResource is not (they only have status 0 or 1).

    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | ResourceStatus        | is_serialized | descriptor_exists | is_loaded | assumed valid | standalone | empty |
    +=======================+===============+===================+===========+===============+============+=======+
    | EMPTY                 | False         | ?                 | False     |       no      |      ?     |  yes  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | PATH_ONLY             | True          | ?                 | False     |       no      |      ?     |  yes  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | SCHEMA_ONLY           | False         | ?                 | False     |       no      |      ?     |  yes  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | DATAFRAME             | False         | False             | True      |       no      |      ?     |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | VALIDATED             | False         | False             | True      |   guaranteed  |      ?     |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | SERIALIZED            | True          | False             | True      |      yes      |     yes    |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | STANDALONE_LOADED     | True          | True              | True      |      yes      |     yes    |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | PACKAGED_LOADED       | True          | True              | True      |      yes      |     no     |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | STANDALONE_NOT_LOADED | True          | True              | False     |      yes      |     yes    |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+
    | PACKAGED_NOT_LOADED   | True          | True              | False     |      yes      |     no     |   no  |
    +-----------------------+---------------+-------------------+-----------+---------------+------------+-------+

    The status of a resource is set at the end of :meth:`Resource.__init__` by
    calling :meth:`Resource._update_status` which, in return calls :meth:`Resource._get_status`.
    """

    EMPTY = 0
    PATH_ONLY = auto()  # only path exists (default in a PathResource)
    SCHEMA_ONLY = auto()  # column_schema available but no dataframe has been loaded
    DATAFRAME = (
        auto()
    )  # dataframe available in memory but not yet validated against the column_schema
    VALIDATED = auto()  # validated dataframe available in memory
    SERIALIZED = (
        auto()
    )  # dataframe serialized to disk but not its descriptor (shouldn't happen) -> can be changed or overwritten
    STANDALONE_LOADED = auto()
    PACKAGED_LOADED = auto()
    STANDALONE_NOT_LOADED = auto()
    PACKAGED_NOT_LOADED = auto()


class ResourceSchema(Data.Schema):
    """Several fields are serialized through the frictionless descriptor "resource" because
    many of the :class:`Resource` object's properties use the wrapped fl.Resource object's fields.
    For example, :attr:`resource_name` uses ``.resource.name`` under the hood.
    """

    basepath = mm.fields.Str(
        required=False,
        allow_none=True,
        metadata=dict(
            description="The directory where the resource is or would be stored."
        ),
    )
    descriptor_filepath = mm.fields.String(allow_none=True, metadata={"expose": False})
    resource = mm.fields.Method(
        serialize="get_frictionless_descriptor",
        deserialize="raw",
        metadata={"expose": False},
    )

    def get_frictionless_descriptor(self, obj: Resource) -> dict:
        return obj._resource.to_dict()

    def raw(self, data):
        return data

    @mm.pre_load
    def unsquash_data_if_necessary(self, data, **kwargs):
        """Data serialized with this schema usually has 'resource' field that contains the frictionless descriptor.
        However, if it has been serialized with the PickleSchema variant, this descriptor has become the top level
        and all other fields have been squashed into it, effectively flattening the dictionary. This method
        reverses this flattening, if necessary.
        """
        if "resource" in data:
            return data
        if isinstance(data, fl.Resource):
            fl_resource = data
        else:
            fl_resource = fl.Resource.from_descriptor(data)
        unsquashed_data = dict(fl_resource.custom)
        fl_resource.custom = {}
        assert fl_resource.custom == {}
        unsquashed_data["resource"] = fl_resource
        return unsquashed_data

    @mm.post_load
    def init_object(self, data, **kwargs):
        if "resource" not in data or data["resource"] is None:
            # probably manually compiled data
            return super().init_object(data, **kwargs)
        if not isinstance(data["resource"], fl.Resource):
            data["resource"] = fl.Resource.from_descriptor(data["resource"])
        return super().init_object(data, **kwargs)


class Resource(Data):
    """A Resource is essentially a wrapper around a :obj:`frictionless.Resource` object. In its
    simple form, it serves merely for storing a file path, but split into a basepath and a relative
    filepath, as per the frictionless philosophy.
    """

    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict,
        descriptor_filepath: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a frictionless descriptor dictionary.

        Args:
            descriptor: Descriptor corresponding to a frictionless resource descriptor.
            descriptor_filepath:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filepath`. Needs to end either in resource.json or resource.yaml.
            basepath: Where the file would be serialized.
            **kwargs: Subclasses can use this method.

        Returns:

        """
        if isinstance(descriptor, (str, Path)):
            raise TypeError(
                f"This method expects a descriptor dictionary. In order to create a "
                f"{cls.name} from a path, use {cls.__name__}.from_descriptor_path() instead."
            )
        if isinstance(descriptor, fl.Resource):
            fl_resource = descriptor
        else:
            fl_resource = fl.Resource.from_descriptor(descriptor)
        if dtype := fl_resource.custom.get("dtype"):
            # the descriptor.custom dict contains serialization data for a DiMCAT object so we will deserialize
            # it with the appropriate dtype class constructor
            Constructor = get_class(dtype)
            if not issubclass(Constructor, cls):
                raise TypeError(
                    f"The descriptor specifies dtype {dtype!r} which is not a subclass of {cls.name}."
                )
            descriptor = dict(
                descriptor,
                descriptor_filepath=descriptor_filepath,
                basepath=basepath,
                **kwargs,
            )
            return Constructor.schema.load(descriptor)
        elif fl_resource.custom != {}:
            warnings.warn(
                f"The descriptor contains unknown data: {fl_resource.custom}.",
                RuntimeWarning,
            )
        return cls(
            resource=fl_resource,
            descriptor_filepath=descriptor_filepath,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_descriptor_path(
        cls,
        descriptor_path: str,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a frictionless descriptor file on disk.

        Args:
            descriptor_path: Absolute path where the JSON/YAML descriptor is located.
            basepath:
                If you do not want the folder where the descriptor is located to be treated as basepath,
                you may specify an absolute path higher up within the ``descriptor_path`` to serve as base.
                The resource's filepath will be adapated accordingly, whereas the resource names
                specified in the descriptor will remain the same.
            **kwargs: Subclasses can use this method.
        """
        basepath, descriptor_filepath = reconcile_base_and_file(
            basepath, descriptor_path
        )
        fl_resource = fl.Resource.from_descriptor(descriptor_path)
        fl_resource.path = make_rel_path(
            fl_resource.normpath, basepath
        )  # adapt the relative path to the basepath
        return cls.from_descriptor(
            descriptor=fl_resource.to_dict(),
            descriptor_filepath=descriptor_filepath,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_filepath(
        cls,
        filepath: str,
        resource_name: Optional[str] = None,
        descriptor_filepath: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a file on disk, be it a JSON/YAML resource descriptor, or a simple path resource.

        Args:
            filepath: Path pointing to a resource descriptor or a simple path resource.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            descriptor_filepath:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filepath`. Needs to end either in resource.json or resource.yaml.
            basepath:
                If you do not want the folder where the file is located to be treated as basepath,
                you may specify an absolute path higher up within the ``filepath`` to serve as base.
                The resource's filepath will be adapated accordingly, whereas the resource names
                specified in the descriptor will remain the same.
            auto_validate:
                By default, the Resource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).

        """
        if filepath.endswith("resource.json") or filepath.endswith("resource.yaml"):
            return cls.from_descriptor_path(
                descriptor_path=descriptor_filepath, basepath=basepath, **kwargs
            )
        return super().from_resource_path(
            filepath=filepath,
            resource_name=resource_name,
            descriptor_filepath=descriptor_filepath,
            basepath=basepath,
            **kwargs,
        )

    @classmethod
    def from_resource(
        cls,
        resource: Resource,
        resource_name: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ):
        """Create a Resource from an existing :obj:`Resource`, specifying its name and,
        optionally, at what path it is to be serialized.

        Args:
            resource: An existing :obj:`frictionless.Resource` or a filepath.
            resource_name:
                Name of the resource used for retrieving it from a DimcatPackage and as filename when the resource
                is stored to a ZIP file.
            basepath:
                Lets you change the basepath of the existing resource.
            **kwargs: Subclasses can use this method.
        """
        if not isinstance(resource, Resource):
            raise TypeError(f"Expected a Resource, got {type(resource)!r}.")
        fl_resource = resource.resource.to_copy()
        descriptor_filepath = resource.descriptor_filepath
        basepath = basepath if basepath else resource.basepath
        new_object = cls(
            resource=fl_resource,
            descriptor_filepath=descriptor_filepath,
            basepath=basepath,
            **kwargs,
        )
        if resource_name:
            new_object.resource_name = resource_name
        new_object._corpus_name = resource._corpus_name
        return new_object

    @classmethod
    def from_resource_path(
        cls,
        resource_path: str,
        resource_name: Optional[str] = None,
        descriptor_filepath: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a file on disk, treating it just as a path even if it's a
        JSON/YAML resource descriptor"""
        if is_default_descriptor_path(resource_path):
            warnings.warn(
                f"You have passed the descriptor path {resource_path!r} to {cls.name}.from_resource_path()"
                f" meaning that the descriptor itself will be treated like a resource and not the resource "
                f"it describes. You may want to use {cls.name}.from_descriptor_path() instead.",
                SyntaxWarning,
            )
        basepath, resource_path = reconcile_base_and_file(basepath, resource_path)
        fname, extension = os.path.splitext(resource_path)
        if not resource_name:
            resource_name = make_valid_frictionless_name(fname)
        options = dict(
            name=resource_name,
            path=resource_path,
            scheme="file",
            format=extension[1:],
        )
        fl_resource = make_fl_resource(**options)
        return cls(
            resource=fl_resource,
            descriptor_filepath=descriptor_filepath,
            basepath=basepath,
            **kwargs,
        )

    class PickleSchema(ResourceSchema):
        @mm.post_dump()
        def squash_data_for_frictionless(self, data, **kwargs):
            squashed_data = data.pop("resource")
            obj_basepath, desc_basepath = data.get("basepath"), squashed_data.get(
                "basepath"
            )
            if (obj_basepath and desc_basepath) and obj_basepath != desc_basepath:
                # first, reconcile potential discrepancy between basepaths
                # by default, the fields of the resource descriptor are overwritten by the fields of the resource object
                filepath = squashed_data.get("path")
                if os.path.isfile(
                    (obj_normpath := os.path.join(obj_basepath, filepath))
                ):
                    self.logger.error(
                        f"Giving the object's basepath {obj_basepath!r} precedence over the descriptor's "
                        f"basepath ({desc_basepath!r}) because it exists."
                    )
                elif os.path.isfile(os.path.join(desc_basepath, filepath)):
                    del data["basepath"]
                    self.logger.error(
                        f"Using the descriptor's basepath {desc_basepath!r} because the object's basepath "
                        f"would result to the invalid path {obj_normpath!r}."
                    )
                else:
                    raise FileNotFoundError(
                        f"Neither the object's basepath {obj_basepath!r} nor the descriptor's basepath "
                        f"{desc_basepath!r} contain the file {filepath!r}."
                    )
            squashed_data.update(data)
            return squashed_data

    class Schema(ResourceSchema):
        pass

    def __init__(
        self,
        resource: fl.Resource = None,
        descriptor_filepath: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ):
        """

        Args:
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filepath:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filepath`. Needs to end either in resource.json or resource.yaml.
            basepath: Where the file would be serialized.
        """
        self.logger.debug(
            f"""
Resource.__init__(
    resource={resource},
    descriptor_filepath={descriptor_filepath},
    basepath={basepath},
    **kwargs={kwargs},
)"""
        )
        self._status = ResourceStatus.EMPTY
        self._resource: fl.Resource = self._make_empty_fl_resource()
        self._corpus_name: Optional[str] = None
        self._is_valid: Optional[bool] = None
        is_fl_resource = isinstance(resource, fl.Resource)
        if is_fl_resource and basepath is None:
            basepath = resource.basepath
        super().__init__(basepath=basepath)
        if is_fl_resource:
            self._resource = resource
        elif resource is None:
            pass
        else:
            raise TypeError(
                f"Expected resource to be a frictionless Resource or a file path, got {type(resource)}."
            )
        if self.basepath:
            self._resource.basepath = self.basepath
        if descriptor_filepath:
            self.descriptor_filepath = descriptor_filepath
        self._update_status()
        self.logger.debug(
            f"""
Resource(
    basepath={self.basepath},
    filepath={self.filepath},
    corpus_name={self.get_corpus_name()},
    resource_name={self.resource_name},
    descriptor_filepath={self.descriptor_filepath},
)"""
        )

    @property
    def basepath(self) -> str:
        return self._basepath

    @basepath.setter
    def basepath(self, basepath: str):
        self.set_basepath(basepath)

    @property
    def corpus_name(self) -> Optional[str]:
        """The name of the corpus this resource belongs to."""
        return self._corpus_name

    @corpus_name.setter
    def corpus_name(self, corpus_name: str):
        valid_name = make_valid_frictionless_name(corpus_name)
        if valid_name != corpus_name:
            self.logger.info(f"Changed {corpus_name!r} name to {valid_name!r}.")
        self._corpus_name = corpus_name

    @property
    def descriptor_filepath(self) -> Optional[str]:
        """The path to the descriptor file on disk, relative to the basepath. If you need to fall back to a default
        value, use :meth:`get_descriptor_filepath` instead."""
        return self._resource.metadata_descriptor_path

    @descriptor_filepath.setter
    def descriptor_filepath(self, descriptor_filepath: str):
        check_descriptor_filepath_argument(descriptor_filepath)
        self._resource.metadata_descriptor_path = descriptor_filepath
        # self._set_descriptor_path(descriptor_filepath)

    @property
    def descriptor_exists(self) -> bool:
        descriptor_path = self.get_descriptor_path()
        if not descriptor_path:
            return False
        return os.path.isfile(descriptor_path)

    @property
    def filepath(self) -> str:
        return self._resource.path

    @filepath.setter
    def filepath(self, filepath: str):
        if os.path.isabs(filepath):
            raise ValueError(f"Filepath must be relative, got {filepath!r}.")
        self._resource.path = filepath

    @property
    def ID(self) -> Tuple[str, str]:
        """The resource's unique ID."""
        if not self.resource_name:
            raise ValueError("Resource name not set.")
        corpus_name = self.get_corpus_name()
        return (corpus_name, self.resource_name)

    @ID.setter
    def ID(self, ID: Tuple[str, str]):
        self.corpus_name, self.resource_name = ID
        self.logger.debug(f"Resource ID updated to {self.ID!r}.")

    @property
    def innerpath(self) -> Optional[str]:
        """If this is a zipped resource, the innerpath is the resource's filepath within the zip."""
        return self._resource.innerpath

    @property
    def is_empty(self) -> bool:
        return self._status == ResourceStatus.EMPTY

    @property
    def is_frozen(self) -> bool:
        """Whether the resource is frozen (i.e. it's pointing to data on the disk) or not."""
        return self.resource_exists or self.descriptor_exists

    @property
    def is_loaded(self) -> bool:
        return False

    @property
    def is_valid(self) -> bool:
        """Returns the result of a previous validation or, if the resource has not been validated
        before, do it now."""
        report = self.validate(raise_exception=False, only_if_necessary=True)
        if report is None:
            return True
        return report.valid

    @property
    def is_packaged(self) -> bool:
        """Returns True if the resource is packaged, i.e. its descriptor_filepath is the one of
        the :class:`Package` it belongs to. Also means that the resource is passive."""
        return self.descriptor_filepath and is_default_package_descriptor_path(
            self.descriptor_filepath
        )

    @property
    def is_serialized(self) -> bool:
        """Returns True if the resource is serialized, i.e. it points to a file on disk and, if it
        is a ZIP file, the :attr:`innerpath` is present in that ZIP file."""
        if not self.resource_exists:
            return False
        if self.is_zipped:
            with zipfile.ZipFile(self.normpath) as zip_file:
                return self.innerpath in zip_file.namelist()
        return True

    @property
    def is_zipped(self) -> bool:
        """Returns True if the filepath points to a .zip file."""
        if not self.filepath:
            return False
        return self.filepath.endswith(".zip")

    @property
    def normpath(self) -> str:
        """Absolute path to the serialized or future tabular file. Raises if basepath is not set."""
        if not self.basepath:
            raise BasePathNotDefinedError
        if not self.filepath:
            raise FilePathNotDefinedError
        return os.path.join(self.basepath, self.filepath)

    @property
    def resource(self) -> fl.Resource:
        return self._resource

    @property
    def resource_exists(self) -> bool:
        """Returns True if the resource's normpath exists on disk.
        If the resource :attr:`is_zipped` and you want to check if the :attr:`innerpath` actually
        exists within the ZIP file, use :attr:`is_serialized` instead."""
        try:
            return os.path.isfile(self.normpath)
        except (BasePathNotDefinedError, FilePathNotDefinedError):
            return False

    @property
    def resource_name(self) -> str:
        return self._resource.name

    @resource_name.setter
    def resource_name(self, resource_name: str):
        valid_name = make_valid_frictionless_name(resource_name)
        if valid_name != resource_name:
            self.logger.info(f"Changed {resource_name!r} name to {valid_name!r}.")
        self._resource.name = resource_name
        if not self._resource.path:
            self._resource.path = self.innerpath

    @property
    def status(self) -> ResourceStatus:
        if self._status == ResourceStatus.EMPTY and self._resource.schema.fields:
            self._status = ResourceStatus.SCHEMA_ONLY
        return self._status

    def copy(self) -> Self:
        """Returns a copy of the resource."""
        return self.from_resource(self)

    def to_dict(self, pickle: bool = False) -> Dict[str, Any]:
        """Returns a dictionary representation of the resource and stores its descriptor to disk."""
        if not pickle:
            return super().to_dict()
        descriptor_path = self.get_descriptor_path(fallback_to_default=True)
        descriptor_dict = self.make_descriptor()

        store_as_json_or_yaml(descriptor_dict, descriptor_path)
        return descriptor_dict

    def get_corpus_name(self) -> str:
        """Returns the value of :attr:`corpus_name` or, if not set, a name derived from the
        resource's filepath.

        Raises:
            ValueError: If neither :attr:`corpus_name` nor :attr:`filepath` are set.
        """

        def return_basepath_name() -> str:
            if self.basepath is None:
                return
            return make_valid_frictionless_name(os.path.basename(self.basepath))

        if self.corpus_name:
            return self.corpus_name
        if not self.filepath:
            return return_basepath_name()
        folder, _ = os.path.split(self.filepath)
        folder = folder.rstrip(os.sep)
        if not folder or folder == ".":
            return return_basepath_name()
        folder_split = folder.split(os.sep)
        if len(folder_split) > 1:
            return make_valid_frictionless_name(folder_split[-1])
        return make_valid_frictionless_name(folder)

    def _get_current_status(self) -> ResourceStatus:
        if self.filepath:
            return ResourceStatus.PATH_ONLY
        return ResourceStatus.EMPTY

    def get_descriptor_filepath(self) -> str:
        """Like :attr:`descriptor_filepath` but returning a default value if None."""
        if self.descriptor_filepath is not None:
            return self.descriptor_filepath
        if not self.filepath:
            if self.innerpath:
                descriptor_filepath = replace_ext(self.innerpath, ".resource.json")
            else:
                descriptor_filepath = f"{self.resource_name}.resource.json"
        else:
            descriptor_filepath = replace_ext(self.filepath, ".resource.json")
        return descriptor_filepath

    def get_descriptor_path(
        self,
        fallback_to_default: bool = False,
    ) -> Optional[str]:
        """Returns the full path to the existing or future descriptor file."""
        try:
            return os.path.join(self.basepath, self.descriptor_filepath)
        except Exception:
            if fallback_to_default:
                return os.path.join(self.get_basepath(), self.get_descriptor_filepath())
            return

    def make_descriptor(self) -> dict:
        """Returns a frictionless descriptor for the resource."""
        return self.pickle_schema.dump(self)

    def _make_empty_fl_resource(self):
        """Create an empty frictionless resource object with a minimal descriptor."""
        return make_fl_resource()

    def set_basepath(self, basepath: str):
        if not self._basepath:
            self._basepath = Data.treat_new_basepath(
                basepath, self.filepath, other_logger=self.logger
            )
            self._resource.basepath = self.basepath
            return
        basepath_arg = resolve_path(basepath)
        if self.is_frozen:
            if basepath_arg == self.basepath:
                return
            raise ResourceIsFrozenError(self.name, self.basepath, basepath_arg)
        assert os.path.isdir(
            basepath_arg
        ), f"Basepath {basepath_arg!r} is not an existing directory."
        self._basepath = basepath_arg
        self._resource.basepath = basepath_arg
        self.logger.debug(f"Updated basepath to {self.basepath!r}")

    def store_descriptor(
        self, descriptor_path: Optional[str] = None, overwrite=True
    ) -> str:
        """Stores the frictionless descriptor to disk based on the resource's configuration and
        returns its path. Does not modify the resource's :attr:`status`.

        Returns:
            The path to the descriptor file on disk. If None, the default is used.

        Raises:
            InvalidResourcePathError: If the resource's path does not point to an existing file on disk.
        """
        if descriptor_path is None:
            descriptor_path = self.get_descriptor_path(fallback_to_default=True)
            if not self.descriptor_filepath:
                self.descriptor_filepath = self.get_descriptor_filepath()
        if not overwrite and os.path.isfile(descriptor_path):
            self.logger.info(
                f"Descriptor exists already and will not be overwritten: {descriptor_path}"
            )
            return descriptor_path
        descriptor_dict = self.make_descriptor()
        resource_filepath = descriptor_dict["path"]
        if resource_filepath:
            # check if storing the descriptor would result in a valid normpath
            if resource_basepath := descriptor_dict.get("basepath"):
                resulting_resource_path = os.path.join(
                    resource_basepath, resource_filepath
                )
                if not os.path.isfile(resulting_resource_path):
                    raise InvalidResourcePathError(resource_filepath, resource_basepath)
            else:
                descriptor_location = os.path.dirname(descriptor_path)
                resulting_resource_path = os.path.join(
                    descriptor_location, resource_filepath
                )
                if not os.path.isfile(resulting_resource_path):
                    raise InvalidResourcePathError(
                        resource_filepath, descriptor_location
                    )
        store_as_json_or_yaml(descriptor_dict, descriptor_path)
        return descriptor_path

    def _update_status(self) -> None:
        self._status = self._get_current_status()

    def validate(
        self,
        raise_exception: bool = False,
        only_if_necessary: bool = False,
    ) -> Optional[fl.Report]:
        """Validate the resource against its descriptor.

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
            return
        if only_if_necessary and self._is_valid is not None:
            return
        report = self._resource.validate()
        self._is_valid = True if report is None else report.valid
        if not report.valid:
            errors = [err.message for task in report.tasks for err in task.errors]
            if get_setting("never_store_unvalidated_data") and raise_exception:
                raise fl.FrictionlessException("\n".join(errors))
        return report

    # endregion Resource


class PathResource(Resource):
    """A resource that does not load frictionless descriptors or warns about them as :class:`Resource` would."""

    @classmethod
    def from_resource_path(
        cls,
        resource_path: str,
        resource_name: Optional[str] = None,
        descriptor_filepath: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ) -> Self:
        """Create a Resource from a file on disk, treating it just as a path even if it's a
        JSON/YAML resource descriptor"""
        with warnings.catch_warnings():
            # suppress warning when receiving descriptor path since this object is treating
            # all paths as equals
            warnings.simplefilter("ignore")
            return super().from_resource_path(
                resource_path=resource_path,
                resource_name=resource_name,
                descriptor_filepath=descriptor_filepath,
                basepath=basepath,
                **kwargs,
            )

    def __init__(
        self,
        resource: fl.Resource,
        descriptor_filepath: Optional[str] = None,
        basepath: Optional[str] = None,
        **kwargs,
    ):
        """

        Args:
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filepath:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filepath`. Needs to end either in resource.json or resource.yaml.
            basepath: Where the file would be serialized.
        """
        self.logger.debug(
            f"""
PathResource.__init__(
    resource={resource},
    descriptor_filepath={descriptor_filepath},
    basepath={basepath},
)"""
        )
        if resource is None:
            super().__init__(
                descriptor_filepath=descriptor_filepath, basepath=basepath, **kwargs
            )
            return
        if not isinstance(resource, fl.Resource):
            raise TypeError(
                f"resource must be of type frictionless.Resource, not {type(resource)}"
            )
        if not resource.path:
            raise ValueError(f"The resource comes without a path: {resource}")
        fl_resource = resource.to_copy()
        if basepath:
            fl_resource.basepath = basepath
        if not fl_resource.normpath:
            fl_resource.basepath = get_setting("default_basepath")
            raise ValueError(f"The resource did not yield a normpath: {fl_resource}.")
        if not os.path.isfile(fl_resource.normpath):
            raise FileNotFoundError(f"Resource does not exist: {fl_resource.normpath}")
        super().__init__(
            resource=fl_resource,
            descriptor_filepath=descriptor_filepath,
            basepath=basepath,
        )
        self.logger.debug(
            f"""
Resource(
    basepath={self.basepath},
    filepath={self.filepath},
    corpus_name={self.get_corpus_name()},
    resource_name={self.resource_name},
    descriptor_filepath={self.descriptor_filepath},
)"""
        )


ResourceSpecs: TypeAlias = Union[Resource, str, Path]


def resource_specs2resource(resource: ResourceSpecs) -> Resource:
    """Converts a resource specification to a resource.

    Args:
        resource: A resource specification.

    Returns:
        A resource.
    """
    if isinstance(resource, Resource):
        return resource
    if isinstance(resource, (str, Path)):
        return Resource.from_descriptor_path(resource)
    raise TypeError(
        f"Expected a Resource, str, or Path. Got {type(resource).__name__!r}."
    )

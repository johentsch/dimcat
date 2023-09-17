from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Iterable, List, MutableMapping, Optional, TypeAlias, Union

import frictionless as fl
import marshmallow as mm
import ms3
import pandas as pd
from dimcat.base import DimcatConfig, ObjectEnum, is_subclass_of
from dimcat.data.resources.base import D, Resource, SomeIndex
from dimcat.data.resources.dc import DimcatIndex, DimcatResource
from dimcat.dc_exceptions import ResourceNotProcessableError
from typing_extensions import Self

logger = logging.getLogger(__name__)


class FeatureName(ObjectEnum):
    Annotations = "Annotations"
    HarmonyLabels = "HarmonyLabels"
    KeyAnnotations = "KeyAnnotations"
    Measures = "Measures"
    Metadata = "Metadata"
    Notes = "Notes"


class Feature(DimcatResource):
    _enum_type = FeatureName


class ColumnFeature(Feature):
    @classmethod
    def from_descriptor(
        cls,
        descriptor: dict | fl.Resource,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
    ) -> Self:
        """Create a ColumnFeature by loading its frictionless descriptor from disk.
        The descriptor's directory is used as ``basepath``. ``descriptor_path`` is expected to end in
        ``.resource.json``.

        Args:
            descriptor: Descriptor corresponding to a frictionless resource descriptor.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where the file would be serialized.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
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
                e.g. replacing the :attr:`column_schema`.
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
        """Create a ColumnFeature from a dataframe, specifying its name and, optionally, at what path it is to be
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
                e.g. replacing the :attr:`column_schema`.
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
        new_object._update_status()
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
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            auto_validate:
                By default, the Resource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
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
        """Create a ColumnFeature from an existing :obj:`Resource`, specifying its name and,
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
                e.g. replacing the :attr:`column_schema`.
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


class Metadata(Feature):
    pass


# region Notes


class NotesFormat(str, Enum):
    NAME = "NAME"
    FIFTHS = "FIFTHS"
    MIDI = "MIDI"
    DEGREE = "DEGREE"
    INTERVAL = "INTERVAL"


class Notes(Feature):
    class Schema(Feature.Schema):
        format = mm.fields.Enum(NotesFormat)
        merge_ties = mm.fields.Boolean(
            load_default=True,
            metadata=dict(
                title="Merge tied notes",
                description="If set, notes that are tied together in the score are merged together, counting them "
                "as a single event of the corresponding length. Otherwise, every note head is counted.",
            ),
        )
        weight_grace_notes = mm.fields.Float(
            load_default=0.0,
            validate=mm.validate.Range(min=0.0, max=1.0),
            metadata=dict(
                title="Weight grace notes",
                description="Set a factor > 0.0 to multiply the nominal duration of grace notes which, otherwise, have "
                "duration 0 and are therefore excluded from many statistics.",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = NotesFormat.NAME,
        merge_ties: bool = True,
        weight_grace_notes: float = 0.0,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        self._format: NotesFormat = format
        self._weight_grace_notes: float = weight_grace_notes
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )

    @property
    def format(self) -> NotesFormat:
        return self._format

    @property
    def weight_grace_notes(self) -> float:
        return self._weight_grace_notes


# endregion Notes
# region Annotations


class Annotations(Feature):
    pass


class HarmonyLabels(Annotations):
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
                :func:`get_descriptor_filename`. Needs to on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where the file would be serialized.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby:
                Pass a list of column names or index levels to groupby something else than the default (by piece).
        """
        super().__init__(
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )


class KeyAnnotations(Annotations):
    pass


# endregion Annotations
# region helpers

FeatureSpecs: TypeAlias = Union[MutableMapping, Feature, FeatureName, str]


def feature_specs2config(feature: FeatureSpecs) -> DimcatConfig:
    """Converts a feature specification into a dimcat configuration.

    Raises:
        TypeError: If the feature cannot be converted to a dimcat configuration.
    """
    if isinstance(feature, DimcatConfig):
        feature_config = feature
    elif isinstance(feature, Feature):
        feature_config = feature.to_config()
    elif isinstance(feature, MutableMapping):
        feature_config = DimcatConfig(feature)
    elif isinstance(feature, str):
        feature_name = FeatureName(feature)
        feature_config = DimcatConfig(dtype=feature_name)
    else:
        raise TypeError(
            f"Cannot convert the {type(feature).__name__} {feature!r} to DimcatConfig."
        )
    if feature_config.options_dtype == "DimcatConfig":
        feature_config = DimcatConfig(feature_config["options"])
    if not is_subclass_of(feature_config.options_dtype, Feature):
        raise TypeError(
            f"DimcatConfig describes a {feature_config.options_dtype}, not a Feature: "
            f"{feature_config.options}"
        )
    return feature_config


def features_argument2config_list(
    features: Optional[FeatureSpecs | Iterable[FeatureSpecs]] = None,
    allowed_features: Optional[Iterable[str | FeatureName]] = None,
) -> List[DimcatConfig]:
    if features is None:
        return []
    if isinstance(features, (MutableMapping, Feature, FeatureName, str)):
        features = [features]
    configs = []
    for specs in features:
        configs.append(feature_specs2config(specs))
    if allowed_features:
        allowed_features = [FeatureName(f) for f in allowed_features]
        for config in configs:
            if config.options_dtype not in allowed_features:
                raise ResourceNotProcessableError(config.options_dtype)
    return configs


# endregion helpers

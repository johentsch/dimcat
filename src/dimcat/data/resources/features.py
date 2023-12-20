from __future__ import annotations

import itertools
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypeAlias,
)

import frictionless as fl
import marshmallow as mm
import ms3
import numpy as np
import numpy.typing as npt
import pandas as pd
from dimcat.base import FriendlyEnum, FriendlyEnumField
from dimcat.data.resources.base import D, FeatureName, S
from dimcat.data.resources.dc import (
    HARMONY_FEATURE_NAMES,
    DimcatIndex,
    Feature,
    Playthrough,
    SliceIntervals,
    UnitOfAnalysis,
)
from dimcat.data.resources.results import tuple2str
from dimcat.data.resources.utils import (
    boolean_is_minor_column_to_mode,
    condense_dataframe_by_groups,
    get_corpus_display_name,
    join_df_on_index,
    make_adjacency_groups,
    make_boolean_mask_from_set_of_tuples,
    merge_ties,
)
from dimcat.dc_exceptions import (
    DataframeIsMissingExpectedColumnsError,
    FeatureIsMissingFormatColumnError,
    ResourceIsMissingPieceIndexError,
)
from dimcat.utils import get_middle_composition_year

module_logger = logging.getLogger(__name__)
phraseComp: TypeAlias = Literal["ante", "body", "codetta", "post"]


class Metadata(Feature):
    _default_analyzer = dict(dtype="Proportions", dimension_column="length_qb")
    _default_value_column = "piece"

    def apply_slice_intervals(
        self,
        slice_intervals: SliceIntervals | pd.MultiIndex,
    ) -> pd.DataFrame:
        """"""
        if isinstance(slice_intervals, DimcatIndex):
            slice_intervals = slice_intervals.index
        if self.is_empty:
            self.logger.warning(f"Resource {self.name} is empty.")
            return pd.DataFrame(index=slice_intervals)
        return join_df_on_index(self.df, slice_intervals)

    def get_composition_years(
        self,
        group_cols: Optional[
            UnitOfAnalysis | str | Iterable[str]
        ] = UnitOfAnalysis.GROUP,
    ):
        group_cols = self._resolve_group_cols_arg(group_cols)
        years = get_middle_composition_year(metadata=self.df)
        if not group_cols:
            return years
        result = years.groupby(group_cols).mean()
        return result

    def get_corpus_names(
        self,
        func: Callable[[str], str] = get_corpus_display_name,
    ):
        """Returns the corpus names in chronological order, based on their pieces' mean composition years.
        If ``func`` is specify, the function will be applied to each corpus name. This is useful for prettifying
        the names, e.g. by removing underscores.
        """
        mean_composition_years = self.get_composition_years(group_cols="corpus")
        sorted_corpus_names = mean_composition_years.sort_values().index.to_list()
        if func is None:
            return sorted_corpus_names
        return [func(corp) for corp in sorted_corpus_names]


# region Annotations
AUXILIARY_HARMONYLABEL_COLUMNS = [
    "cadence",
    "label",
    "phraseend",
    "chord_tones",
    "chord_type",
    "figbass",
    "form",
    "numeral",
    "chord",
    "root",
]
"""These columns are included in sub-features of HarmonyLabels to enable more means of investigation,
such as groupers."""

KEY_CONVENIENCE_COLUMNS = [
    "globalkey_is_minor",
    "localkey_is_minor",
    "globalkey_mode",
    "localkey_mode",
    "localkey_resolved",
    "localkey_and_mode",
]
"""These columns are computed by default for all Annotations that include keys, where global keys are given as note
names, and local keys are given as Roman numerals. In both cases, lowercase strings are interpreted as minor keys."""


def extend_keys_feature(
    feature_df,
):
    columns_to_add = (
        "globalkey_mode",
        "localkey_mode",
        "localkey_resolved",
        "localkey_and_mode",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("localkey", "localkey_is_minor", "globalkey_is_minor")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [
        feature_df,
        boolean_is_minor_column_to_mode(feature_df.globalkey_is_minor).rename(
            "globalkey_mode"
        ),
        boolean_is_minor_column_to_mode(feature_df.localkey_is_minor).rename(
            "localkey_mode"
        ),
        ms3.transform(
            feature_df, ms3.resolve_relative_keys, ["localkey", "localkey_is_minor"]
        ).rename("localkey_resolved"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    concatenate_this = [
        feature_df,
        feature_df[["localkey", "globalkey_mode"]]
        .apply(safe_row_tuple, axis=1)
        .rename("localkey_and_mode"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class Annotations(Feature):
    pass


class DcmlAnnotations(Annotations):
    _auxiliary_column_names = [
        "color",
        "color_a",
        "color_b",
        "color_g",
        "color_r",
    ]
    _convenience_column_names = (
        [
            "added_tones",
            "bass_note",
            "cadence",
            "changes",
            "chord",
            "chord_tones",
            "chord_type",
            "figbass",
            "form",
            "globalkey",
            "localkey",
        ]
        + KEY_CONVENIENCE_COLUMNS
        + [
            "numeral",
            "pedal",
            "pedalend",
            "phraseend",
            "relativeroot",
            "root",
            "special",
        ]
    )
    _feature_column_names = ["label"]
    _default_value_column = "label"
    _extractable_features = HARMONY_FEATURE_NAMES + (
        FeatureName.CadenceLabels,
        FeatureName.PhraseLabels,
    )

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._apply_playthrough(feature_df)
        feature_df = extend_keys_feature(feature_df)
        return self._sort_columns(feature_df)


def make_chord_col(df: D, cols: Optional[List[str]] = None, name: str = "chord"):
    """The 'chord' column contains the chord part of a DCML label, i.e. without indications of key, pedal, cadence, or
    phrase. This function can re-create this column, e.g. if the feature columns were changed. To that aim, the function
    takes a DataFrame and the column names that it adds together, creating new strings.
    """
    if cols is None:
        cols = ["numeral", "form", "figbass", "changes", "relativeroot"]
    cols = [c for c in cols if c in df.columns]
    summing_cols = [c for c in cols if c not in ("changes", "relativeroot")]
    if len(summing_cols) == 1:
        chord_col = df[summing_cols[0]].fillna("").astype("string")
    else:
        chord_col = df[summing_cols].fillna("").astype("string").sum(axis=1)
    if "changes" in cols:
        chord_col += ("(" + df.changes.astype("string") + ")").fillna("")
    if "relativeroot" in cols:
        chord_col += ("/" + df.relativeroot.astype("string")).fillna("")
    return chord_col.rename(name)


def extend_harmony_feature(
    feature_df,
):
    """Requires previous application of :func:`transform_keys_feature`."""
    columns_to_add = (
        "root_roman",
        "pedal_resolved",
        "chord_and_mode",
        "chord_reduced",
        "chord_reduced_and_mode",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = (
        "chord",
        "form",
        "figbass",
        "pedal",
        "numeral",
        "relativeroot",
        "localkey_is_minor",
        "localkey_mode",
    )
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [feature_df]
    if "root_roman" not in feature_df.columns:
        concatenate_this.append(
            (feature_df.numeral + ("/" + feature_df.relativeroot).fillna("")).rename(
                "root_roman"
            )
        )
    if "chord_reduced" not in feature_df.columns:
        concatenate_this.append(
            (
                reduced_col := make_chord_col(
                    feature_df,
                    cols=["numeral", "form", "figbass", "relativeroot"],
                    name="chord_reduced",
                )
            )
        )
    else:
        reduced_col = feature_df.chord_reduced
    if "chord_reduced_and_mode" not in feature_df.columns:
        concatenate_this.append(
            (reduced_col + ", " + feature_df.localkey_mode).rename(
                "chord_reduced_and_mode"
            )
        )
    if "pedal_resolved" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df, ms3.resolve_relative_keys, ["pedal", "localkey_is_minor"]
            ).rename("pedal_resolved")
        )
    if "chord_and_mode" not in feature_df.columns:
        concatenate_this.append(
            feature_df[["chord", "localkey_mode"]]
            .apply(safe_row_tuple, axis=1)
            .rename("chord_and_mode")
        )
    # if "root_roman_resolved" not in feature_df.columns:
    #     concatenate_this.append(
    #         ms3.transform(
    #             feature_df,
    #             ms3.rel2abs_key,
    #             ["numeral", "localkey_resolved", "localkey_resolved_is_minor"],
    #         ).rename("root_roman_resolved")
    #     )
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


def chord_tones2interval_structure(
    fifths: Iterable[int], reference: Optional[int] = None
) -> Tuple[str]:
    """The fifth are interpreted as intervals expressing distances from the local tonic ("neutral degrees").
    The result will be a tuple of strings that express the same intervals but expressed with respect to the given
    reference (neutral degree), removing unisons.
    If no reference is specified, the first degree (usually, the bass note) is used as such.
    """
    try:
        fifths = tuple(fifths)
        if len(fifths) == 0:
            return ()
    except Exception:
        return ()
    if reference is None:
        reference = fifths[0]
    elif reference in fifths:
        position = fifths.index(reference)
        if position > 0:
            fifths = fifths[position:] + fifths[:position]
    adapted_intervals = [
        ms3.fifths2iv(adapted)
        for interval in fifths
        if (adapted := interval - reference) != 0
    ]
    return tuple(adapted_intervals)


def add_chord_tone_scale_degrees(
    feature_df,
):
    """Turns 'chord_tones' column into multiple scale-degree columns."""
    columns_to_add = (
        "scale_degrees",
        "scale_degrees_and_mode" "scale_degrees_major",
        "scale_degrees_minor",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("chord_tones", "localkey_is_minor", "localkey_mode")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [feature_df]
    if "scale_degrees" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df, ms3.fifths2sd, ["chord_tones", "localkey_is_minor"]
            ).rename("scale_degrees")
        )
    if "scale_degrees_major" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.chord_tones, ms3.fifths2sd, minor=False).rename(
                "scale_degrees_major"
            )
        )
    if "scale_degrees_minor" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.chord_tones, ms3.fifths2sd, minor=True).rename(
                "scale_degrees_minor"
            )
        )
    feature_df = pd.concat(concatenate_this, axis=1)
    if "scale_degrees_and_mode" not in feature_df.columns:
        sd_and_mode = pd.Series(
            feature_df[["scale_degrees", "localkey_mode"]].itertuples(
                index=False, name=None
            ),
            index=feature_df.index,
            name="scale_degrees_and_mode",
        )
        concatenate_this = [feature_df, sd_and_mode.apply(tuple2str)]
        feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


def add_chord_tone_intervals(
    feature_df,
):
    """Turns 'chord_tones' column into one or two additional columns, depending on whether a 'root' column is
    present, where the chord_tones (which come as fifths) are represented as strings representing intervals over the
    bass_note and above the root, if present.
    """
    columns_to_add = (
        "intervals_over_bass",
        "intervals_over_root",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("chord_tones",)  # "root" is optional
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [feature_df]
    if "intervals_over_bass" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df.chord_tones, chord_tones2interval_structure
            ).rename("intervals_over_bass")
        )
    if "intervals_over_root" not in feature_df.columns and "root" in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df, chord_tones2interval_structure, ["chord_tones", "root"]
            ).rename("intervals_over_root")
        )
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class HarmonyLabelsFormat(FriendlyEnum):
    """Format to display the chord labels in. ROMAN stands for Roman numerals, ROMAN_REDUCED for the same numerals
    without any suspensions, alterations, additions, etc."""

    ROMAN = "ROMAN"
    ROMAN_REDUCED = "ROMAN_REDUCED"
    SCALE_DEGREE = "SCALE_DEGREE"
    SCALE_DEGREE_MAJOR = "SCALE_DEGREE_MAJOR"
    SCALE_DEGREE_MINOR = "SCALE_DEGREE_MINOR"


class HarmonyLabels(DcmlAnnotations):
    _auxiliary_column_names = DcmlAnnotations._auxiliary_column_names + [
        "cadence",
        "label",
        "phraseend",
    ]
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS + [
        "added_tones",
        "bass_note",
        "changes",
        "chord_and_mode",
        "chord_reduced",
        "chord_reduced_and_mode",
        "chord_tones",
        "scale_degrees",
        "scale_degrees_and_mode",
        "scale_degrees_major",
        "scale_degrees_minor",
        "intervals_over_bass",
        "intervals_over_root",
        "chord_type",
        "figbass",
        "form",
        "numeral",
        "pedal",
        "pedalend",
        "relativeroot",
        "root",
        "special",
    ]
    _feature_column_names = [
        "globalkey",
        "localkey",
        "chord",
    ]
    _default_value_column = "chord_and_mode"

    class Schema(DcmlAnnotations.Schema):
        format = FriendlyEnumField(
            HarmonyLabelsFormat,
            load_default=HarmonyLabelsFormat.ROMAN,
            metadata=dict(
                expose=True,
                description="Format to display the chord labels in. ROMAN stands for Roman numerals, ROMAN_REDUCED "
                "for the same numerals without any suspensions, alterations, additions, etc.",
            ),
        )

    def __init__(
        self,
        format: HarmonyLabelsFormat = HarmonyLabelsFormat.ROMAN,
        resource: fl.Resource = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = False,
        default_groupby: Optional[str | list[str]] = None,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        """

        Args:
            format:
                Format to display the chord labels in. ROMAN stands for Roman numerals,
                ROMAN_REDUCED for the same numerals without any suspensions, alterations, additions,
                etc.
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where to store serialization data and its descriptor by default.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby: Name of the fields for grouping this resource (usually after a Grouper has been applied).
            playthrough:
                Defaults to ``Playthrough.SINGLE``, meaning that first-ending (prima volta) bars are dropped in order
                to exclude incorrect transitions and adjacencies between the first- and second-ending bars.
        """
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            playthrough=playthrough,
        )

    @property
    def format(self) -> HarmonyLabelsFormat:
        return self._format

    @format.setter
    def format(self, format: HarmonyLabelsFormat):
        format = HarmonyLabelsFormat(format)
        if self.format == format:
            return
        if format == HarmonyLabelsFormat.ROMAN:
            new_formatted_column = "chord_and_mode"
        elif format == HarmonyLabelsFormat.ROMAN_REDUCED:
            new_formatted_column = "chord_reduced_and_mode"
        elif format == HarmonyLabelsFormat.SCALE_DEGREE:
            new_formatted_column = "scale_degrees_and_mode"
        elif format == HarmonyLabelsFormat.SCALE_DEGREE_MAJOR:
            new_formatted_column = "scale_degrees_major"
        elif format == HarmonyLabelsFormat.SCALE_DEGREE_MINOR:
            new_formatted_column = "scale_degrees_minor"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    @property
    def formatted_column(self) -> str:
        if self.format == HarmonyLabelsFormat.ROMAN:
            if "mode" in self.default_groupby:
                return "chord"
            else:
                return "chord_and_mode"
        elif self._format == HarmonyLabelsFormat.ROMAN_REDUCED:
            if "mode" in self.default_groupby:
                return "chord_reduced"
            else:
                return "chord_reduced_and_mode"
        elif self._format == HarmonyLabelsFormat.SCALE_DEGREE:
            if "mode" in self.default_groupby:
                return "scale_degrees"
            else:
                return "scale_degrees_and_mode"
        if self._formatted_column is not None:
            return self._formatted_column
        if self._default_formatted_column is not None:
            return self._default_formatted_column
        return

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._apply_playthrough(feature_df)
        feature_df = self._drop_rows_with_missing_values(
            feature_df, column_names=self._feature_column_names
        )
        feature_df = extend_keys_feature(feature_df)
        feature_df = extend_harmony_feature(feature_df)
        feature_df = add_chord_tone_intervals(feature_df)
        feature_df = add_chord_tone_scale_degrees(feature_df)
        return self._sort_columns(feature_df)


def safe_row_tuple(row):
    try:
        return ", ".join(row)
    except TypeError:
        return pd.NA


def extend_bass_notes_feature(
    feature_df,
):
    """Requires previous application of :func:`transform_keys_feature`."""
    columns_to_add = (
        "bass_note_over_local_tonic",
        "bass_degree",
        "bass_degree_and_mode",
        "bass_degree_major",
        "bass_degree_minor",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    expected_columns = ("bass_note", "localkey_is_minor", "localkey_mode")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    concatenate_this = [feature_df]
    if "bass_note_over_local_tonic" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.bass_note, ms3.fifths2iv).rename(
                "bass_note_over_local_tonic"
            )
        )
    if "bass_degree" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(
                feature_df, ms3.fifths2sd, ["bass_note", "localkey_is_minor"]
            ).rename("bass_degree")
        )
    if "bass_degree_major" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.bass_note, ms3.fifths2sd, minor=False).rename(
                "bass_degree_major"
            )
        )
    if "bass_degree_minor" not in feature_df.columns:
        concatenate_this.append(
            ms3.transform(feature_df.bass_note, ms3.fifths2sd, minor=True).rename(
                "bass_degree_minor"
            )
        )
    feature_df = pd.concat(concatenate_this, axis=1)
    if "bass_degree_and_mode" not in feature_df.columns:
        concatenate_this = [
            feature_df,
            feature_df[["bass_degree", "localkey_mode"]]
            .apply(safe_row_tuple, axis=1)
            .rename("bass_degree_and_mode"),
        ]
        feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class BassNotesFormat(FriendlyEnum):
    """Format to display the bass notes in. INTERVAL stands for the interval between the bass note and the local
    tonic, FIFTHS expresses that same interval as a number of fifths, SCALE_DEGREE expresses the bass note as a scale
    degree depending on the local key (i.e. scale degrees 3, 6, 7 are minor intervals in minor and major intervals in
    major), whereas SCALE_DEGREE_MAJOR and SCALE_DEGREE_MINOR express the bass note as a scale degree independent of
    the local key"""

    FIFTHS = "FIFTHS"
    INTERVAL = "INTERVAL"
    SCALE_DEGREE = "SCALE_DEGREE"
    SCALE_DEGREE_MAJOR = "SCALE_DEGREE_MAJOR"
    SCALE_DEGREE_MINOR = "SCALE_DEGREE_MINOR"


class BassNotes(HarmonyLabels):
    _default_formatted_column = "bass_note_over_local_tonic"
    _default_value_column = "bass_note"
    _auxiliary_column_names = (
        DcmlAnnotations._auxiliary_column_names + AUXILIARY_HARMONYLABEL_COLUMNS
    )
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS + [
        "bass_degree",
        "bass_degree_and_mode",
        "bass_degree_major",
        "bass_degree_minor",
        "bass_note_over_local_tonic",
        "intervals_over_bass",
        "intervals_over_root",
        "scale_degrees",
        "scale_degrees_and_mode",
        "scale_degrees_major",
        "scale_degrees_minor",
    ]
    _feature_column_names = [
        "globalkey",
        "localkey",
        "bass_note",
    ]
    _extractable_features = None

    class Schema(DcmlAnnotations.Schema):
        format = FriendlyEnumField(
            BassNotesFormat,
            load_default=BassNotesFormat.INTERVAL,
            metadata=dict(
                expose=True,
                description="Format to display the bass notes in. INTERVAL stands for the interval between the bass "
                "note and the local tonic, FIFTHS expresses that same interval as a number of fifths, "
                "SCALE_DEGREE expresses the bass note as a scale degree depending on the local key (i.e. "
                "scale degrees 3, 6, 7 are minor intervals in minor and major intervals in major), "
                "whereas SCALE_DEGREE_MAJOR and SCALE_DEGREE_MINOR express the bass note as a scale "
                "degree independent of the local key",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = BassNotesFormat.INTERVAL,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
    ) -> None:
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
        )

    @property
    def format(self) -> BassNotesFormat:
        return self._format

    @format.setter
    def format(self, format: BassNotesFormat):
        format = BassNotesFormat(format)
        if self.format == format:
            return
        if format == BassNotesFormat.INTERVAL:
            new_formatted_column = "bass_note_over_local_tonic"
        elif format == BassNotesFormat.FIFTHS:
            new_formatted_column = "bass_note"
        elif format == BassNotesFormat.SCALE_DEGREE:
            new_formatted_column = "bass_degree_and_mode"
        elif format == BassNotesFormat.SCALE_DEGREE_MAJOR:
            new_formatted_column = "bass_degree_major"
        elif format == BassNotesFormat.SCALE_DEGREE_MINOR:
            new_formatted_column = "bass_degree_minor"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    @property
    def formatted_column(self) -> str:
        if self.format == BassNotesFormat.SCALE_DEGREE:
            if "mode" in self.default_groupby:
                return "bass_degree"
            else:
                return "bass_degree_and_mode"
        if self._formatted_column is not None:
            return self._formatted_column
        if self._default_formatted_column is not None:
            return self._default_formatted_column
        return

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._apply_playthrough(feature_df)
        feature_df = self._drop_rows_with_missing_values(
            feature_df, column_names=self._feature_column_names
        )
        feature_df = extend_keys_feature(feature_df)
        feature_df = extend_bass_notes_feature(feature_df)
        feature_df = add_chord_tone_intervals(feature_df)
        feature_df = add_chord_tone_scale_degrees(feature_df)
        return self._sort_columns(feature_df)


def extend_cadence_feature(
    feature_df,
):
    columns_to_add = (
        "cadence_type",
        "cadence_subtype",
    )
    if all(col in feature_df.columns for col in columns_to_add):
        return feature_df
    if "cadence" not in feature_df.columns:
        raise DataframeIsMissingExpectedColumnsError(
            "cadence",
            feature_df.columns.to_list(),
        )
    split_labels = feature_df.cadence.str.split(".", expand=True).rename(
        columns={0: "cadence_type", 1: "cadence_subtype"}
    )
    feature_df = pd.concat([feature_df, split_labels], axis=1)
    return feature_df


class CadenceLabelFormat(FriendlyEnum):
    """Format to display the cadence labels in. RAW stands for 'as-is'. TYPE omits the subtype, reducing more
    specific labels, whereas SUBTYPE displays subtypes only, omitting all labels that do not specify one.
    """

    RAW = "RAW"
    TYPE = "TYPE"
    SUBTYPE = "SUBTYPE"


class CadenceLabels(DcmlAnnotations):
    _auxiliary_column_names = ["label", "chord"]
    _convenience_column_names = (
        ["globalkey", "localkey"]
        + KEY_CONVENIENCE_COLUMNS
        + [
            "cadence_type",
            "cadence_subtype",
        ]
    )
    _feature_column_names = ["cadence"]
    _default_value_column = "cadence"
    _default_analyzer = "CadenceCounter"
    _extractable_features = None

    class Schema(DcmlAnnotations.Schema):
        format = FriendlyEnumField(
            CadenceLabelFormat,
            load_default=CadenceLabelFormat.RAW,
            metadata=dict(
                expose=True,
                description="Format to display the cadence labels in. RAW stands for 'as-is'. TYPE omits the subtype, "
                "reducing more specific labels, whereas SUBTYPE displays subtypes only, omitting all "
                "labels that do not specify one.",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = CadenceLabelFormat.RAW,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        """

        Args:
            format:
                Format to display the cadence labels in. RAW stands for 'as-is'. TYPE omits the
                subtype, reducing more specific labels, whereas SUBTYPE displays subtypes only,
                omitting all labels that do not specify one.
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where to store serialization data and its descriptor by default.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby: Name of the fields for grouping this resource (usually after a Grouper has been applied).
            playthrough:
                Defaults to ``Playthrough.SINGLE``, meaning that first-ending (prima volta) bars are dropped in order
                to exclude incorrect transitions and adjacencies between the first- and second-ending bars.
        """
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            playthrough=playthrough,
        )

    @property
    def format(self) -> CadenceLabelFormat:
        return self._format

    @format.setter
    def format(self, format: CadenceLabelFormat):
        format = CadenceLabelFormat(format)
        if self.format == format:
            return
        if format == CadenceLabelFormat.RAW:
            new_formatted_column = "cadence"
        elif format == CadenceLabelFormat.TYPE:
            new_formatted_column = "cadence_type"
        elif format == CadenceLabelFormat.SUBTYPE:
            new_formatted_column = "cadence_subtype"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._apply_playthrough(feature_df)
        try:
            feature_df = extend_keys_feature(feature_df)
        except DataframeIsMissingExpectedColumnsError:
            pass
        feature_df = self._drop_rows_with_missing_values(
            feature_df, column_names=self._feature_column_names
        )
        feature_df = extend_cadence_feature(feature_df)
        return self._sort_columns(feature_df)


class KeyAnnotations(DcmlAnnotations):
    _auxiliary_column_names = ["label"]
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS
    _feature_column_names = ["globalkey", "localkey"]
    _extractable_features = None
    _default_value_column = "localkey_and_mode"

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._apply_playthrough(feature_df)
        feature_df = extend_keys_feature(feature_df)
        groupby_levels = feature_df.index.names[:-1]
        group_keys, _ = make_adjacency_groups(
            feature_df.localkey, groupby=groupby_levels
        )
        feature_df = condense_dataframe_by_groups(
            feature_df, group_keys, logger=self.logger
        )
        return self._sort_columns(feature_df)


def _get_index_intervals_for_phrases(
    markers: S,
    n_ante: int = 0,
    n_post: int = 0,
    logger: Optional[logging.Logger] = None,
) -> List[Tuple[int, int, Optional[int], int, int]]:
    """Expects a Series with a RangeIndex and computes (from, to) index position intervals based on the presence of
    either the start_symbol or the end_symbol. If both are found, an error is thrown. If None is found, the result is
    an empty list.

    The function operates based on the constants

        start_symbol ``"{"``
            If this symbol is present in any of the series' strings, intervals will be formed starting from one to the
            next occurrences (within strings). The interval for the last symbol reaches until the end of the series
            (that is, the last index position + 1).

        end_symbol ``"\\"``
            If this symbol is present in any of the series' strings, intervals will be formed starting from the first
            index position to the position of the first end_symbol + 1, and from there until one after the next, and
            so on.

    Args:
        markers:
            A Series containing either start or end symbols of phrases. Expected to have a RangeIndex. When the series
            corresponds to a chunk of a larger one, the RangeIndex should correspond to the respective positions in
            the original series.
        n_ante: Pass a positive integer to have the intervals include n earlier positions.
        n_post:
            Pass a positive integer > 0 to have the intervals include n subsequent positions. The minimum is 1 because
            for new-style phrase endings (``}``) the end_symbol may actually appear only with the beginning of the
            subsequent phrase in the case of ``}{``.
        logger:

    Returns:
        A list of (first_i, start_i, end_i, subsequent_i, stop_i) index positions that can be used for slicing rows
        of the dataframe from which the series was taken. The meaning of the included slice intervals is as follows:

        * ``[start_i:start_i)``: The n_ante positions before the phrase.
        * ``[start_i:end_i]``: The body of the phrase, including end symbol.
        * ``[end_i:subsequent_i)``:
          The codetta, i.e., the part between the end_symbol and the subsequent phrase. In the case of phrase overlap,
          the two are identical and the codetta is empty.
        * ``[subsequent_i:stop_i)``: The n_post positions after the phrase.
    """
    if logger is None:
        logger = module_logger
    present_symbols = markers.unique()
    start_symbol, end_symbol = "{", r"\\"
    has_start = start_symbol in present_symbols
    has_end = end_symbol in present_symbols
    if not (has_start or has_end):
        return []
    if has_start and has_end:
        logger.warning(
            f"Currently I can create phrases either based on end symbols or on start symbols, but this df has both:"
            f":\n{markers.value_counts().to_dict()}\nUsing {start_symbol}, ignoring {end_symbol}..."
        )
    ix_min = markers.index.min()
    ix_max = markers.index.max() + 1
    if has_start:
        end_symbol = "}"
        start_symbol_mask = markers.str.contains(start_symbol).fillna(False)
        starts_ix = start_symbol_mask.index[start_symbol_mask].to_list()
        end_symbol_mask = markers.str.contains(end_symbol).fillna(False)
        ends_ix = end_symbol_mask.index[end_symbol_mask].to_list()

        def include_end_ix(fro, to):
            potential = range(fro + 1, to + 1)
            included_ends = [ix for ix in ends_ix if ix in potential]
            n_ends = len(included_ends)
            if not n_ends:
                inspect_series = markers.loc[fro + 1 : to + 1].dropna()
                logger.warning(
                    f"Phrase [{fro}:{to}] was expected to have an end symbol within [{fro+1}:{to+1}]:\n{inspect_series}"
                )
                return (fro, None, to)
            elif n_ends > 2:
                inspect_series = markers.loc[fro + 1 : to + 1].dropna()
                logger.warning(
                    f"Phrase [{fro}:{to}] has multiple end symbols within [{fro+1}:{to+1}]:\n{inspect_series}"
                )
                return (fro, None, to)
            end_ix = included_ends[0]
            return (fro, end_ix, to)

        start_end_subsequent = [
            include_end_ix(fro, to)
            for fro, to in zip(starts_ix, starts_ix[1:] + [ix_max])
        ]
    else:
        end_symbol_mask = markers.str.contains(end_symbol).fillna(False)
        subsequent_ix = (end_symbol_mask.index[end_symbol_mask] + 1).to_list()
        start_end_subsequent = [
            (fro, to - 1, to)
            for fro, to in zip([ix_min] + subsequent_ix[:-1], subsequent_ix)
        ]
    result = []
    for start_i, end_i, subsequent_i in start_end_subsequent:
        first_i = start_i
        if n_ante:
            new_first_i = start_i - n_ante
            if new_first_i >= ix_min:
                first_i = new_first_i
            else:
                first_i = ix_min
        stop_i = subsequent_i
        if n_post:
            new_stop_i = subsequent_i + n_post
            if new_stop_i <= ix_max:
                stop_i = new_stop_i
            else:
                stop_i = ix_max
        result.append((first_i, start_i, end_i, subsequent_i, stop_i))
    return result


def get_index_intervals_for_phrases(
    harmony_labels: D,
    group_cols: List[str],
    n_ante: int = 0,
    n_post: int = 0,
    logger: Optional[logging.Logger] = None,
) -> Dict[Any, List[Tuple[int, int]]]:
    """Returns a list of slice intervals for selecting the rows belonging to a phrase."""
    if logger is None:
        logger = module_logger
    phraseends_reset = harmony_labels.reset_index()
    group_intervals = {}
    groupby = phraseends_reset.groupby(group_cols)
    for group, markers in groupby.phraseend:
        first_start_end_sbsq_last = _get_index_intervals_for_phrases(
            markers, n_ante=n_ante, n_post=n_post, logger=logger
        )
        group_intervals[group] = first_start_end_sbsq_last
    return group_intervals


def make_sequence_non_repeating(
    sequence: S,
) -> tuple:
    """Returns values in the given sequence without immediate repetitions. Fails if the sequence contains NA."""
    return tuple(val for val, _ in itertools.groupby(sequence))


def _condense_component(
    component_df: D,
    qstamp_col_position: int,
    duration_col_position: int,
    localkey_col_position: int,
    label_col_position: int,
    chord_col_position: int,
) -> S:
    """Returns a series which condenses the phrase components into a row."""
    first_row = component_df.iloc[0]
    component_info = _compile_component_info(
        component_df,
        qstamp_col_position,
        duration_col_position,
        localkey_col_position,
        label_col_position,
        chord_col_position,
    )
    row_values = first_row.to_dict()
    row_values.update(component_info)
    return pd.Series(row_values, name=first_row.name)


def _make_groupwise_range_index(idx: pd.Index) -> npt.NDArray:
    """Turns adjacency groups into integer ranges starting from 0.

    The algorithm builds on Warren Weckesser's approach via https://stackoverflow.com/a/20033438
    """
    arr = idx.to_numpy()
    start_mask = arr != np.roll(arr, 1)
    (start_index,) = np.where(start_mask)
    group_lengths = start_index[1:] - start_index[:-1]
    incr = np.asarray(~start_mask, int)
    incr[start_index[1:]] = 1 - group_lengths
    incr.cumsum(out=incr)
    return incr


def make_multiindex_for_unstack(
    idx: pd.Index, new_level_name: str = "i"
) -> pd.MultiIndex:
    """Turns an index that contains adjacency groups (adjacent entries having the same value) into
    a 2-level MultiIndex where the new level represents an individual integer range for each group,
    starting at 0.
    """
    old_level_name = idx.name
    groupwise_ranges = _make_groupwise_range_index(idx)
    result = pd.MultiIndex.from_arrays(
        [idx, groupwise_ranges], names=[old_level_name, new_level_name]
    )
    return result


def _transform_phrase_data(
    phrase_df,
    columns: str | List[str] = "chord",
    components: phraseComp | List[phraseComp] = "body",
    droplevels: bool | Hashable | Sequence[Hashable] = False,
    reverse: bool = False,
    new_level_name: str = "i",
    wide_format: bool = False,
):
    """Returns a dataframe containing the requested phrase components and harmony columns.

    Args:
        phrase_df: PhraseAnnotations dataframe.
        columns:
            Column(s) to include in the result. Passing a list results in an additional index level
            called "column".
        components:
            Which of the four phrase components to include, ∈ {'ante', 'body', 'codetta', 'post'}.
        droplevels:
            Can be a boolean or any level specifier accepted by :meth:`pandas.MultiIndex.droplevel()`.
            If False (default), all levels are retained. If True, only the phrase_id level and
            the ``new_level_name`` are retained. In all other cases, the indicated (string or
            integer) value(s) must be valid and cause one of the index (or column if
            ``wide_format``) levelse to be dropped. ``new_level_name`` cannot be dropped. If
            ``wide_format`` is True, dropping 'phrase_id' will likely lead to an exception
            regarding duplicate index values.
        reverse:
            Pass True to reverse the order of harmonies so that each phrase's last label comes
            first.
        new_level_name:
            Defaults to 'i', which is the the name of the original level that will be replaced
            by this new one. The new one represents the individual integer range for each
            phrase, starting at 0.
        wide_format:
            Pass True to pivot the result so that each row (group) exhibits the ``columns`` values
            of one phrase, one per column. The columns are according to ``new_level_name``.

    Returns:
        Dataframe representing partial information on the selected phrases in long or wide format.
    """
    result = phrase_df.loc[pd.IndexSlice[:, :, :, components], columns]
    include_column_level = not isinstance(columns, str)
    if reverse:
        result = result[::-1]
    phrase_ids = result.index.get_level_values("phrase_id")
    if droplevels is True:
        new_index = make_multiindex_for_unstack(
            phrase_ids, new_level_name=new_level_name
        )
    else:
        old_index = result.index.droplevel(-1)
        if not droplevels:
            pass
        else:
            old_index = old_index.droplevel(droplevels)
        new_level = pd.Series(
            _make_groupwise_range_index(phrase_ids), name=new_level_name
        )
        new_index_df = pd.concat([old_index.to_frame(index=False), new_level], axis=1)
        new_index = pd.MultiIndex.from_frame(new_index_df)
    result.index = new_index
    if wide_format:
        result = result.unstack(level=new_level_name)
        if include_column_level:
            result.columns.rename("column", level=0, inplace=True)
            result = result.stack("column")
        result = result.sort_index(axis=1)
    return result


def _compile_component_info(
    component_df: D,
    qstamp_col_position,
    duration_col_position,
    localkey_col_position,
    label_col_position,
    chord_col_position,
    key_prefix: Optional[str] = "",
):
    start_qstamp = component_df.iat[0, qstamp_col_position]
    end_qstamp = (
        component_df.iat[-1, qstamp_col_position]
        + component_df.iat[-1, duration_col_position]
    )
    new_duration = float(end_qstamp - start_qstamp)
    columns = component_df.iloc(axis=1)
    localkeys = tuple(columns[localkey_col_position])
    modulations = make_sequence_non_repeating(localkeys)
    labels = tuple(columns[label_col_position])
    chords = tuple(columns[chord_col_position])
    component_info = dict(
        localkeys=localkeys,
        n_modulations=len(modulations) - 1,
        modulatory_sequence=modulations,
        n_labels=len(labels),
        labels=labels,
        n_chords=len(chords),
        chords=chords,
    )
    duration_key = "duration_qb"
    if key_prefix:
        component_info = {
            f"{key_prefix}{key}": val for key, val in component_info.items()
        }
        if key_prefix != "phrase_":
            # phrase duration is used as the main 'duration_qb' column
            duration_key = f"{key_prefix}duration_qb"
    component_info[duration_key] = new_duration
    return component_info


def condense_components(raw_phrase_df: D) -> D:
    qstamp_col_position = raw_phrase_df.columns.get_loc("quarterbeats")
    duration_col_position = raw_phrase_df.columns.get_loc("duration_qb")
    localkey_col_position = raw_phrase_df.columns.get_loc("localkey")
    label_col_position = raw_phrase_df.columns.get_loc("label")
    chord_col_position = raw_phrase_df.columns.get_loc("chord")
    groupby_levels = raw_phrase_df.index.names[:-1]
    return raw_phrase_df.groupby(groupby_levels).apply(
        _condense_component,
        qstamp_col_position,
        duration_col_position,
        localkey_col_position,
        label_col_position,
        chord_col_position,
    )


def _condense_phrase(
    phrase_df: D,
    qstamp_col_position: int,
    duration_col_position: int,
    localkey_col_position: int,
    label_col_position: int,
    chord_col_position: int,
) -> dict:
    """Returns a series which condenses the phrase into a row."""
    component_indices = phrase_df.groupby("phrase_component").indices
    body_idx = component_indices.get("body")
    codetta_idx = component_indices.get("codetta")
    first_phrase_i = body_idx[0]
    last_body_i = body_idx[-1]
    end_label = phrase_df.iat[last_body_i, label_col_position]
    end_chord = phrase_df.iat[last_body_i, chord_col_position]
    if "}" in end_label:
        interlocked_ante = "}" in phrase_df.iat[first_phrase_i, label_col_position]
        interlocked_post = codetta_idx is None
    else:
        # old-style phrase endings didn't provide the means to encode phrase interlocking
        interlocked_ante, interlocked_post = pd.NA, pd.NA
    if codetta_idx is None:
        # if no codetta is defined, the phrase info will simply be copied from the body component
        component_index_iterable = component_indices.items()
    else:
        phrase_idx = np.concatenate([body_idx[:-1], codetta_idx])
        component_index_iterable = [("phrase", phrase_idx), *component_indices.items()]
    first_body_row = phrase_df.iloc[first_phrase_i]
    row_values = first_body_row.to_dict()
    for group, component_df in (
        (group, phrase_df.take(idx)) for group, idx in component_index_iterable
    ):
        component_info = _compile_component_info(
            component_df,
            qstamp_col_position,
            duration_col_position,
            localkey_col_position,
            label_col_position,
            chord_col_position,
            key_prefix=f"{group}_",
        )
        row_values.update(component_info)
    if codetta_idx is None:
        phrase_info = {}
        for key, value in component_info.items():
            if key.startswith("body_"):
                if key == "body_duration_qb":
                    phrase_key = "duration_qb"
                else:
                    phrase_key = key.replace("body_", "phrase_")
                phrase_info[phrase_key] = value
        row_values.update(phrase_info)
    row_values["interlocked_ante"] = interlocked_ante
    row_values["interlocked_post"] = interlocked_post
    row_values["end_label"] = end_label
    row_values["end_chord"] = end_chord
    return row_values


def condense_phrases(raw_phrase_df: D) -> D:
    qstamp_col_position = raw_phrase_df.columns.get_loc("quarterbeats")
    duration_col_position = raw_phrase_df.columns.get_loc("duration_qb")
    localkey_col_position = raw_phrase_df.columns.get_loc("localkey")
    label_col_position = raw_phrase_df.columns.get_loc("label")
    chord_col_position = raw_phrase_df.columns.get_loc("chord")
    # we're not using :meth:`pandas.DataFrameGroupBy.apply` because the series returned by _condense_phrases may have
    # varying lengths, which would result in a series, not a dataframe. Instead, we're collecting groupwise row dicts
    # and then creating a dataframe from them.
    groupby_levels = raw_phrase_df.index.names[:-2]
    group2dict = {
        group: _condense_phrase(
            phrase_df,
            qstamp_col_position,
            duration_col_position,
            localkey_col_position,
            label_col_position,
            chord_col_position,
        )
        for group, phrase_df in raw_phrase_df.groupby(groupby_levels)
    }
    result = pd.DataFrame.from_dict(group2dict, orient="index")
    result.index.names = groupby_levels
    nullable_int_cols = {
        col_name: "Int64"
        for comp, col in itertools.product(
            ("phrase_", "ante_", "body_", "codetta_", "post_"),
            ("n_modulations", "n_labels", "n_chords"),
        )
        if (col_name := comp + col) in result.columns
    }
    result = result.astype(nullable_int_cols)
    return result


def _make_concatenated_ranges(
    starts: npt.NDArray[np.int64],
    stops: npt.NDArray[np.int64],
    counts: npt.NDArray[np.int64],
):
    """Helper function that is a vectorized version of the equivalent but roughly 100x slower

    .. code-block:: python

       np.array([np.arange(start, stop) for start, stop in zip(starts, stops)]).flatten()

    Solution adapted from Warren Weckesser's via https://stackoverflow.com/a/20033438


    Args:
        starts: Array of index range starts.
        stops:  Array of index range stops (exclusive).
        counts: Corresponds to stops - starts. 0-count ranges need to be excluded beforehand.

    Returns:

    """

    counts1 = counts[:-1]
    reset_index = np.cumsum(counts1)
    reset_values = 1 + starts[1:] - stops[:-1]
    incr = np.ones(counts.sum(), dtype=int)
    incr[0] = starts[0]
    incr[reset_index] = reset_values
    incr.cumsum(out=incr)
    return incr


def _make_range_boundaries(
    first: int, start: int, end: int, sbsq: int, stop: int
) -> npt.NDArray[np.int64]:
    """Turns the individual tuples output by :func:`_get_index_intervals_for_phrases` into four range boundaries.
    The four intervals are [first:start), [start:end], [end:sbsq), [sbsq:stop), which correspond to the components
    (ante, body, codetta, post) of the phrase. The body interval is right-inclusive, which means that the end
    symbol is included both in the body and, in the beginning of the 'codetta' or 'post' component.
    """
    return np.array([[first, start], [start, end + 1], [end, sbsq], [sbsq, stop]])


def make_take_mask_and_index(
    ix_intervals: List[Tuple[int, int, Optional[int], int, int]],
    logger: logging.Logger,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Takes a list of (first_i, start_i, end_i, subsequent_i, stop_i) index positions and turns them into

    * an array of corresponding index positions that can be used as argument for :meth:`pandas.DataFrame.take`
    * an array of equal length that specifies the corresponding phrase IDs (which come from an integer range)
    * an array of equal length that specifies the corresponding phrase components (ante, body, codetta, post)
    """
    range_boundaries = []
    for first, start, end, sbsq, last in ix_intervals:
        if end is None:
            logger.info("Skipping phrase with undefined end symbol.")
            continue
        range_boundaries.append(_make_range_boundaries(first, start, end, sbsq, last))
    ranges = np.vstack(range_boundaries)
    starts, stops = ranges.T
    counts = stops - starts
    not_empty_mask = counts > 0
    if not_empty_mask.any():
        take_mask = _make_concatenated_ranges(
            starts[not_empty_mask], stops[not_empty_mask], counts[not_empty_mask]
        )
    else:
        take_mask = _make_concatenated_ranges(starts, stops, counts)
    n_repeats = int(counts.shape[0] / 4)
    phrase_ids = np.repeat(np.arange(n_repeats), 4)
    names = np.tile(np.array(["ante", "body", "codetta", "post"]), n_repeats)
    id_level = phrase_ids.repeat(counts)
    name_level = names.repeat(counts)
    return take_mask, id_level, name_level


def make_raw_phrase_df(
    feature_df: D,
    ix_intervals: List[Tuple[int, int, Optional[int], int, int]],
    logger: Optional[logging.Logger] = None,
):
    """Takes the intervals generated by :meth:`get_index_intervals_for_phrases` and returns a dataframe with two
    additional index levels, one expressing a running count of phrases used as IDs, and one exhibiting for each phrase
    between one and for of the phrase_component names (ante, body, codetta, post), where 'body' is guaranteed to be
    ,present.
    """
    if logger is None:
        logger = module_logger
    take_mask, id_level, name_level = make_take_mask_and_index(
        ix_intervals, logger=logger
    )
    phrase_df = feature_df.take(take_mask)
    old_index = phrase_df.index.to_frame(index=False)
    new_levels = pd.DataFrame(
        dict(
            phrase_id=id_level,
            phrase_component=name_level,
        )
    )
    nlevels = phrase_df.index.nlevels
    new_index = pd.concat(
        [
            old_index.take(range(nlevels - 1), axis=1),
            new_levels,
            old_index.take([-1], axis=1),
        ],
        axis=1,
    )
    phrase_df.index = pd.MultiIndex.from_frame(new_index)
    # here we correct durations for the fact that the end symbol is included both as last symbol of the body and the
    # first symbol of the codetta or subsequent phrase. At the end of the body, the duration is set to 0.
    body_end_positions = _get_body_end_positions_from_raw_phrases(phrase_df)
    duration_col_position = phrase_df.columns.get_loc("duration_qb")
    phrase_df.iloc[body_end_positions, duration_col_position] = 0.0
    return phrase_df


def _get_body_end_positions_from_raw_phrases(phrase_df: D) -> List[int]:
    """Returns for each phrase body the index position of the last row."""
    body_end_positions = []
    for (phrase_id, phrase_component), idx in phrase_df.groupby(
        ["phrase_id", "phrase_component"]
    ).indices.items():
        if phrase_component != "body":
            continue
        body_end_positions.append(idx[-1])
    return body_end_positions


def tuple_contains(series_with_tuples: S, *values: Hashable):
    """Function that can be used in queries passed to :meth:`PhraseLabels.filter_phrase_data` to select rows in which
    the column's tuples contain any of the given values.

    Example

    """
    values = set(values)
    return series_with_tuples.map(values.intersection).astype(bool)


class PhraseAnnotations(DcmlAnnotations):
    _auxiliary_column_names = ["label", "localkey", "chord"]
    _convenience_column_names = KEY_CONVENIENCE_COLUMNS + [
        "first_label",
        "last_label",
        "n_localkeys",
        "localkeys",
        "n_chords",
        "chords",
    ]
    _feature_column_names = ["phraseend"]
    _extractable_features = [FeatureName.PhraseComponents, FeatureName.PhraseLabels]
    _default_value_column = "duration_qb"

    class Schema(DcmlAnnotations.Schema):
        n_ante = mm.fields.Int()
        n_post = mm.fields.Int()

    def __init__(
        self,
        n_ante: int = 0,
        n_post: int = 0,
        format=None,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        """

        Args:
            n_ante:
                By default, each phrase includes information about the included labels from beginning to end. Specify a
                larger integer in order to include additional information on the n labels preceding the phrase. These
                are generally part of a previous phrase.
            n_post:
                By default, each phrase includes information about the included labels from beginning to end. Specify a
                larger integer in order to include additional information on the n labels following the phrase. These
                are generally part of a subsequent phrase.
            format: Not in use.
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where to store serialization data and its descriptor by default.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby: Name of the fields for grouping this resource (usually after a Grouper has been applied).
            playthrough:
                Defaults to ``Playthrough.SINGLE``, meaning that first-ending (prima volta) bars are dropped in order
                to exclude incorrect transitions and adjacencies between the first- and second-ending bars.
        """
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            playthrough=playthrough,
        )
        self.n_ante = n_ante
        self.n_post = n_post

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        level_names = feature_df.index.names
        if "phrase_id" in level_names and "phrase_component" in level_names:
            return feature_df
        feature_df = self._apply_playthrough(feature_df)
        feature_df.chord.ffill(inplace=True)
        feature_df = extend_keys_feature(feature_df)
        groupby_levels = feature_df.index.names[:-1]
        group_intervals = get_index_intervals_for_phrases(
            harmony_labels=feature_df,
            group_cols=groupby_levels,
            n_ante=self.n_ante,
            n_post=self.n_post,
            logger=self.logger,
        )
        ix_intervals = sum(group_intervals.values(), [])
        return make_raw_phrase_df(feature_df, ix_intervals, self.logger)

    def filter_phrase_data(
        self,
        columns: str | List[str] = "label",
        components: phraseComp | List[phraseComp] = "body",
        droplevels: bool = False,
        reverse: bool = False,
        new_level_name: str = "i",
        wide_format: bool = False,
        query: Optional[str] = None,
    ) -> D:
        """

        Args:
            columns:
                Column(s) to include in the result. Passing a list results in an additional index
                level called "column".
            components:
                Which of the four phrase components to include, ∈ {'ante', 'body', 'codetta', 'post'}.
            droplevels:
                Can be a boolean or any level specifier accepted by :meth:`pandas.MultiIndex.droplevel()`.
                If False (default), all levels are retained. If True, only the phrase_id level and
                the ``new_level_name`` are retained. In all other cases, the indicated (string or
                integer) value(s) must be valid and cause one of the index (or column if
                ``wide_format``) levelse to be dropped. ``new_level_name`` cannot be dropped. If
                ``wide_format`` is True, dropping 'phrase_id' will likely lead to an exception
                regarding duplicate index values.
            reverse:
                Pass True to reverse the order of harmonies so that each phrase's last label comes
                first.
            new_level_name:
                Defaults to 'i', which is the the name of the original level that will be replaced
                by this new one. The new one represents the individual integer range for each
                phrase, starting at 0.
            wide_format:
                Pass True to unstack the result so that the columns for each phrase are concatenated
                side by side.
            query:
                A convient way to include only those phrases in the result that match the criteria
                formulated in the string query. A query is a string and generally takes the form
                "<column_name> <operator> <value>". Several criteria can be combined using boolean
                operators, e.g. "localkey_mode == 'major' & label.str.contains('/')". This option
                is particularly interesting when used on :class:`PhraseLabels` because it enables
                queries based on the properties of phrases such as
                "body_n_modulations == 0 & end_label.str.contains('IAC')".  For the columns
                containing tuples, you can used a special function to filter those rows that
                contain any of the specified values:
                "@tuple_contains(body_chords, 'V(94)', 'V(9)', 'V(4)')".

        Returns:
            Dataframe representing partial information on the selected phrases in long or wide format.
        """
        phrase_df = self.phrase_df
        if query:
            if self.name == "PhraseAnnotations":
                phrase_df = phrase_df.query(query)
            else:
                # for PhraseComponents and PhraseLabels, the filtering is performed on their respective feature df,
                # then the phrase_df (which corresponds to a PhraseAnnotations dataframe) is filtered based on the
                # result
                filtered_df = self.df.query(query)
                idx = filtered_df.index
                mask = make_boolean_mask_from_set_of_tuples(
                    phrase_df.index, set(idx), idx.names
                )
                phrase_df = phrase_df[mask]
        return _transform_phrase_data(
            phrase_df=phrase_df,
            columns=columns,
            components=components,
            droplevels=droplevels,
            reverse=reverse,
            new_level_name=new_level_name,
            wide_format=wide_format,
        )

    @property
    def phrase_df(self) -> D:
        """Alias for :meth:`df`."""
        return self.df


class PhraseComponents(PhraseAnnotations):
    _extractable_features = None

    @property
    def phrase_df(self) -> D:
        """Returns the df that corresponds to the :class:`PhraseAnnotations` feature from which the
        PhraseComponents were derived.
        """
        return self._phrase_df

    def _format_dataframe(self, feature_df: D) -> D:
        self._phrase_df = super()._format_dataframe(feature_df)
        return condense_components(self._phrase_df)


class PhraseLabels(PhraseAnnotations):
    _extractable_features = None

    @property
    def phrase_df(self) -> D:
        """Returns the df that corresponds to the :class:`PhraseAnnotations` feature from which the
        PhraseLabels were derived.
        """
        return self._phrase_df

    def _format_dataframe(self, feature_df: D) -> D:
        self._phrase_df = super()._format_dataframe(feature_df)
        return condense_phrases(self._phrase_df)


# endregion Annotations
# region Controls


class Articulation(Feature):
    pass


# endregion Controls
# region Events


class NotesFormat(FriendlyEnum):
    """Format to display the notes in. NAME stands for note names, FIFTHS for the number of fifths from C,
    and MIDI for MIDI numbers."""

    NAME = "NAME"
    FIFTHS = "FIFTHS"
    MIDI = "MIDI"


def merge_tied_notes(feature_df, groupby=None):
    expected_columns = ("duration", "tied", "midi", "staff")
    if not all(col in feature_df.columns for col in expected_columns):
        raise DataframeIsMissingExpectedColumnsError(
            [col for col in expected_columns if col not in feature_df.columns],
            feature_df.columns.to_list(),
        )
    unique_values = feature_df.tied.unique()
    if 0 not in unique_values and -1 not in unique_values:
        # no tied notes (only <NA>) or has already been tied (only not-null value is 1)
        return feature_df
    if groupby is None:
        return merge_ties(feature_df)
    else:
        return feature_df.groupby(groupby, group_keys=False).apply(merge_ties)


def extend_notes_feature(feature_df):
    if "tpc_name" in feature_df.columns:
        return feature_df
    concatenate_this = [
        feature_df,
        ms3.transform(feature_df.tpc, ms3.tpc2name).rename("tpc_name"),
    ]
    feature_df = pd.concat(concatenate_this, axis=1)
    return feature_df


class Notes(Feature):
    _auxiliary_column_names = [
        "chord_id",
        "gracenote",
        "midi",
        "name",
        "nominal_duration",
        "octave",
        "scalar",
        "tied",
        "tremolo",
    ]
    _convenience_column_names = [
        "tpc_name",
    ]
    _feature_column_names = ["tpc"]
    _default_analyzer = "PitchClassVectors"
    _default_value_column = "tpc"

    class Schema(Feature.Schema):
        format = FriendlyEnumField(
            NotesFormat,
            load_default=NotesFormat.NAME,
            metadata=dict(
                expose=True,
                description="Format to display the notes in. NAME stands for note names, FIFTHS for the number of "
                "fifths from C, and MIDI for MIDI numbers.",
            ),
        )
        merge_ties = mm.fields.Boolean(
            load_default=False,
            metadata=dict(
                title="Merge tied notes",
                expose=True,
                description="If False (default), each row corresponds to a note head, even if it does not the full "
                "duration of the represented sounding event or even an onset. Setting to True results in "
                "notes being tied over to from a previous note to be merged into a single note with the "
                "summed duration. After the transformation, only note heads that actually represent a note "
                "onset remain.",
            ),
        )
        weight_grace_notes = mm.fields.Float(
            load_default=0.0,
            validate=mm.validate.Range(min=0.0, max=1.0),
            metadata=dict(
                title="Weight grace notes",
                expose=True,
                description="Set a factor > 0.0 to multiply the nominal duration of grace notes which, otherwise, have "
                "duration 0 and are therefore excluded from many statistics.",
            ),
        )

    def __init__(
        self,
        format: NotesFormat = NotesFormat.NAME,
        merge_ties: bool = False,
        weight_grace_notes: float = 0.0,
        resource: Optional[fl.Resource | str] = None,
        descriptor_filename: Optional[str] = None,
        basepath: Optional[str] = None,
        auto_validate: bool = True,
        default_groupby: Optional[str | list[str]] = None,
        playthrough: Playthrough = Playthrough.SINGLE,
    ) -> None:
        """

        Args:
            format:
                :attr:`format`. Format to display the notes in. The default NAME stands for note names, FIFTHS for
                the number of fifths from C, and MIDI for MIDI numbers.
            merge_ties:
                If False (default), each row corresponds to a note head, even if it does not the full duration of the
                represented sounding event or even an onset. Setting to True results in notes being tied over to from a
                previous note to be merged into a single note with the summed duration. After the transformation,
                only note heads that actually represent a note onset remain.
            weight_grace_notes:
                Set a factor > 0.0 to multiply the nominal duration of grace notes which, otherwise, have duration 0
                and are therefore excluded from many statistics.
            resource: An existing :obj:`frictionless.Resource`.
            descriptor_filename:
                Relative filepath for using a different JSON/YAML descriptor filename than the default
                :func:`get_descriptor_filename`. Needs to end on one of the file extensions defined in the
                setting ``package_descriptor_endings`` (by default 'resource.json' or 'resource.yaml').
            basepath: Where to store serialization data and its descriptor by default.
            auto_validate:
                By default, the DimcatResource will not be validated upon instantiation or change (but always before
                writing to disk). Set True to raise an exception during creation or modification of the resource,
                e.g. replacing the :attr:`column_schema`.
            default_groupby: Name of the fields for grouping this resource (usually after a Grouper has been applied).
            playthrough:
                Defaults to ``Playthrough.SINGLE``, meaning that first-ending (prima volta) bars are dropped in order
                to exclude incorrect transitions and adjacencies between the first- and second-ending bars.
        """
        super().__init__(
            format=format,
            resource=resource,
            descriptor_filename=descriptor_filename,
            basepath=basepath,
            auto_validate=auto_validate,
            default_groupby=default_groupby,
            playthrough=playthrough,
        )
        self._merge_ties = bool(merge_ties)
        self._weight_grace_notes = float(weight_grace_notes)

    @property
    def format(self) -> NotesFormat:
        return self._format

    @format.setter
    def format(self, format: NotesFormat):
        format = NotesFormat(format)
        if self.format == format:
            return
        if format == NotesFormat.NAME:
            new_formatted_column = "tpc_name"
        elif format == NotesFormat.FIFTHS:
            new_formatted_column = "tpc"
        elif format == NotesFormat.MIDI:
            new_formatted_column = "midi"
        else:
            raise NotImplementedError(f"Unknown format {format!r}.")
        if self.is_loaded and new_formatted_column not in self.field_names:
            raise FeatureIsMissingFormatColumnError(
                self.resource_name, new_formatted_column, format, self.name
            )
        self._format = format
        self._formatted_column = new_formatted_column

    @property
    def merge_ties(self) -> bool:
        return self._merge_ties

    @property
    def weight_grace_notes(self) -> float:
        return self._weight_grace_notes

    def _format_dataframe(self, feature_df: D) -> D:
        """Called by :meth:`_prepare_feature_df` to transform the resource dataframe into a feature dataframe.
        Assumes that the dataframe can be mutated safely, i.e. that it is a copy.
        """
        feature_df = self._apply_playthrough(feature_df)
        feature_df = self._drop_rows_with_missing_values(
            feature_df, column_names=self._feature_column_names
        )
        if self.merge_ties:
            try:
                groupby = self.get_grouping_levels(UnitOfAnalysis.PIECE)
            except ResourceIsMissingPieceIndexError:
                groupby = None
                self.logger.info(
                    "Dataframe has no piece index. Merging ties without grouping."
                )
            feature_df = merge_tied_notes(feature_df, groupby=groupby)
        if self.weight_grace_notes:
            feature_df = ms3.add_weighted_grace_durations(
                feature_df, self.weight_grace_notes
            )
        feature_df = extend_notes_feature(feature_df)
        return self._sort_columns(feature_df)


# endregion Events
# region Structure
class Measures(Feature):
    pass


# endregion Structure
# region helpers


# endregion helpers

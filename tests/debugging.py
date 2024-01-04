import os
from pathlib import Path

from dimcat import Dataset
from dimcat.data import resources
from dimcat.data.resources import DimcatIndex
from dimcat.steps import groupers
from dimcat.utils import resolve_path

from docs.notebooks.utils import make_criterion, make_stage_data


def resource_names(path):
    return os.sep.join(path.split(os.sep)[-2:])


def load_unittest_corpora():
    here = Path(__file__).parent
    package_path = (
        here / ".." / "unittest_metacorpus" / "unittest_metacorpus.datapackage.json"
    )
    return Dataset.from_package(package_path)


def load_dcml_corpora():
    here = Path(__file__).parent
    package_path = here / ".." / "docs" / "manual" / "dcml_corpora.datapackage.json"
    return Dataset.from_package(package_path)


def load_distant_listening_corpus():
    package_path = resolve_path(
        "~/distant_listening_corpus/distant_listening_corpus.datapackage.json"
    )
    return Dataset.from_package(package_path)


def make_grouper(D):
    piece_index = resources.PieceIndex.from_resource(D.get_metadata())
    grouping = {f"group_{i}": piece_index.sample(3) for i in range(1, 4)}
    return groupers.CustomPieceGrouper.from_grouping(grouping)


def get_piece_groups(dataset):
    input_package = dataset.inputs.get_package()
    piece_index = input_package.get_piece_index()
    n_groups = 4
    grouping = {f"group_{i}": piece_index.sample(i) for i in range(n_groups)}
    grouped_pieces = DimcatIndex.from_grouping(grouping)
    return grouped_pieces


if __name__ == "__main__":
    D = load_distant_listening_corpus()
    phrase_annotations = D.get_feature("PhraseAnnotations")
    uncompressed = make_stage_data(
        phrase_annotations,
        columns=["chord_and_mode", "duration_qb"],
        wide_format=False,
    )
    criterion = make_criterion(
        phrase_annotations,
        columns=["root"],
        criterion_name="root_degree",
        join_str=False,
    )
    print(uncompressed.regroup_phrases(criterion))

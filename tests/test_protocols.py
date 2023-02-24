from dimcat.analyzer.base import Result
from dimcat.data import AnalyzedData
from dimcat.dtypes import PNotesTable


def test_tabular_type(dataset):
    notes = dataset.get_facet("notes")
    assert isinstance(notes, PNotesTable)


def test_result_type(dataset, analyzer):
    D = analyzer.process_data(dataset)
    assert isinstance(D, AnalyzedData)
    result = D.get_result_object()
    result_type = type(result)
    assert issubclass(
        result_type, Result
    ), f"{analyzer.__class__} doesn't produce a Result, but a {result_type}"
    if not issubclass(result_type, Result):
        print(f"{analyzer.__class__} doesn't produce a Result, but a {result_type}")
        assert issubclass(result_type, Result)
    print()
    print(result)

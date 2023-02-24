from dataclasses import asdict
from enum import Enum
from typing import Type, Union

import pytest
from dimcat.data.facet import get_facet_class
from dimcat.data.loader import DcmlLoader
from dimcat.dtypes.base import ConfiguredDataframe


@pytest.fixture()
def loader(small_corpora_path):
    return DcmlLoader(directory=small_corpora_path)


def default_methods_test(
    configured_dataframe: ConfiguredDataframe,
    supposed_name: Union[str, Enum],
    constructor: Type[ConfiguredDataframe],
):
    assert isinstance(configured_dataframe, constructor)
    assert configured_dataframe.dtype is supposed_name
    assert configured_dataframe.name == supposed_name
    facet_id = configured_dataframe.identifier
    facet_df = configured_dataframe.df
    facet_config = configured_dataframe.config
    facet1 = constructor.from_config(
        df=facet_df, config=facet_config, identifiers=facet_id
    )
    assert facet1 == configured_dataframe
    facet2 = constructor.from_id(facet_id, df=facet_df)
    assert facet2 == configured_dataframe
    facet3 = constructor.from_default(df=facet_df, identifiers=facet_id)
    assert facet3 == configured_dataframe
    facet4 = constructor.from_df(df=facet_df, identifiers=facet_id)
    assert facet4 == configured_dataframe
    facet5 = constructor(**asdict(configured_dataframe))
    assert facet5 == configured_dataframe


def test_loader(loader):
    for piece in loader.iter_pieces():
        available = piece.get_available_facets()
        for facet_name in available.keys():
            print(f"Testing {facet_name} returned by {piece.piece_id}:")
            facet_class = get_facet_class(facet_name)
            extracted_facet_object = piece.get_facet(facet_name)
            default_methods_test(extracted_facet_object, facet_name, facet_class)

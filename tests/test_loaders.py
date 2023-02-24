from dataclasses import asdict

import pytest
from dimcat.data.facet import get_facet_class
from dimcat.data.loader import DcmlLoader


@pytest.fixture()
def loader(small_corpora_path):
    return DcmlLoader(directory=small_corpora_path)


def test_loader(loader):
    for piece in loader.iter_pieces():
        available = piece.get_available_facets()
        for facet_name in available.keys():
            print(f"Testing {facet_name} returned by {piece.piece_id}:")
            facet_class = get_facet_class(facet_name)
            extracted_facet_object = piece.get_facet(facet_name)
            assert isinstance(extracted_facet_object, facet_class)
            assert extracted_facet_object.dtype is facet_name
            assert extracted_facet_object.name == facet_name
            facet_id = extracted_facet_object.identifier
            facet_df = extracted_facet_object.df
            facet_config = extracted_facet_object.config
            facet1 = facet_class.from_config(
                df=facet_df, config=facet_config, identifiers=facet_id
            )
            assert facet1 == extracted_facet_object
            facet2 = facet_class.from_id(facet_id, df=facet_df)
            assert facet2 == extracted_facet_object
            facet3 = facet_class.from_default(df=facet_df, identifiers=facet_id)
            assert facet3 == extracted_facet_object
            facet4 = facet_class.from_df(df=facet_df, identifiers=facet_id)
            assert facet4 == extracted_facet_object
            facet5 = facet_class(**asdict(extracted_facet_object))
            assert facet5 == extracted_facet_object

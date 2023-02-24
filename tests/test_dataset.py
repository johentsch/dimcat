import pytest
from dimcat.data.loader import DcmlLoader


def test_duplicating_dcml_loader(dataset):
    dataset.show_available_facets()
    ms3_parse = dataset.loaders[0].loader
    new_loader = DcmlLoader()
    new_loader.set_loader(ms3_parse)
    with pytest.raises(ValueError):
        dataset.set_loader(new_loader)

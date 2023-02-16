import pytest
from dimcat.base import Data, PipelineStep

PS_TYPES = dict(PipelineStep._registry)
D_TYPES = dict(Data._registry)


@pytest.fixture(ids=list(D_TYPES.keys()), params=D_TYPES.values())
def data_subclass(request):
    return request.param


def test_data_subclasses(data_subclass):
    data_obj = data_subclass()
    print(data_obj.name)
    assert isinstance(data_obj.name, str)


@pytest.fixture(ids=list(PS_TYPES.keys()), params=PS_TYPES.values())
def ps_subclass(request):
    return request.param


def test_ps_subclasses(ps_subclass):
    pipeline_step = ps_subclass()
    print(pipeline_step.name)
    assert isinstance(pipeline_step.name, str)

import pytest

from eski import models


@pytest.fixture
def registered_system(request):
    """Return requested registered system by key"""
    model = request.node.funcargs.get("registered_system_key")
    return models.system_from_model(model)

import pytest

from eski import models


@pytest.fixture
def registered_system(request):
    """Return requested registered system by key"""
    system = request.node.funcargs.get("registered_system_key")

    return models.registered_systems[system]

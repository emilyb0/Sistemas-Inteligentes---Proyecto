import pytest

from src.modelo import VERSIONES


@pytest.fixture(scope="module", params=VERSIONES)
def version(request):
    """Corre las pruebas del módulo una vez por versión del modelo (MODELO_VERSION)."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("MODELO_VERSION", request.param)
        mp.delenv("MODELO_PATH", raising=False)
        yield request.param

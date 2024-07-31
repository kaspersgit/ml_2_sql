import pytest
from ml2sql import main  # Import your app instance

@pytest.fixture
def runner():
    return app.test_cli_runner()


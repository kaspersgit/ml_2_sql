"""Top-level package for ml2sql."""
# ml2sql/__init__.py

__app_name__ = "ml2sql"
__version__ = "0.1.0"

(
    SUCCESS,
    DIR_ERROR,
    FILE_ERROR,
) = range(3)

ERRORS = {DIR_ERROR: "config directory error", FILE_ERROR: "config file error"}

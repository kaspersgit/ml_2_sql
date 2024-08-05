"""ml2sql entry point script."""
# ml2sql/main.py

from ml2sql.cli_run import cli_run
from ml2sql.cli_check_model import cli_check_model
from ml2sql.cli_init import cli_init
from ml2sql.quick_clean_data import quick_clean_data
from ml2sql import __version__, __app_name__
import typer
from pathlib import Path

app = typer.Typer()


def version_callback(value: bool):
    if value:
        print(f"{__app_name__} v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        callback=version_callback,
        is_eager=True,
        help="Show the version and exit.",
    ),
):
    pass


@app.command()
def init(
    dest: Path = typer.Option(
        Path.cwd(),
        help="Destination folder for the initialized project. Defaults to current directory.",
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        readable=True,
        resolve_path=True,
    ),
):
    """
    Initialize the project by creating necessary folders
    and copying demo data to them
    """

    cli_init(dest)


@app.command()
def run():
    """
    Run main script: clean data, train model, plot metrics and save SQL version
    """
    cli_run()


@app.command()
def check_model():
    """
    Run model check script: load trained model and apply on new data
    """
    cli_check_model()


@app.command()
def clean_data(
    data_path: str = typer.Option(
        None, "--data-path", help="Path to the CSV file to be cleaned"
    ),
):
    """
    Select a csv file and have it cleaned to be used for modeling
    """
    quick_clean_data(data_path)

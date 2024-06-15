"""ml2sql entry point script."""
# ml2sql/main.py

from ml2sql.cli_run import cli_run
from ml2sql.cli_check_model import cli_check_model
from ml2sql.quick_clean_data import quick_clean_data
from ml2sql import __version__, __app_name__
import typer
from pathlib import Path
import os

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
def init():
    """
    Initialize the project by creating necessary folders
    """

    # Get the current working directory
    current_dir = Path.cwd()

    # List of folders to create
    folders_to_create = ["input", "trained_models"]
    subfolders_to_create = {"input": ["data", "config"]}

    # Create folders
    for folder in folders_to_create:
        folder_path = current_dir / folder
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created folder: {folder_path}")
        except OSError as e:
            print(f"Error creating folder {folder_path}: {e.strerror}")

        if folder in subfolders_to_create:
            for subfolder in subfolders_to_create[folder]:
                subfolder_path = folder_path / subfolder
                try:
                    os.makedirs(subfolder_path, exist_ok=True)
                    print(f"Created folder: {subfolder_path}")
                except OSError as e:
                    print(f"Error creating folder {subfolder_path}: {e.strerror}")

    print("Project initialized successfully!")


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

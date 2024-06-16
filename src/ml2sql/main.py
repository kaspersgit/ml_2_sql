"""ml2sql entry point script."""
# ml2sql/main.py

from ml2sql.cli_run import cli_run
from ml2sql.cli_check_model import cli_check_model
from ml2sql.quick_clean_data import quick_clean_data
from ml2sql import __version__, __app_name__
import typer
from pathlib import Path
import importlib.resources as pkg_resources
import shutil
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

    # List of folders to create and their subfolders
    folders_structure = {"input": ["data", "configuration"], "trained_models": []}

    # Create folders and subfolders
    for folder, subfolders in folders_structure.items():
        folder_path = current_dir / folder
        try:
            os.makedirs(folder_path, exist_ok=True)
            print(f"Created folder: {folder_path}")
        except OSError as e:
            print(f"Error creating folder {folder_path}: {e.strerror}")

        for subfolder in subfolders:
            subfolder_path = folder_path / subfolder
            try:
                os.makedirs(subfolder_path, exist_ok=True)
                print(f"Created subfolder: {subfolder_path}")
            except OSError as e:
                print(f"Error creating subfolder {subfolder_path}: {e.strerror}")

    # Function to copy directory contents from source to destination
    def copy_directory_contents(src, dest):
        if not dest.exists():
            os.makedirs(dest)
        for item in os.listdir(src):
            s = os.path.join(src, item)
            d = os.path.join(dest, item)
            if os.path.isdir(s):
                copy_directory_contents(s, d)
            else:
                shutil.copy2(s, d)

    # Copy data directory contents
    try:
        with pkg_resources.as_file(
            pkg_resources.files(f"{__app_name__}.data").joinpath("data")
        ) as data_src_path:
            data_dest_path = current_dir / "input" / "data"
            copy_directory_contents(data_src_path, data_dest_path)
            print(f"Data directory contents copied to {data_dest_path}")
    except Exception as e:
        print(f"Error copying data directory contents: {e}")

    # Copy configuration directory contents
    try:
        with pkg_resources.as_file(
            pkg_resources.files(f"{__app_name__}.data").joinpath("configuration")
        ) as config_src_path:
            config_dest_path = current_dir / "input" / "configuration"
            copy_directory_contents(config_src_path, config_dest_path)
            print(f"Configuration directory contents copied to {config_dest_path}")
    except Exception as e:
        print(f"Error copying configuration directory contents: {e}")

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

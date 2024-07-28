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
        )
):
    """
    Initialize the project by creating necessary folders
    and copying demo data to them
    """

    # List of folders to create and their subfolders
    folders_structure = {
        "input": ["data", "configuration"],
        "trained_models": []
    }

    # Create folders and subfolders
    for folder, subfolders in folders_structure.items():
        folder_path = dest / folder
        try:
            folder_path.mkdir(parents=True, exist_ok=True)
            typer.echo(f"Created folder: {folder_path}")

            for subfolder in subfolders:
                subfolder_path = folder_path / subfolder
                subfolder_path.mkdir(parents=True, exist_ok=True)
                typer.echo(f"Created subfolder: {subfolder_path}")

        except OSError as e:
            typer.echo(f"Error creating folder {folder_path}: {e}", err=True)
            raise typer.Exit(code=1)

    # Copy demo data from package data to input folder
    dest_data_path = dest / "input"

    # Using pkg_resources to access package data
    try:
        # For Python 3.9+
        if hasattr(pkg_resources, 'files'):
            data_path = pkg_resources.files(__app_name__).joinpath("data")
            if data_path.is_dir():
                for item in data_path.iterdir():
                    if item.is_file():
                        shutil.copy2(item, dest_data_path)
                    elif item.is_dir():
                        shutil.copytree(item, dest_data_path / item.name, dirs_exist_ok=True)
                typer.echo("Copied demo data to input folder")
            else:
                typer.echo(f"Warning: Demo data folder not found in package at {data_path}", err=True)
        # For Python 3.7-3.8
        else:
            with pkg_resources.files(__app_name__).joinpath("data") as data_path:
                if data_path.is_dir():
                    for item in data_path.iterdir():
                        if item.is_file():
                            shutil.copy2(item, dest_data_path)
                        elif item.is_dir():
                            shutil.copytree(item, dest_data_path / item.name, dirs_exist_ok=True)
                    typer.echo("Copied demo data to input/data folder")
                else:
                    typer.echo(f"Warning: Demo data folder not found in package at {data_path}", err=True)
    except Exception as e:
        typer.echo(f"Error copying demo data: {e}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"ml2sql project initialized in {dest}")


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

import sys

sys.path.append("scripts")

import os
import subprocess


def get_package_version(package_name, pip_path):
    try:
        output = subprocess.check_output(
            [pip_path, "show", package_name], universal_newlines=True
        )
        version_line = next(
            (line for line in output.splitlines() if "Version:" in line), None
        )
        if version_line:
            version = version_line.split("Version:")[1].strip()
            return version
        else:
            return "Package not found"
    except subprocess.CalledProcessError:
        return "Package not found"


def get_package_version_from_requirements(package_name, requirements_file):
    try:
        with open(requirements_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(package_name):
                    parts = line.split("==")
                    if len(parts) == 2:
                        version = parts[1]
                        return version
                    else:
                        return "Version not pinned"
            return "Package not found"
    except FileNotFoundError:
        return "Requirements file not found"


def test_interpret_model_version():
    package_name = "interpret"
    print(f"Checking for package: {package_name}")

    # Check platform, windows is different from linux/mac
    if sys.platform == "win32":
        pip_path = ".ml2sql\\Scripts\\pip"
    else:
        pip_path = ".ml2sql/bin/pip"

    installed_package_version = get_package_version(package_name, pip_path)
    print(f"The version in virt env {pip_path} is: {installed_package_version}")

    # Get version from requirements.txt
    requirements_file = "docs/requirements.txt"
    req_package_version = get_package_version_from_requirements(
        package_name, requirements_file
    )
    print(f"The version in {requirements_file} is: {req_package_version}")

    # Check if installed version matches version in requirements.txt
    assert installed_package_version == req_package_version

    # check tested models if they are of installed package version
    version_suffix = installed_package_version.replace(".", "")
    folder_path = "tests/model"

    print(f"Looking for files ending with '_v{version_suffix}' in '{folder_path}'")

    files_with_version = []
    for filename in os.listdir(folder_path):
        filename = filename.split(".")[0]
        files_with_version.append(filename.endswith(f"_v{version_suffix}"))

    # Checking if all models in testing direcotry are of installed version
    assert all(files_with_version)

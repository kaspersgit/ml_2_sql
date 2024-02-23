import os
import subprocess
import sys

VENV_NAME = ".ml2sql"

# Create virtual environment
if sys.platform.startswith("win"):
    print(f"Creating virtual environment '{VENV_NAME}'...")
    subprocess.run(["python", "-m", "venv", VENV_NAME], shell=True)
else:
    print(f"Creating virtual environment '{VENV_NAME}'...")
    subprocess.run(["python3", "-m", "venv", VENV_NAME])

# Install packages in virtual environment
print("Installing packages in virtual environment...")
if sys.platform.startswith("win"):
    install_packages = os.path.join(VENV_NAME, "Scripts", "pip")
else:
    install_packages = os.path.join(VENV_NAME, "bin", "pip")

subprocess.run([f"{install_packages}", "install", "-r", "requirements.txt"])

print("Virtual environment created and packages installed successfully.")

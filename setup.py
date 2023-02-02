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

# Activate virtual environment
if sys.platform.startswith("win"):
    activate_script = os.path.join(VENV_NAME, "Scripts", "activate")
    print(f"Activating virtual environment using '{activate_script}'...")
    subprocess.run(f"{activate_script}", shell=True)
else:
    activate_script = os.path.join(VENV_NAME, "bin", "activate")
    print(f"Activating virtual environment using '{activate_script}'...")
    subprocess.run(f"source {activate_script}")


# Install packages in virtual environment
print(f"Installing packages in virtual environment...")
subprocess.run(["pip", "install", "-r", "requirements.txt"])

print("Virtual environment created and packages installed successfully.")

# Activate virtual environment
if sys.platform.startswith("win"):
    activate_script = os.path.join(VENV_NAME, "Scripts", "activate")
    command = f"{activate_script}"
else:
    activate_script = os.path.join(VENV_NAME, "bin", "activate")
    command = f"source {activate_script}"

subprocess.call(command, shell=True)
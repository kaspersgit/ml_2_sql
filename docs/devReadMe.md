## Developing
### Setup venv
Mac/Linux:
```
python3 -m venv .ml2sql
source .ml2sql/bin/activate
python -m pip install --index-url https://pypi.org/simple \
    -r docs/requirements-dev.txt \
    -e .  # <- the app/pkg itself
```

Windows 
```
python -m venv .ml2sql
.ml2sql/Script/activate
python -m pip install --index-url https://pypi.org/simple \
    -r docs/requirements-dev.txt \
    -e .  # <- the app/pkg itself
```

### Testing
- Activate `.ml2sql` venv
- Run `pytest`

### Package management (pinning)
With the virtual env activated
- Compile user requirements.txt file: `python -m piptools compile --index-url=https://pypi.org/simple -o docs/requirements.txt pyproject.toml`
- Compile dev requirements-dev.txt file: `python -m piptools compile --index-url=https://pypi.org/simple --extra dev -o docs/requirements-dev.txt -c docs/requirements.txt pyproject.toml`
  (Making sure packages in both files have the same version, [stackoverflow source](https://stackoverflow.com/questions/76055688/generate-aligned-requirements-txt-and-dev-requirements-txt-with-pip-compile))

### Building package
https://packaging.python.org/en/latest/tutorials/packaging-projects/

### Release process
PyPI Release Process

Update your package version
Update the version in your __init__.py or wherever you keep your version number.
Update CHANGELOG.md
Document the changes in this new version.
Commit changes and tag the release
git add .
git commit -m "Prepare release X.Y.Z"
git tag vX.Y.Z
git push origin main
git push origin vX.Y.Z

Build your package
rm -rf dist/*  # Clean old builds
python -m build

Upload to TestPyPI
twine upload --repository testpypi dist/*

Test installation from TestPyPI
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ ml2sql

Test the installed package
Ensure it works as expected.
Upload to PyPI
If all tests pass:
twine upload dist/*

Verify PyPI installation
pip install ml2sql

Create a GitHub release (if applicable)
Go to your GitHub repository and create a new release based on the tag you just pushed.

Remember to increment your version number appropriately for each release, following semantic versioning principles.
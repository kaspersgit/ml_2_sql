## Developing
### Setup venv
Mac/Linux:
```
python3 -m venv .ml2sql
.ml2sql/bin/python -m pip install --index-url https://pypi.org/simple -r docs/requirements-dev.txt
```

Windows 
```
python -m venv .ml2sql
.ml2sql/Scripts/python -m pip install --index-url https://pypi.org/simple -r docs/requirements-dev.txt
```

### Testing
- Activate `.ml2sql` venv
- Run `pytest`

### Package management (pinning)
With the virtual env activated
- Compile user requirements.txt file: `python -m piptools compile -o docs/requirements.txt pyproject.toml`
- Compile dev requirements-dev.txt file: `python -m piptools compile --extra dev -o docs/requirements-dev.txt -c docs/requirements.txt pyproject.toml`
  (Making sure packages in both files have the same version, [stackoverflow source](https://stackoverflow.com/questions/76055688/generate-aligned-requirements-txt-and-dev-requirements-txt-with-pip-compile))
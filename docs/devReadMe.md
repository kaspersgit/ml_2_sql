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
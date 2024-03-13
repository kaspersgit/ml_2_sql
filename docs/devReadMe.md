## Developing
### Setup venv
- `python3 -m venv .ml2sql`
- `source .ml2sql/bin/activate`
- `pip install requirements-dev.txt`

### Testing
- Activate `.ml2sql` venv
- Run `pytest`

### Package management
With the virtual env activated
- Compile user requirements.txt file: `python -m piptools compile -o docs/requirements.txt pyproject.toml`
- Compile dev requirements-dev.txt file: `python -m piptools compile --extra dev -o docs/requirements-dev.txt -c docs/requirements.txt pyproject.toml`
  (Making sure packages in both files have the same version, [stackoverflow source](https://stackoverflow.com/questions/76055688/generate-aligned-requirements-txt-and-dev-requirements-txt-with-pip-compile))
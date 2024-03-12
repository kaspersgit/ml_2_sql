## Developing
### Setup venv
- `python3 -m venv .ml2sql`
- `source .ml2sql/bin.activate`
- `pip install -r requirements.txt -r requirements-dev.txt`

### Testing
- Activate `.ml2sql` venv
- Run `pytest`

### Package management
- Create requirements.txt file through `python -m piptools compile -o requirements.txt pyproject.toml`

# Contributing: logging, error handling, and code style

Shared standards for all pipeline code (ingestion, features, model, agent),
so every step logs and fails the same way and the code stays consistent.

## Logging

Use the shared logger in `src/logging_utils.py`. At the top of a step:

~~~python
from src.logging_utils import setup_logging
setup_logging("my_step")          # console + logs/my_step.log
~~~

Do not call `logging.basicConfig` yourself and do not use `print` for status.

## Error handling

Use the helpers in `src/error_handling.py`.

Wrap each step's entry point so failures are logged with a traceback and the
process exits non-zero (so Airflow / `run_main` detect it):

~~~python
from src.error_handling import log_stage, require, DataUnavailableError

@log_stage("structured_features")
def main():
    require(export_dir.exists(),
            f"export dir missing: {export_dir}", DataUnavailableError)
    ...
~~~

- `@log_stage("name")`: logs `START` / `DONE` / `FAILED` with timing.
- `with stage("name"):`: same, for part of a function.
- `require(cond, msg, ErrorType)`: raise a clear, typed error when a check fails.

Raise a typed error, not a bare `Exception`:

| Situation | Exception |
| --- | --- |
| Source missing / empty / unreachable | `DataUnavailableError` |
| Schema / column / value check failed | `DataValidationError` |
| Missing or invalid setting / env var | `ConfigError` |

## Code style (PEP 8)

Format and lint before every commit:

~~~bash
pip install -r requirements-dev.txt   # one time
black .
flake8 .
~~~

Line length is 100 (set in `pyproject.toml` and `.flake8`).

## Tests

~~~bash
pytest
~~~

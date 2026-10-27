# Contributing: logging, error handling, and code style

Shared standards for all pipeline code (ingestion, features, model, agent),
so every step logs and fails the same way and the code stays consistent.

## Logging

- Scripts in `src/` use the shared logger in `src/logging_utils.py`. At the top of a step:

  ~~~python
  from src.logging_utils import run_main
  run_main(main, "my_step")       # console + logs/my_step.log, exit 1 on error
  ~~~

- Pipeline stages in `Data-Pipeline/scripts/` use `tracked_step("name")` from
  `Data-Pipeline/scripts/tracking.py`, which also writes a run record and sends the
  failure alert.

Do not use `print` for status. (`src/bias/run_bias_*.py` still set up logging with
`logging.basicConfig`; move them to `run_main` when they are next changed.)

## Error handling

Use the helpers in `src/error_handling.py`.

Raise typed errors at the failure points (missing or unreadable data, bad values,
missing settings). `run_main` / `tracked_step` log the traceback once and exit
non-zero, so Airflow marks the task failed. Use `@log_stage` / `stage()` to time
sub-steps inside a long function:

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

Raise a typed error, not a bare `Exception`. Each one is also the matching
built-in error, so existing `except FileNotFoundError` / `except ValueError` code
keeps working:

| Situation | Exception |
| --- | --- |
| Source missing / empty / unreachable | `DataUnavailableError` (a `FileNotFoundError`) |
| Schema / column / value check failed | `DataValidationError` (a `ValueError`) |
| Missing or invalid setting / env var | `ConfigError` (a `RuntimeError`) |

## Code style (PEP 8)

Lint before every commit:

~~~bash
pip install -r requirements-dev.txt   # one time
flake8 .
~~~

Line length is 99, PEP 8's upper limit (set in `pyproject.toml` and `.flake8`).
`black` is configured but optional: run it only on files you are changing, so it
does not reformat other people's open work.

## Tests

~~~bash
pip install -r requirements.txt -r requirements-dev.txt
pytest                                   # src/ code: tests/
cd Data-Pipeline && pip install -r requirements.txt && pytest   # pipeline: Data-Pipeline/tests/
~~~

## Branches

Work on a branch named `<name>/<topic>` (for example `nithish/s1-text-cleaning`) and
push it; the integration branch merges everyone's branches so each commit keeps its author.

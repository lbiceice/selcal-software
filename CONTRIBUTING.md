# Contributing to SelCal

Thank you for considering a contribution. SelCal is maintained by Bingbing Liang
(School of AI Engineering, Guangzhou College of Technology and Business).

## Reporting a problem or asking a question

- Open an issue at https://github.com/lbiceice/selcal-software/issues. Include the SelCal
  version (`selcal --version`), Python and NumPy versions, operating system, the exact command
  and its full output. If a result looks wrong, attach the saved record (`run.sqlite`) or an
  export folder (`selcal export`) and say what you expected.
- Do not post private or restricted data. A small synthetic series that reproduces the problem
  is the most useful attachment.
- For questions that are not bug reports, e-mail vincentdsnail@gmail.com.

## Proposing a change

1. Fork the repository and create a branch from `main`.
2. Install the development environment: `uv sync --locked --all-extras` (or
   `python -m pip install -e ".[dev]"`).
3. Make the change with a test that fails before it and passes after it.
4. Run the full checks before opening a pull request:
   - `uv run --locked --all-extras python -m pytest`
   - `uv run --locked --all-extras ruff check src tests scripts`
   - `uv run --locked --all-extras mypy src`
5. Open a pull request that explains what changed and why, and links the issue it addresses.

Statistical changes (null models, statistics, p-value or tie rules, the attainability guard)
must keep the existing reference records verifiable or explain, with evidence, why a record's
decision changes. Changes to saved-record formats need a migration or a clear refusal message
for older records.

## Maintenance and releases

- Supported: the latest release on Python 3.11-3.13 with NumPy >= 1.26, < 2.5, on macOS,
  Linux and Windows. The hosted workflow builds and tests the source archive on all three
  systems for every release candidate.
- Releases follow semantic versioning and are listed in `CHANGELOG.md`. Each release is tagged
  in this repository and archived on Zenodo (concept DOI 10.5281/zenodo.23261952, which always
  resolves to the latest version).
- Bug reports are answered as time allows, normally within two weeks; fixes for incorrect
  results take priority over new features.

## Licence

By contributing you agree that your contribution is licensed under the BSD-3-Clause licence
of this repository (`LICENSE.txt`).

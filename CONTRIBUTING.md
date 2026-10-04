# Contributing to p9kit

Thanks for taking a look. Bug reports, questions and patches are all welcome.

## Reporting a problem

Open an issue with: what you ran, what happened, what you expected, and your Python and
package versions (`python -c "import p9kit; print(p9kit.__version__)"`). A short snippet that
reproduces the problem is worth more than a long description.

## Asking a question

Open an issue with the "question" label. Questions about whether a result is meaningful are
welcome too, since that is most of what this package is for.

## Making a change

1. Fork the repository and create a branch.
2. Install the development extras: `pip install -e ".[dev]"`.
3. Add or update a test in `tests/` that fails before your change and passes after it.
4. Run `pytest` locally; everything should pass in under a minute without network access.
5. Open a pull request describing what changed and why.

## House style

* Tests must not need the internet. Code that downloads data is tested through its pure parts.
* Any new scientific claim needs a test that would catch it being wrong, not just a smoke test.
* Honest comments are preferred over confident ones: if an approximation is crude, say so in
  the docstring.

## Code of conduct

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

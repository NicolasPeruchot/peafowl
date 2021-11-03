develop:
	pip install -e '.[dev]'
	pre-commit install

setup-tests:
	pip install -e '.[test]'

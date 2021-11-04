develop:
	pip install -e '.[dev]'
	python -m spacy download en_core_web_md
	pre-commit install

setup-tests:
	pip install -e '.[test]'

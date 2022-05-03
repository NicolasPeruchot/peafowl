develop:
	pip install -e '.[dev]'
	python -m spacy download en_core_web_md
	pre-commit install

setup-tests:
	pip install -e '.[test]'
	python -m spacy download en_core_web_md

test:
	pytest

lda-app:
	streamlit run peafowl/app/lda.py

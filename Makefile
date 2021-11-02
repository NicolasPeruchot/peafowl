install:
	python -m pip install -r requirements.txt
	pre-commit install
	python -m spacy download en_core_web_sm

"""Preprocessing functions."""

from typing import List

import spacy


def is_junk(token: spacy.tokens.token.Token) -> bool:
    """Test if token is junk."""
    return token.is_stop or token.is_punct


def lemmatizer(doc: spacy.tokens.doc.Doc) -> List[str]:
    """Lemmatize one doc."""
    lemma = [token.lemma_ for token in doc if is_junk(token) is False]
    return lemma


def lemmatizer_dataset(data: List[str]) -> List[List[str]]:
    """Lemmatize a Series of articles."""
    nlp = spacy.load("en_core_web_md")
    docs = list(nlp.pipe(data))
    lemmatize_data = list(map(lemmatizer, docs))
    return lemmatize_data

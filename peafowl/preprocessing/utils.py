"""Preprocessing functions."""

from typing import List

import pandas as pd
import spacy


def is_not_junk(token: spacy.tokens.token.Token) -> bool:
    """Test if token is junk."""
    return token.is_stop is False and token.is_punct is False


def lemmatizer(text: str, nlp: spacy.language.Language) -> List[str]:
    """Lemmatize one text."""
    doc = nlp(text)
    lemma = [token.lemma_ for token in doc if is_not_junk(token)]
    return lemma


def lemmatizer_dataset(data: pd.Series) -> pd.Series:
    """Lemmatize a Series of articles."""
    nlp = spacy.load("en_core_web_md")
    lemmatize_data = data.apply(lambda x: lemmatizer(x, nlp))
    return lemmatize_data

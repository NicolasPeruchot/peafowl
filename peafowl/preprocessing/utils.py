"""Preprocessing functions."""

from typing import List

import spacy


def is_not_junk(token: spacy.tokens.token.Tokenn):
    """Test if token is junk."""
    return token.is_stop is False and token.is_punct is False


def lemmatizer(data: List) -> List:
    """Lemmatize a list of articles."""
    nlp = spacy.load("en_core_web_md")
    lemmatized_articles = []
    for sentence in data:
        doc = nlp(sentence)
        lemma = [token.lemma_ for token in doc if is_not_junk(token)]
        if lemma:
            lemmatized_articles.append(lemma)
    return lemmatized_articles

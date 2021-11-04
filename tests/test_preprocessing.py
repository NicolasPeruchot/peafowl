"""Preprocessing tests."""

import spacy

from peafowl.preprocessing.utils import is_not_junk, lemmatizer


def test_lemmatizer():
    """Test the lemmatizer."""
    # given
    text = "This text should be lemmatize."
    expected_output = ["text", "lemmatize"]
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    # when
    output = lemmatizer(doc=doc)
    # then
    assert expected_output == output


def test_is_not_junk():
    """Test is_not_junk function."""
    # given
    text = "Some words are junk!"
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    expected_output = [False, True, False, True, False]
    # when
    output = [is_not_junk(token) for token in doc]
    # then
    assert expected_output == output

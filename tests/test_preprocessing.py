"""Preprocessing tests."""

import spacy

from peafowl.preprocessing.utils import is_junk, lemmatizer


def test_lemmatizer():
    """Test the lemmatizer."""
    # given
    text = "This text should be lemmatized."
    expected_output = ["text", "lemmatize"]
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    # when
    output = lemmatizer(doc=doc)
    # then
    assert expected_output == output


def test_is_junk():
    """Test is_junk function."""
    # given
    text = "Some words are junk!"
    nlp = spacy.load("en_core_web_md")
    doc = nlp(text)
    expected_output = [True, False, True, False, True]
    # when
    output = [is_junk(token) for token in doc]
    # then
    assert expected_output == output

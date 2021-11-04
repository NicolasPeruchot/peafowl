"""Pass the ci."""

import spacy

from peafowl.preprocessing.utils import lemmatizer


def test_lemmatizer():
    """Pass the ci."""
    # given
    sentence = "This sentence should be lemmatize."
    expected_output = ["sentence", "lemmatize"]
    nlp = spacy.load("en_core_web_md")
    # when
    output = lemmatizer(text=sentence, nlp=nlp)
    # then
    assert expected_output == output

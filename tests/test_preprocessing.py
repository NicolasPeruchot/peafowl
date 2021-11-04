"""Pass the ci."""

import spacy

from peafowl.preprocessing.utils import lemmatizer


def test_lemmatizer():
    """Pass the ci."""
    # given
    sentence = "This sentence should be lemmatize."
    expected_output = ["sentence", "lemmatize"]
    nlp = spacy.load("en_core_web_md")
    doc = nlp(sentence)
    # when
    output = lemmatizer(doc=doc)
    # then
    assert expected_output == output

"""Viz with pyLDAvis."""
import warnings

import numpy as np
import pyLDAvis
import tomotopy


def prepare_viz_LDA(model: tomotopy.LDAModel) -> pyLDAvis._prepare.PreparedData:
    """Prepare LDA Model for viz."""
    topic_term_dists = np.stack([model.get_topic_word_dist(k) for k in range(model.k)])
    doc_topic_dists = np.stack([doc.get_topic_dist() for doc in model.docs])
    doc_lengths = np.array([len(doc.words) for doc in model.docs])
    vocab = list(model.used_vocabs)
    term_frequency = model.used_vocab_freq

    warnings.filterwarnings("ignore")
    prepared_data = pyLDAvis.prepare(
        topic_term_dists, doc_topic_dists, doc_lengths, vocab, term_frequency
    )
    return prepared_data

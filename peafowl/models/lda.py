"""Functions for the GuidedLDA interface."""

import warnings

from typing import Dict, List

import pandas as pd
import tomotopy as tp

from peafowl.models.utils import prepare_viz_LDA
from peafowl.preprocessing.utils import lemmatizer_dataset


warnings.filterwarnings("ignore", category=DeprecationWarning)


class LDA:
    """LDA."""

    def __init__(self, n_topics: int = 0, seeds: Dict[str, List] = {}) -> None:
        """Init."""
        self.seeds = seeds
        self.is_guided = seeds != {}
        if n_topics != 0 and len(seeds) != 0:
            raise TypeError("Number of seeds and seeds can't be provided at the same time.")
        self.n = max(len(self.seeds), n_topics)
        self.model = tp.LDAModel(k=self.n)

    def explore(self, data: pd.Series, size: int = 10):
        """Show examples of data."""
        pd.set_option("display.max_colwidth", None)
        return data.sample(size)

    def fit(self, data: pd.Series):
        """Train on data."""
        data = lemmatizer_dataset(data)
        for x in data:
            self.model.add_doc(x)
        if self.is_guided:
            for k, v in self.seeds.items():
                for word in v:
                    self.model.set_word_prior(word, [1.0 if k == i else 0 for i in range(self.n)])
        for _ in range(0, 100, 10):
            self.model.train(10)
        return None

    def report(self):
        """Get a feel for the model's performance (topic labelling + topic coherence).

        References:
            - https://bab2min.github.io/tomotopy/v0.12.1/en/label.html
            - https://bab2min.github.io/tomotopy/v0.12.1/en/coherence.html
        """
        # extract candidates for auto topic labeling
        extractor = tp.label.PMIExtractor(min_cf=10, min_df=5, max_len=5, max_cand=10000)
        cands = extractor.extract(self.model)

        # ranking the candidates of labels for a specific topic
        labeler = tp.label.FoRelevance(self.model, cands, min_df=5, smoothing=1e-2, mu=0.25)
        for k in range(self.n):
            print("== Topic #{} ==".format(k))
            print(
                "Labels:", ", ".join(label for label, score in labeler.get_topic_labels(k, top_n=5))
            )
            for word, prob in self.model.get_topic_words(k, top_n=10):
                print(word, prob, sep="\t")

    def viz(self):
        """Visualisation for a trained model."""
        prepared_data = prepare_viz_LDA(model=self.model)
        return prepared_data

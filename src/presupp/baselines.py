"""Non-neural baselines, scored like the models.

prior gives every passage the training label frequencies; tfidf_lr is word 1-2 gram TF-IDF with
a class-balanced logistic regression per label, untuned.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from presupp.labels import target_matrix


def fit_predict(model_type: str, parts: dict, gold, labels: list[str], seed: int):
    """({part: probs} for every part but train, metadata)."""
    y = target_matrix(parts["train"]["passage_id"], gold, labels, "hard")
    others = {k: df for k, df in parts.items() if k != "train"}

    if model_type == "prior":
        prevalence = y.mean(0)
        return {k: np.tile(prevalence, (len(df), 1)) for k, df in others.items()}, {
            "prevalence": prevalence.tolist()}

    if model_type != "tfidf_lr":
        raise ValueError(f"unknown baseline {model_type!r}")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x_train = vectorizer.fit_transform(parts["train"]["text"])
    x_other = {k: vectorizer.transform(df["text"]) for k, df in others.items()}
    probs = {k: np.zeros((len(df), len(labels))) for k, df in others.items()}
    for j in range(len(labels)):
        if y[:, j].min() == y[:, j].max():  # no positives or no negatives
            for k in probs:
                probs[k][:, j] = y[0, j]
            continue
        clf = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)
        clf.fit(x_train, y[:, j])
        for k in probs:
            probs[k][:, j] = clf.predict_proba(x_other[k])[:, 1]
    return probs, {"vocabulary_size": len(vectorizer.vocabulary_)}

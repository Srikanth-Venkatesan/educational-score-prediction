import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from ..irt.irt_models import irt_2pl


class TfidfRidgeIRT:

    def __init__(self, max_features=20):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english"
        )
        self.diff_model = Ridge(alpha=1.0)
        self.disc_model = Ridge(alpha=1.0)

    def fit(self, train_q, num_cols):
        X_text = self.vectorizer.fit_transform(train_q["full_text"])
        X = np.hstack([X_text.toarray(), train_q[num_cols].values])

        self.diff_model.fit(X, train_q["Difficulty"])
        self.disc_model.fit(X, train_q["Discrimination"])

    def predict(self, test_df, num_cols):
        test_q = test_df.drop_duplicates("Question Id")

        X_text = self.vectorizer.transform(test_q["full_text"])
        X = np.hstack([X_text.toarray(), test_q[num_cols].values])

        test_q["b_hat"] = self.diff_model.predict(X).clip(-3, 3)
        test_q["a_hat"] = self.disc_model.predict(X).clip(0, 3)

        test_df = test_df.merge(
            test_q[["Question Id", "a_hat", "b_hat"]],
            on="Question Id",
            how="left"
        )

        test_df["p_correct"] = irt_2pl(
            test_df["ability"],
            test_df["a_hat"],
            test_df["b_hat"]
        )

        test_df["y_hat"] = (test_df["p_correct"] >= 0.5).astype(int)
        test_df["expected_score"] = test_df["y_hat"] * test_df["Mark"]

        pred_scores = (
            test_df.groupby("student_email")["expected_score"]
            .sum()
            .reset_index()
        )

        actual_scores = (
            test_df.groupby("student_email")["Quiz1_score"]
            .first()
            .reset_index()
        )

        return pred_scores.merge(actual_scores, on="student_email")

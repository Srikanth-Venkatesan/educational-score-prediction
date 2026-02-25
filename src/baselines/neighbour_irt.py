from ..irt.irt_models import irt_2pl


class NeighbourBasedIRT:
    """
    Expert-based IRT using precomputed mean difficulty
    and discrimination values.
    """

    def predict(self, df):
        df = df.copy()

        df["p_correct"] = irt_2pl(
            df["ability"],
            df["Mean_Discrimination"],
            df["Mean_Difficulty"]
        )

        df["y_hat"] = (df["p_correct"] >= 0.5).astype(int)
        df["expected_score"] = df["y_hat"] * df["Mark"]

        pred_scores = (
            df.groupby("student_email")["expected_score"]
            .sum()
            .reset_index()
        )

        actual_scores = (
            df.groupby("student_email")["Quiz1_score"]
            .first()
            .reset_index()
        )

        return pred_scores.merge(actual_scores, on="student_email")

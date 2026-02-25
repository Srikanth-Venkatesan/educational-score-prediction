import argparse
import pandas as pd

from src.ability.ability import compute_rank_based_ability
from src.baselines.regression_models import get_regression_models
from src.evaluation.metrics import rmse, mad
from src.evaluation.rolling import rolling_term_split
from src.utils.term_utils import sort_terms


def main(data_path):
    df = pd.read_csv(data_path)

    df["M1_ability"] = compute_rank_based_ability(df["M1_Score"])
    df["S1_ability"] = compute_rank_based_ability(df["S1_Score"])

    terms = sort_terms(df["stats2_term_code"].unique())
    models = get_regression_models()

    for train_df, test_df, train_term, test_term in rolling_term_split(
        df, "stats2_term_code", terms
    ):
        X_train = train_df[["M1_Score", "S1_Score"]]
        y_train = train_df["Quiz1_score"]

        X_test = test_df[["M1_Score", "S1_Score"]]
        y_test = test_df["Quiz1_score"]

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            print(
                f"{name} | Train till {train_term} → "
                f"Predict {test_term} | "
                f"RMSE={rmse(y_test, preds):.2f}, "
                f"MAD={mad(y_test, preds):.2f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()

    main(args.data_path)

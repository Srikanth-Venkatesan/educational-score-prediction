import argparse
import pandas as pd

from src.ability.ability import compute_rank_based_ability
from src.baselines.neighbour_irt import NeighbourBasedIRT
from src.evaluation.metrics import rmse, mad
from src.evaluation.rolling import rolling_term_split
from src.utils.term_utils import sort_terms


def main(data_path):
    df = pd.read_csv(data_path)

    # ---------- Ability Computation ----------
    df["M1_ability"] = compute_rank_based_ability(df["M1_Score"])
    df["S1_ability"] = compute_rank_based_ability(df["S1_Score"])
    df["ability"] = 0.5 * df["M1_ability"] + 0.5 * df["S1_ability"]

    # ---------- Sort Terms ----------
    terms = sort_terms(df["stats2_term_code"].unique())

    model = NeighbourBasedIRT()

    for train_df, test_df, train_term, test_term in rolling_term_split(
        df, "stats2_term_code", terms
    ):

        # Neighbour model does NOT train
        eval_df = model.predict(test_df)

        print(
            f"Neighbour-IRT | Train till {train_term} → "
            f"Predict {test_term} | "
            f"RMSE={rmse(eval_df['Quiz1_score'], eval_df['expected_score']):.2f}, "
            f"MAD={mad(eval_df['Quiz1_score'], eval_df['expected_score']):.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()

    main(args.data_path)

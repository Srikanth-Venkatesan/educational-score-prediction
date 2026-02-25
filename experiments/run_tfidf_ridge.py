import argparse
import pandas as pd

from src.ability.ability import compute_rank_based_ability
from src.baselines.tfidf_ridge import TfidfRidgeIRT
from src.evaluation.metrics import rmse, mad
from src.evaluation.rolling import rolling_term_split
from src.utils.term_utils import sort_terms


NUM_COLS = [
    "solution_word_count",
    "tax_Identify - relevant information/variables",
    "tax_Relationship",
    "tax_Formulation",
    "tax_Evaluation",
    "tax_Reasoning"
]


def main(data_path):
    df = pd.read_csv(data_path)

    # ---------- Ability Computation ----------
    df["M1_ability"] = compute_rank_based_ability(df["M1_Score"])
    df["S1_ability"] = compute_rank_based_ability(df["S1_Score"])
    df["ability"] = 0.5 * df["M1_ability"] + 0.5 * df["S1_ability"]

    # ---------- Create full_text column ----------
    df["question_type"] = df["question_type"].astype(str).str.upper().str.strip()

    df["full_text"] = (
        "__QTYPE_" + df["question_type"] + "__ " +
        df["question_text"].fillna("")
    )

    # ---------- Sort Terms ----------
    terms = sort_terms(df["stats2_term_code"].unique())

    for train_df, test_df, train_term, test_term in rolling_term_split(
        df, "stats2_term_code", terms
    ):

        # ---------- Train on question-level data ----------
        train_q = train_df.drop_duplicates("Question Id")

        model = TfidfRidgeIRT()
        model.fit(train_q, NUM_COLS)

        # ---------- Predict ----------
        eval_df = model.predict(test_df, NUM_COLS)

        print(
            f"TFIDF+Ridge | Train till {train_term} → "
            f"Predict {test_term} | "
            f"RMSE={rmse(eval_df['Quiz1_score'], eval_df['expected_score']):.2f}, "
            f"MAD={mad(eval_df['Quiz1_score'], eval_df['expected_score']):.2f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    args = parser.parse_args()

    main(args.data_path)

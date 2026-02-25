import pandas as pd


def compute_rank_based_ability(scores: pd.Series) -> pd.Series:
    """
    Convert raw scores into IRT-style ability [-3, 3]
    using percentile rank transformation.
    """
    ranks = scores.rank(pct=True)
    return 6 * (ranks - 0.5)

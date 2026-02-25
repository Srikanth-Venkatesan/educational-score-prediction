def rolling_term_split(df, term_col, sorted_terms):
    """
    Rolling-origin evaluation generator.
    """
    for i in range(len(sorted_terms) - 1):
        train_terms = sorted_terms[: i + 1]
        test_term = sorted_terms[i + 1]

        train_df = df[df[term_col].isin(train_terms)]
        test_df = df[df[term_col] == test_term]

        yield train_df, test_df, train_terms[-1], test_term

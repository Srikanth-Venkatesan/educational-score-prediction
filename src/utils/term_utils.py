def term_to_index(term):
    """
    Converts term code like 'F2-2023' into sortable index.
    """
    sem, year = term.split("-")
    sem_no = {"F1": 1, "F2": 2, "F3": 3}[sem]
    return int(year) * 10 + sem_no


def sort_terms(terms):
    """
    Sort term codes chronologically.
    """
    return sorted(terms, key=term_to_index)

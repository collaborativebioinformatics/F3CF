# Reads:  nothing directly (imported by temporal_accretion.py and temporal_cmf.py)
# Writes: nothing
# Does:   the temporal replay split: train on pairs first reported before a year, test on that year's new pairs


def split_year(edges, year, excluded_pmids=frozenset()):
    """(train, new pairs in year, test = new pairs between a gene and phenotype already in train).
    Pairs whose first-year reports all come from excluded publications are not new in that year."""
    train = edges[edges["year"] < year]
    new = edges[edges["year"] == year]
    if excluded_pmids:
        new = new[new["first_pmids"].str.split(";").map(lambda pmids: bool(set(pmids) - excluded_pmids))]
    known_nodes = new["gene"].isin(set(train["gene"])) & new["phenotype"].isin(set(train["phenotype"]))
    return train, new, new[known_nodes]

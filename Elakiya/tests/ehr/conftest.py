import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIXTURES = Path(__file__).parent / "fixtures"


def make_events(rows):
    """rows: dicts with person_id, kind, code_system, code and optional analyte/value/unit/date/source."""
    from metametagraphs.ehr.events import finalize

    df = pd.DataFrame(rows)
    for c, d in (("source", "test"), ("analyte", ""), ("value", np.nan), ("unit", ""), ("raw_value", ""), ("date", "")):
        if c not in df:
            df[c] = d
    df["date"] = df["date"].fillna("")
    return finalize(df)


def persons(sexes: dict) -> pd.DataFrame:
    return pd.DataFrame({"sex": list(sexes.values())}, index=pd.Index(list(sexes), name="person_id"))


@pytest.fixture(scope="session")
def toy_dir(tmp_path_factory):
    from metametagraphs.ehr.toy import write_toy

    d = tmp_path_factory.mktemp("toy")
    write_toy(d, n=300, seed=7)
    return d

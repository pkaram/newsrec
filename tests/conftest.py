from types import SimpleNamespace

import pytest

from tests.support import experiment_ratings


@pytest.fixture
def train_test_split():
    ratings = experiment_ratings()
    return SimpleNamespace(
        data_train=ratings.iloc[:80].reset_index(drop=True).copy(),
        data_test=ratings.iloc[80:].reset_index(drop=True).copy(),
    )

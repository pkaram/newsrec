from types import SimpleNamespace

import pandas as pd
import pytest

from newsrec.evaluation.evaluation import Evaluator
from newsrec.object_mappings import model_mappings
from newsrec.recommenders.bpr import BPR
from newsrec.recommenders.ease import EASE
from newsrec.recommenders.implicit import iALS
from newsrec.recommenders.itemknn import ItemKNN
from newsrec.recommenders.popularity import Popular
from newsrec.recommenders.svd import SVD


def fit(model, params, data, top_k=3):
    model.pass_parameters(params)
    model.train_model(data=data, k=top_k)
    return model


def assert_recommendations(model, data, top_k, test_users_only):
    recos = model.recos
    assert list(recos.columns) == ["userid", "itemid", "rank"]
    assert not recos.empty
    assert recos["userid"].notna().all()
    assert recos["itemid"].notna().all()
    assert recos.duplicated(["userid", "itemid"]).sum() == 0
    assert recos.groupby("userid").size().max() <= top_k

    for _, user_recos in recos.groupby("userid"):
        ranks = user_recos["rank"].tolist()
        assert ranks == sorted(ranks)
        assert len(ranks) == len(set(ranks))
        assert min(ranks) >= 0

    train_pairs = set(zip(data.data_train["userid"], data.data_train["itemid"]))
    reco_pairs = set(zip(recos["userid"], recos["itemid"]))
    assert train_pairs.isdisjoint(reco_pairs)
    if test_users_only:
        assert set(recos["userid"]) <= set(data.data_test["userid"])

    evaluator = Evaluator(model)
    evaluator.calculate_metrics()
    for value in evaluator.metrics().values():
        assert 0.0 <= float(value) <= 1.0


def test_supported_models_are_registered():
    assert set(model_mappings) == {"Popular", "iALS", "SVD", "BPR", "ItemKNN", "EASE"}


def test_popular_recommends_unseen_items_in_popularity_order():
    train = pd.DataFrame(
        {
            "userid": ["u1", "u1", "u3", "u3", "u3", "u4"],
            "itemid": ["a", "b", "a", "b", "c", "a"],
            "rating": [1, 1, 1, 1, 1, 1],
            "timestamp": [1, 2, 3, 4, 5, 6],
        }
    )
    test = pd.DataFrame(
        {
            "userid": ["u1", "u2"],
            "itemid": ["c", "a"],
            "rating": [1, 1],
            "timestamp": [7, 8],
        }
    )
    data = SimpleNamespace(data_train=train, data_test=test)

    model = fit(Popular(), 0, data, top_k=2)

    by_user = {
        user: list(zip(frame["itemid"], frame["rank"]))
        for user, frame in model.recos.groupby("userid")
    }
    assert by_user == {"u1": [("c", 0)], "u2": [("a", 0), ("b", 1), ("c", 2)]}


@pytest.mark.parametrize(
    ("cls", "params", "test_users_only"),
    [
        (SVD, {"singular_values_n": 2}, True),
        (iALS, {"factors": 4, "reg": 0.05, "iter": 2, "log": False}, False),
        (iALS, {"factors": 4, "reg": 0.05, "iter": 2, "log": True}, False),
        (BPR, {"factors": 4, "learning_rate": 0.01, "reg": 0.01, "iter": 2, "random_state": 42}, True),
        (ItemKNN, {"neighbors": 5, "similarity": "cosine"}, True),
        (ItemKNN, {"neighbors": 5, "similarity": "tfidf"}, True),
        (ItemKNN, {"neighbors": 5, "similarity": "bm25"}, True),
        (EASE, {"lamb": 50}, True),
    ],
)
def test_model_recommends_items_the_user_has_not_seen(cls, params, test_users_only, train_test_split):
    model = fit(cls(), params, train_test_split)
    assert_recommendations(model, train_test_split, top_k=3, test_users_only=test_users_only)


def test_parameter_defaults():
    svd = SVD()
    svd.pass_parameters({})
    assert svd.singular_values_n == 20

    ials = iALS()
    ials.pass_parameters({})
    assert (ials.factors, ials.reg, ials.iter, ials.log) == (100, 0.01, 15, False)

    bpr = BPR()
    bpr.pass_parameters({})
    assert bpr.factors == 100
    assert bpr.iter == 100
    assert bpr.random_state is None

    itemknn = ItemKNN()
    itemknn.pass_parameters({})
    assert (itemknn.neighbors, itemknn.similarity, itemknn.k1, itemknn.b) == (20, "cosine", 1.2, 0.75)

    ease = EASE()
    ease.pass_parameters(0)
    assert ease.lamb == 500


def zero_rating_split():
    train = pd.DataFrame(
        {"userid": ["u1", "u1"], "itemid": ["a", "b"], "rating": [0.0, 0.0], "timestamp": [1, 2]}
    )
    test = pd.DataFrame(
        {"userid": ["u1"], "itemid": ["c"], "rating": [1.0], "timestamp": [3]}
    )
    return SimpleNamespace(data_train=train, data_test=test)


@pytest.mark.parametrize(
    ("cls", "params", "message"),
    [
        (BPR, {}, "BPR needs at least one interaction with rating > 0"),
        (ItemKNN, {}, "ItemKNN needs at least one interaction with rating > 0"),
        (EASE, {"lamb": 50}, "EASE needs at least one interaction with rating > 0"),
    ],
)
def test_implicit_models_reject_ratings_that_are_not_positive(cls, params, message):
    model = cls()
    model.pass_parameters(params)
    with pytest.raises(ValueError, match=message):
        model.train_model(zero_rating_split(), k=3)


def test_itemknn_rejects_unknown_similarity(train_test_split):
    model = ItemKNN()
    model.pass_parameters({"neighbors": 5, "similarity": "jaccard"})

    with pytest.raises(ValueError, match="cosine"):
        model.train_model(train_test_split, k=3)


@pytest.mark.parametrize("lamb", [0, -5])
def test_ease_rejects_non_positive_lambda(lamb, train_test_split):
    model = EASE()
    model.pass_parameters({"lamb": lamb})

    with pytest.raises(ValueError, match="lamb must be positive"):
        model.train_model(train_test_split, k=3)

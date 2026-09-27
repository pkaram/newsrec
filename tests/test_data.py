import pandas as pd
import pytest

from newsrec.dataloader.dataloader import DataLoader, get_config
from newsrec.datasplitting.splitter import Splitter

from tests.support import COLUMNS


def ratings_frame():
    return pd.DataFrame(
        {
            "timestamp": [5, 4, 3, 2, 1],
            "rating": [1.0, 1.0, 1.0, 1.0, 1.0],
            "extra": ["x", "x", "x", "x", "x"],
            "itemid": ["i1", "i2", "i3", "i4", "i5"],
            "userid": ["u1", "u1", "u1", "u2", "u2"],
        }
    )


def write_ratings(path):
    pd.DataFrame(
        {
            "userid": ["u1", "u2"],
            "itemid": ["i1", "i2"],
            "rating": [1.0, 0.5],
            "timestamp": [1, 2],
            "source": ["click", "click"],
        }
    ).to_csv(path, index=False)


def test_get_config_loads_yaml(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("top_k: 4\n", encoding="utf-8")

    assert get_config(config) == {"top_k": 4}


def test_get_config_reports_unreadable_path(capsys):
    assert get_config("does-not-exist.yml") is None
    assert "Error:" in capsys.readouterr().out


def test_loader_reads_ratings_features_and_settings(tmp_path):
    ratings = tmp_path / "ratings.csv"
    users = tmp_path / "users.csv"
    items = tmp_path / "items.csv"
    write_ratings(ratings)
    pd.DataFrame({"userid": ["u1"], "age": [30]}).to_csv(users, index=False)
    pd.DataFrame({"itemid": ["i1"], "category": ["news"]}).to_csv(items, index=False)
    config = tmp_path / "config.yml"
    config.write_text(
        f"""datapaths:
  user_item_ratings: {ratings}
  user_features: {users}
  item_features: {items}
split_per: 0.25
top_k: 7
temporal: False
""",
        encoding="utf-8",
    )

    loader = DataLoader(config)

    assert loader.top_k == 7
    assert loader.split_per == 0.25
    assert loader.temporal is False
    assert list(loader.user_item_info["userid"]) == ["u1", "u2"]
    assert "source" in loader.user_item_info.columns
    assert loader.user_features_info["age"].tolist() == [30]
    assert loader.item_features_info["category"].tolist() == ["news"]


def test_loader_uses_defaults_when_optional_settings_are_omitted(tmp_path):
    ratings = tmp_path / "ratings.csv"
    write_ratings(ratings)
    config = tmp_path / "config.yml"
    config.write_text(
        f"""datapaths:
  user_item_ratings: {ratings}
  user_features:
  item_features:
""",
        encoding="utf-8",
    )

    loader = DataLoader(config)

    assert loader.top_k == 10
    assert loader.split_per == 0.2
    assert loader.temporal is True
    assert loader.user_features_info is None
    assert loader.item_features_info is None


def test_loader_requires_ratings_path(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text(
        """datapaths:
  user_item_ratings:
  user_features:
  item_features:
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="user_item_ratings"):
        DataLoader(config)


def test_splitter_temporal_cut_uses_timestamp_and_drops_extra_columns():
    original = ratings_frame()
    users = pd.DataFrame({"userid": ["u1"], "age": [30]})
    items = pd.DataFrame({"itemid": ["i1"], "category": ["news"]})

    split = Splitter(
        original,
        user_features=users,
        item_features=items,
        temporal=True,
        split_per=0.4,
    )

    assert list(split.data_train.columns) == COLUMNS
    assert list(split.data_train["itemid"]) == ["i5", "i4", "i3"]
    assert list(split.data_test["itemid"]) == ["i2", "i1"]
    assert len(split.data_train) + len(split.data_test) == len(original)
    assert list(original["itemid"]) == ["i1", "i2", "i3", "i4", "i5"]
    assert split.user_features is users
    assert split.item_features is items


def test_splitter_without_temporal_keeps_row_order():
    split = Splitter(ratings_frame(), temporal=False, split_per=0.4)

    assert list(split.data_train["itemid"]) == ["i1", "i2", "i3"]
    assert list(split.data_test["itemid"]) == ["i4", "i5"]


def test_splitter_requires_the_ratings_columns():
    ratings = ratings_frame().drop(columns=["rating"])

    with pytest.raises(ValueError, match="userid,itemid,rating,timestamp"):
        Splitter(ratings)


def test_splitter_requires_userid_on_user_features():
    with pytest.raises(ValueError, match="userid should exist"):
        Splitter(ratings_frame(), user_features=pd.DataFrame({"age": [30]}))


def test_splitter_requires_itemid_on_item_features():
    with pytest.raises(ValueError, match="itemid should exist"):
        Splitter(ratings_frame(), item_features=pd.DataFrame({"category": ["news"]}))

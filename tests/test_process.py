import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from newsrec.evaluation.evaluation import Evaluator
from newsrec.metadatahandler.datahandler import DataHandler
from newsrec.run import run_config

from tests.support import experiment_ratings


METRIC_COLUMNS = ["item_coverage", "user_coverage", "precision", "map", "ndcg"]


def write_experiment(tmp_path, models):
    ratings = tmp_path / "ratings.csv"
    experiment_ratings().to_csv(ratings, index=False)
    config = tmp_path / "config.yml"
    config.write_text(
        f"""datapaths:
  user_item_ratings: {ratings}
  user_features:
  item_features:

split_per: 0.2
top_k: 3
temporal: True

models:
{models}
""",
        encoding="utf-8",
    )
    return config


def results_table(tmp_path):
    tables = list((tmp_path / "metadata").glob("*.csv"))
    assert len(tables) == 1
    return pd.read_csv(tables[0]), tables[0]


def test_evaluator_scores_a_known_ranking():
    train = pd.DataFrame(
        {"userid": ["u1", "u1"], "itemid": ["a", "b"], "rating": [1.0, 1.0], "timestamp": [1, 2]}
    )
    test = pd.DataFrame(
        {"userid": ["u1", "u2"], "itemid": ["c", "f"], "rating": [1.0, 1.0], "timestamp": [3, 4]}
    )
    recos = pd.DataFrame({"userid": ["u1", "u1"], "itemid": ["d", "c"], "rank": [0, 1]})
    model = SimpleNamespace(
        recos=recos,
        data=SimpleNamespace(data_train=train, data_test=test),
    )

    evaluator = Evaluator(model)
    evaluator.calculate_metrics()
    metrics = evaluator.metrics()

    assert metrics["item_coverage"] == pytest.approx(1.0)
    assert metrics["user_coverage"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["map"] == pytest.approx(0.25)
    assert metrics["ndcg"] == pytest.approx(1 / np.log2(3))


def test_evaluator_ndcg_is_zero_when_there_are_no_recommendations():
    model = SimpleNamespace(
        recos=pd.DataFrame(columns=["userid", "itemid", "rank"]),
        data=SimpleNamespace(
            data_train=pd.DataFrame({"itemid": ["a"]}),
            data_test=pd.DataFrame({"userid": ["u1"], "itemid": ["b"]}),
        ),
    )

    evaluator = Evaluator(model)
    evaluator.calc_ndcg()

    assert evaluator.ndcg == 0.0


def test_results_table_orders_columns_and_stores_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = {
        "model": "Popular",
        "description": "Popular",
        "model_params": 0,
        "eval_metrics": {
            "item_coverage": np.float64(0.5),
            "user_coverage": np.float64(1.0),
            "precision": np.float64(0.25),
            "map": np.float64(0.125),
            "ndcg": np.float64(0.5),
        },
    }
    handler = DataHandler("config.yml", "metadata", datetime(2026, 9, 27, 12, 0, 0))
    handler.append_results(result)
    table = handler.write_results_table([result])

    lines = Path(handler.file).read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0]) == {"config_path": "config.yml"}
    assert json.loads(lines[1])["eval_metrics"]["precision"] == 0.25
    assert list(table.columns) == ["model", "description", "params", *METRIC_COLUMNS]
    assert table.iloc[0]["params"] == {}
    assert table.iloc[0]["precision"] == pytest.approx(0.25)


def test_run_expands_a_parameter_grid(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = write_experiment(
        tmp_path,
        """  Popular:
    parameters:
      ignored_a: [1, 2]
      ignored_b: [10, 20]
""",
    )

    run_config(str(config))

    table, _ = results_table(tmp_path)
    assert len(table) == 4
    assert set(table["model"]) == {"Popular"}
    assert table["description"].tolist() == ["Popular"] * 4


def test_run_requires_at_least_one_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ratings = tmp_path / "ratings.csv"
    experiment_ratings().to_csv(ratings, index=False)
    config = tmp_path / "config.yml"
    config.write_text(
        f"""datapaths:
  user_item_ratings: {ratings}
  user_features:
  item_features:
models:
""",
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="at least 1 valid model"):
        run_config(str(config))


def test_run_fits_every_model_and_writes_metrics(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = write_experiment(
        tmp_path,
        """  Popular:
    parameters:

  SVD:
    parameters:
      singular_values_n: [2]

  iALS:
    parameters:
      reg: [0.05]
      factors: [4]
      iter: [2]
      log: [False]

  BPR:
    parameters:
      factors: [4]
      learning_rate: [0.01]
      reg: [0.01]
      iter: [2]
      random_state: [42]

  ItemKNN:
    parameters:
      neighbors: [5]
      similarity: [cosine]

  EASE:
    parameters:
      lamb: [50]
""",
    )

    run_config(str(config))

    table, csv_path = results_table(tmp_path)
    assert table["model"].tolist() == ["Popular", "SVD", "iALS", "BPR", "ItemKNN", "EASE"]
    for column in METRIC_COLUMNS:
        assert table[column].between(0, 1).all()
    log_path = csv_path.with_suffix(".txt")
    logged = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert logged[0]["config_path"] == str(config)
    assert [row["model"] for row in logged[1:]] == table["model"].tolist()

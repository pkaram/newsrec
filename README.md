# newsrec

**work in progress**

This is a framework to experiment on recommendations related to news to users based on past behavior. An experiment can 
be set by defining a yaml file for the settings, algorithms and parameters to be tested.

Python 3.12 and dependencies are managed with [uv](https://docs.astral.sh/uv/). From the repo root:

```
uv sync
```

Then run scripts with `uv run python ...`. The yaml file has the following structure


### Yaml File Structure

```
datapaths:
  user_item_ratings: file1.csv #includes userid, itemid, rating, timestamp. Minimum required dataset.
  user_features: file2.csv #additional user related data to be used from a model.
  item_features: file3.csv #additional item related data to be used from a model.

split_per: 0.2 #splitting ratio for train/test dataset
top_k: 10 #top n recommendations to be considered for evaluation
temporal: True #temporal split of data parameter

#Models listed below are currently supported
models:

  Popular:
    parameters:

  SVD: 
    parameters:
      singular_values_n: [50]

  iALS:
    parameters:
      reg: [0.05]
      factors: [50]
      iter: [2,5]
      log: [True]
```


Mimimum requirements for an experiment to run is the existence of a csv file which includes at least the following 4 
columns:
* userid (string or int), id of a user
* itemid (string or int), id of an item
* rating (float), value that indicates preference of user to an item
* timestamp (int or float), value that corresponds to timestamp

Order of columns is not important and additional columns will be omitted.

After having defined the *yml_files/your_configuration.yml* you can run your experiment by creating a 
*run_experiment.py* file in your repo which includes the following:
```
from newsrec.run import run_config
run_config('yml_files/your_configuration.yml')
```
and then simply run:
```
uv run python run_experiment.py
```

The run will print results for all models and parameter combinations, then a summary table.
It also saves JSON lines and a CSV table in the *metadata* directory, which is created after the first experiment.

## Example

`mind_prepare_dataset.py` downloads MIND-small train if `datasets/MIND/behaviors.tsv` is missing, then writes `datasets/MIND/data/ratings.csv`. You can also place the Kaggle [MIND](https://www.kaggle.com/arashnic/mind-news-dataset) files in `datasets/MIND` yourself.

```
uv run python mind_prepare_dataset.py
```

This will create *datasets/MIND/data/ratings.csv* file which can be used to run *yml_files/example_mind.yml* experiment configuration:

```
uv run python run_example_mind.py
```

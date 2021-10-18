import sys
from datetime import datetime
from itertools import product
from pprint import pprint

from .dataloader import dataloader
from .datasplitting import splitter
from .object_mappings import model_mappings
from .evaluation import evaluation
from .metadatahandler import datahandler


def run_config(config_path=None):
    dtn = datetime.now()
    print(f"{dtn.strftime('%Y-%m-%d %H:%M:%S')}: Run started")
    mthandler = datahandler.DataHandler(path=config_path, folder='metadata', run_datetime=dtn)

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Data Loading")
    loader = dataloader.DataLoader(config_path)
    top_k = loader.top_k

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Data Splitting")

    data = splitter.Splitter(data=loader.user_item_info,
                             user_features=loader.user_features_info,
                             item_features=loader.item_features_info,
                             temporal=loader.temporal,
                             split_per=loader.split_per)

    models_to_check = loader.path.get('models', None)
    if not models_to_check:
        sys.exit('Provide at least 1 valid model name to run')

    models_list = list(models_to_check)
    results = []

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Running Models")
    for m in models_list:
        print("======================================================")
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Run for model {m} started")
        model = model_mappings.get(m)
        model_params = loader.path.get('models').get(m).get('parameters', None)

        if model_params:
            params_combinations = [dict(zip(model_params, v)) for v in product(*model_params.values())]
        else:
            params_combinations = [0]

        for params_combi in params_combinations:
            print("======================================================")
            model.pass_parameters(params_combi)

            model.train_model(data=data, k=top_k)
            evaluator = evaluation.Evaluator(model)
            evaluator.calculate_metrics()

            results_temp = {'model': m,
                            'description': model.description,
                            'model_params': params_combi,
                            'eval_metrics': evaluator.metrics()
                        }
            pprint(results_temp, width=1)
            mthandler.append_results(metadata=results_temp)
            results.append(results_temp)

        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Run for model {m} completed")

    #print("=====================RESULTS==========================")
    #print(results)
    print("======================================================")
    print(f"Results can be found in: {mthandler.file}")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: Run  completed")


if __name__ == '__main__':
    run_config(config_path='path to yml file')

import os
import json

import pandas as pd


class DataHandler:
    def __init__(self, path, folder, run_datetime):
        self.folder = folder or 'metadata'
        self.path = path
        self.filename = run_datetime.strftime('%Y%m%d_%H_%M_%S')
        self.file = None
        self.table_file = None
        self.recos = None
        self.create_folder()
        self.create_txt_file()

    def append_results(self, metadata):
        with open(self.file, 'a') as myFile:
            myFile.write('\n')
            myFile.write(json.dumps(metadata))

    def write_results_table(self, results):
        rows = []
        for result in results:
            row = {
                'model': result.get('model'),
                'description': result.get('description'),
            }
            params = result.get('model_params')
            if isinstance(params, dict):
                row.update(params)
            metrics = result.get('eval_metrics') or {}
            for name, value in metrics.items():
                row[name] = float(value) if hasattr(value, 'item') else value
            rows.append(row)

        df = pd.DataFrame(rows)
        front = ['model', 'description']
        metric_cols = ['item_coverage', 'user_coverage', 'precision', 'map']
        other = [c for c in df.columns if c not in front + metric_cols]
        ordered = front + other + [c for c in metric_cols if c in df.columns]
        df = df[ordered]

        self.table_file = os.path.splitext(self.file)[0] + '.csv'
        df.to_csv(self.table_file, index=False)
        return df

    def write_recos(self):
        pass

    def create_folder(self):
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def create_txt_file(self):
        self.file = os.getcwd() + '/' + self.folder + '/' + self.filename + '.txt'
        if not os.path.exists(self.file):
            with open(self.file, 'w') as myFile:
                myFile.write(json.dumps({'config_path':self.path}))

import os
import json


class DataHandler:
    def __init__(self, path, folder, run_datetime):
        self.folder = folder or 'metadata'
        self.path = path
        self.filename = run_datetime.strftime('%Y%m%d_%H_%M_%S')
        self.file = None
        self.recos = None
        self.create_folder()
        self.create_txt_file()

    def append_results(self, metadata):
        with open(self.file, 'a') as myFile:
            myFile.write('\n')
            myFile.write(json.dumps(metadata))

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

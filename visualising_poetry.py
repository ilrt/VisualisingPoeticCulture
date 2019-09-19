import glob
import pandas as pd
import settings as settings
from zipfile import ZipFile
import shutil
import os
import urllib
import private_settings as private


class VisualisingPoetrySetup:

    def __init__(self):
        pass

    def clean(self, target):
        """ Clean a target directory of files and sub directories """
        if os.path.exists(target):
            shutil.rmtree(target)
        os.makedirs(target)

    def clean_source(self):
        """ Clean the source directory of Excel files """
        self.clean(settings.sources['excel'])

    def clean_preprocessed(self):
        """ Clean the preprocessed directory """
        self.clean(settings.sources['pickle'])

    def download_data(self):
        """ Get the zip file specified in the private settings """
        urllib.request.urlretrieve(private.database_zip_url, settings.sources['excel'] + 'database.zip')

    def unpack_zip(self):
        """ Extract a zip file of Excel files """
        for file in glob.glob(settings.sources['excel'] + '*.zip', recursive=False):
            with ZipFile(file) as zip_object:
                for item in zip_object.namelist():
                    if item.endswith('.xlsx'):
                        zip_object.extract(item, path=settings.sources['excel'])

    def write_pickle_data_frames(self):
        """ Create a data frame and pickle of the Excel sheet with poem data """
        for file in glob.glob(settings.sources['excel'] + '**/*.xlsx', recursive=True):
            filename = self.name_from_full_path(file)
            df = pd.read_excel(file, sheet_name='poem data')
            df.to_pickle(settings.sources['pickle'] + filename + '.pickle')

    def name_from_full_path(self, full_path):
        """ Get the name of the paper from the path / filename """
        filename = full_path.split("/")[-1]
        return filename.replace('.xlsx', '')

    def setup(self):
        self.clean_source()
        self.clean_preprocessed()
        self.download_data()
        self.unpack_zip()
        self.write_pickle_data_frames()

    def setup_if_needed(self):
        if not os.path.exists(settings.sources['excel']) or not os.path.exists(settings.sources['excel'] +
                                                                               'database.zip'):
            print("No zip file. Getting data.")
            self.setup()
            print("Setup complete")


class VisualisingPoetryQuery:

    def pickle_as_single_data_frame(self):
        """ Load all the pickle files as a single data frame """
        files = glob.glob(settings.sources['pickle'] + '*.pickle', recursive=True)
        results = []
        for file in files:
            results.append(pd.read_pickle(file))

        return pd.concat(results, sort=False)

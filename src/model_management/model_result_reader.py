import os
import pandas as pd
from os import listdir
from os.path import isfile, join
from src.utils.general_utility_functions import from_string_to_dict


def read_folder_metadata(path):
    """
    Read all the metadata.zip file in a given folder.
    They are result of hp tuning.
    :param path: path of the folder
    :return: list of dataframes containing information in the metadata.zip files in the folder
    """
    metadata_file_list = read_metadata_file_list(path)
    metadata_content_list = []
    for f in metadata_file_list:
        metadata_file_path = path + f
        metadata_content_list.append(read_tuning_metadata_file(metadata_file_path))
    return metadata_content_list


def best_model_reader(path):
    """
    Read all the files in the given directory folder (assumed to terminate with a '/' and then
    return the list of best model in all the files.

    It basically read the last line of the report files generated after running hyper-parameter tuning code.

    :param path: directory where .txt of report files of hp. tuning are stored
    :return: list of dictionary that contains the hp of the best model found. There will be 1 dictionary for each file
    """
    only_files = [f for f in listdir(path) if isfile(join(path, f))]

    # Filter zip files
    only_files = [f for f in only_files if f[-3:] != 'zip']

    best_model_list = []
    for file in only_files:
        f = open(path + file, "r")
        lines = f.read().splitlines()
        last_line = lines[-1]
        param_dict = from_string_to_dict(last_line.split('{')[1][:-1])
        best_model_list.append(param_dict)

    return best_model_list


def read_metadata_file_list(path):
    """
    Read the list of metadata files in a given directory
    :param path: directory path
    :return: list of file names of the metadata files in the given directory
    """
    files = [f for f in listdir(path) if isfile(join(path, f))]

    metadata_files = [f for f in files if f[-12:] == 'metadata.zip']

    return metadata_files


def read_tuning_metadata_file(zip_file_path: os.path):
    """
    Return pandas DataFrame containing all the general information contained in the
    zip file regarding a hyperparameter tuning result

    :param zip_file_path: relative path of the zip file
    :return: DataFrame containing all the general information of the zip file
    """
    import zipfile
    import json
    import shutil

    print("...Loading metadata")

    data_file = None
    try:
        print(zip_file_path)
        data_file = zipfile.ZipFile(zip_file_path)
    except (FileNotFoundError, zipfile.BadZipFile):
        print("Unable to find data zip file!")

    recommender_name_path = data_file.extract("algorithm_name_recommender.json", path=zip_file_path + "decompressed/")
    hyperparameters_list_path = data_file.extract("hyperparameters_list.json", path=zip_file_path + "decompressed/")
    result_on_validation_path = data_file.extract("result_on_validation_list.json",
                                                  path=zip_file_path + "decompressed/")
    time_on_train_list_path = data_file.extract("time_on_train_list.json", path=zip_file_path + "decompressed/")
    time_on_validation_list_path = data_file.extract("time_on_validation_list.json",
                                                     path=zip_file_path + "decompressed/")

    with open(recommender_name_path, "r") as file:
        recommender_name = json.load(file)

    with open(hyperparameters_list_path, "r") as file:
        hyperparameters_list = json.load(file)

    with open(result_on_validation_path, "r") as file:
        result_on_validation = json.load(file)

    with open(time_on_train_list_path, "r") as file:
        time_on_train_list = json.load(file)

    with open(time_on_validation_list_path, "r") as file:
        time_on_validation_list = json.load(file)

    print("...Loading recommender: {}".format(recommender_name))

    print("...Clean temporary files")

    data_file.close()
    shutil.rmtree(zip_file_path + "decompressed", ignore_errors=True)

    print("...Generating pandas DataFrame")

    hyperparameters_df = pd.DataFrame(hyperparameters_list)
    result_on_validation_df = pd.DataFrame(result_on_validation)
    time_on_train_df = pd.DataFrame(time_on_train_list, columns=['TrainingElapsedTime'])
    time_on_validation_df = pd.DataFrame(time_on_validation_list, columns=['ValidationElapsedTime'])

    df: pd.DataFrame = pd.concat([hyperparameters_df, result_on_validation_df,
                                  time_on_train_df, time_on_validation_df], axis=1)

    print("Loading complete!")

    return df

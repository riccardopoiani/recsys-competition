import os

from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.utils.general_utility_functions import get_project_root_path


def set_env_variables():
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def read_split_load_data(k_out, allow_cold_users, seed):
    root_data_path = os.path.join(get_project_root_path(), "data/")
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=k_out, use_validation_set=False,
                                               allow_cold_users=allow_cold_users,
                                               force_new_split=True, seed=seed)
    data_reader.load_data()

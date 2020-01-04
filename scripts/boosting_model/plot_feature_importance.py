from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.model.Ensemble.Boosting.LightGBMRecommender import LightGBMRecommender
from src.model.Ensemble.Boosting.boosting_preprocessing import preprocess_dataframe_after_reading, get_label_array
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from src.utils.general_utility_functions import get_split_seed

if __name__ == '__main__':
    model_path = "../../report/hp_tuning/boosting/Jan01_17-53-13_k_out_value_3_eval/best_model_44.bin"
    model = lgb.Booster(model_file=model_path)

    #fig, ax = plt.subplots(figsize=(10, 20))
    lgb.plot_importance(model)
    fig_t = plt.gcf()
    fig_t.show()

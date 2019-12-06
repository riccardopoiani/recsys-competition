
import xlearn as xl
import os

from src.utils.general_utility_functions import get_project_root_path

if __name__ == '__main__':
    root = get_project_root_path()
    fm_data_path = os.path.join(root, "resources", "fm_data")
    model = xl.create_fm()
    #model.setTXTModel("./model.txt")
    model.setTrain(fm_data_path + "/train_compressed.txt")

    param = {'task': 'binary', 'lr': 0.1, 'k': 200, 'lambda': 0.0001, 'epoch': 100, 'opt': 'adagrad', 'metric': 'acc'}
    model.fit(param, "./model.out")

    model.setSigmoid()
    model.setTest(fm_data_path + "/train.txt")
    model.predict("./model.out", "./output.txt")

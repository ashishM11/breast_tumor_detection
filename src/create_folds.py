import os
import pandas as pd
import config as cnf
from sklearn import model_selection


def generate_folds(DATA_FILE_LOCATION):
    """ This method will generate tain data folds for cross validation.

    Args:
        DATA_FILE_LOCATION (String): Will fetch it from config file.
    """
    df = pd.read_csv(DATA_FILE_LOCATION)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    K_fold = model_selection.StratifiedKFold(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(K_fold.split(X=df,y=df['diagnosis'].values)):
        print(len(train_idx),"\t",len(val_idx))
        df.loc[val_idx,"kfold"] = fold

    os.chdir(os.getcwd())
    df.to_csv(cnf.TRAINING_FILE,index=False) 

if __name__ == "__main__":
    generate_folds(cnf.DATA_FILE_LOCATION)

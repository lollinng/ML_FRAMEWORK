import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    df = pd.read_csv('input/train.csv')
    df["kfold"] = -1  # adding k fold column to categorize the kfold later

    # shuffling the data
    df = df.sample(frac=1).reset_index(drop=True)
    kf = model_selection.StratifiedKFold(n_splits=5,shuffle=False)
    kf_gen = kf.split(X=df,y=df.target.values)

    for fold, (train_idx,val_idx) in enumerate(kf_gen):
        print(len(train_idx,),len(val_idx))
        # here kfold column and val_idx is value to be written as kfold number
        df.loc[val_idx,'kfold'] = fold # add kfold to kfold column

    df.to_csv('input/train_folds.csv',index=False)
 
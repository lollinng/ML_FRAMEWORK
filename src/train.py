import os
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
import joblib

from . import dispatcher

TRAINING_DATA = os.environ.get("TRAINING_DATA")
TEST_DATA = os.environ.get("TEST_DATA")
FOLD = int(os.environ.get("FOLD"))
MODEL = os.environ.get("MODEL")


FOLD_MAPPING = {
    0 : [1,2,3,4],
    1 : [0,2,3,4],
    2 : [0,1,3,4],
    3 : [0,1,2,4],
    4 : [0,1,2,3]
}



if __name__ == "__main__":
    
    # splitting db
    df = pd.read_csv(TRAINING_DATA)
    test_df = pd.read_csv(TEST_DATA)
    train_df = df[df.kfold.isin(FOLD_MAPPING)].reset_index(drop=True)
    valid_df = df[df.kfold==FOLD].reset_index(drop=True)

    ytrain = train_df.target.values
    yvalid = valid_df.target.values

    

    train_df = train_df.drop(['id','target','kfold'],axis=1)
    valid_df = valid_df.drop(['id','target','kfold'],axis=1)

    valid_df = valid_df[train_df.columns]


    # LABEL ENCODING EVERY ROW
    label_encoders = {}
    for c in train_df.columns:
        lbl = preprocessing.LabelEncoder()

        train_df.loc[:, c] = train_df.loc[:, c].astype(str).fillna("NONE")
        valid_df.loc[:, c] = valid_df.loc[:, c].astype(str).fillna("NONE")
        test_df.loc[:, c] = test_df.loc[:, c].astype(str).fillna("NONE")

        train_col_corpus = train_df[c].values.tolist()
        val_col_corpus = valid_df[c].values.tolist()
        test_col_corpus = test_df[c].values.tolist()
        col_corpus = train_col_corpus + val_col_corpus + test_col_corpus
        lbl.fit(col_corpus)

        # replacing all data in c column 
        train_df.loc[:,c] = lbl.transform(train_col_corpus)
        valid_df.loc[:,c] = lbl.transform(val_col_corpus)

        label_encoders[c] = lbl      # saving lbl object with column name

    # Training

    # RANDOM FOREST
    clf = dispatcher.MODELS[MODEL]
    clf.fit(train_df,ytrain)
    preds = clf.predict_proba(valid_df)[:,1]
    
    print(metrics.roc_auc_score(yvalid,preds))        # roc_auc used for sparse matrix

    joblib.dump(label_encoders,f'models/{MODEL}_{FOLD}_label_encoder.pkl')
    joblib.dump(clf,f'models/{MODEL}_{FOLD}_.pkl')
    joblib.dump(train_df.columns,f'models/{MODEL}_{FOLD}_cols.pkl')
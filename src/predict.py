import os
import numpy as np
import pandas as pd
import joblib

from . import dispatcher


def predict(test_data_path,model_type,model_path):
    # splitting db
    df = pd.read_csv(test_data_path)
    test_idx = df['id'].values
    predictions = None
    

    for FOLD in range(5):
        print(FOLD)
        df = pd.read_csv(test_data_path)    # since df is getting updated in label encoding
        encoders = joblib.load(os.path.join('models',f'{model_type}_{FOLD}_label_encoder.pkl'))
        cols = joblib.load(os.path.join('models',f'{model_type}_{FOLD}_cols.pkl'))
        for c in encoders:
            lbl = encoders[c]
            df.loc[:, c] = df.loc[:, c].astype(str).fillna("NONE")
            test_col_corpus = df[c].values.tolist()
            df.loc[:, c] = lbl.transform(test_col_corpus) # label encoding test df features
        
        # importing model and cols idk why
        clf = joblib.load(os.path.join(f'models/{model_type}_{FOLD}_.pkl'))
        
        print(cols)
        df = df[cols]
        preds = clf.predict_proba(df)[:,1]
 
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    
    predictions /=5

    sum = pd.DataFrame(np.column_stack((test_idx,predictions)),columns=['id','target'])
    return sum

if __name__ == "__main__":
    submission = predict(
        test_data_path="input/test.csv", 
        model_type="randomforest", 
        model_path="models/"
    )
    submission.loc[:, "id"] = submission.loc[:, "id"].astype(int)
    submission.to_csv(f"models/rf_submission.csv", index=False)
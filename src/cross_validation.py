
"""
- binary classification
- multi class classification             #  classes are mutually exclusive
- multi label classification             #  each label represents a different classification   . For eg- red_male,white_male,red_female,white_female
- single column regression               # each label maybe related to each other but we have to select one . For eg - red,white;female,male, so o/p for red_female is (0,0)
- multi column regression 
- holdout taste problem                  # for time-series data (to avoid future data) and for large data 

"""


import pandas as pd
from sklearn import model_selection


class CrossValidation:

    def __init__(
            self,
            df,
            target_cols,
            shuffle,
            problem_type="binary_classification",
            num_folds = 5,
            random_state = 42,
            multi_label_delimiter = ','
        ):
        self.df = df
        self.target_cols = target_cols
        self.num_targets = len(target_cols)
        self.problem_type = problem_type
        self.num_folds = num_folds
        self.shuffle = shuffle
        self.random_state = random_state
        self.multi_label_delimiter = multi_label_delimiter

        if self.shuffle == True:
            self.df = df.sample(frac=1).reset_index(drop=True)

        self.df["kfold"] = -1

    def split(self):



        if self.problem_type in ("binary_classification","multiclass_classification"):
            if self.num_targets!=1 :
                raise Exception("Invalid number of targets for this problem type")
            target = self.target_cols[0]
            unique_values = self.df[target].nunique()  # number of classes            
            if unique_values == 1:
                raise Exception("only one unique value found")
            elif unique_values > 1 :
                kf =  model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                    shuffle=self.shuffle)
                kf_gen = kf.split(X=self.df,y = self.df[target].values)
                for fold, (train_idx,val_idx) in enumerate(kf_gen):
                    self.df.loc[val_idx,'kfold'] = fold
                 

        elif self.problem_type in ('single_col_regression','multi_col_regression'):
            target = self.target_cols[0]
            if self.num_targets!=1 and self.problem_type=='single_col_regression':
                raise Exception("Invalid number of targets for this problem type")
            if self.num_targets<2 and self.problem_type=='multi_col_regression':
                raise Exception("Invalid number of targets for this problem type")
            kf = model_selection.KFold(n_splits=self.num_folds)
            kf_gen = kf.split(X=self.df)
            for fold, (train_idx,val_idx) in enumerate(kf_gen):
                self.df.loc[val_idx,'kfold'] = fold


        elif self.problem_type.startswith('holdout_'):
            # holdout percentage - 10% holdout means 90% 0 and 10% 1 in kfold
            holdout_percentage = int(self.problem_type.split("_")[1])
            n = len(self.df)
            num_holdout_sample = int( n* holdout_percentage/100)
            self.df.loc[:n-num_holdout_sample,"kfold"] = 0
            self.df.loc[n-num_holdout_sample:,"kfold"]= 1


        elif self.problem_type == "multi_label_classification":
            if self.num_targets !=1:
                raise Exception("Invalid number of targets for this problem type")

            target_col = self.df[self.target_cols[0]]
            # converting (0,1,0) to  3 using delimiter=',',this 3 is taken as class for creating kfold
            targets = target_col.apply(lambda x: len(str(x).split(self.multi_label_delimiter)))

            kf =  model_selection.StratifiedKFold(n_splits=self.num_folds,
                                                    shuffle=self.shuffle)
            kf_gen = kf.split(X=self.df,y = targets)
            for fold, (train_idx,val_idx) in enumerate(kf_gen):
                self.df.loc[val_idx,'kfold'] = fold


        else:
            raise Exception("problem type not understood")

        return self.df

if __name__ == "__main__":

    # path = 'input/train.csv'
    # path = 'input/train_reg.csv'
    path = 'input/train_multi_label.csv'
    df = pd.read_csv(path)
    # cv = CrossValidation(df,target_cols=['target'],problem_type='binary_classification')
    # cv = CrossValidation(df,target_cols=['target'],problem_type='holdout_10')  # 10% holdout means 90% 0 and 10% 1 in kfold
    # cv = CrossValidation(df,target_cols=['SalePrice'],problem_type='single_col_regression')
    cv = CrossValidation(
            df,target_cols=['attribute_ids'],
            shuffle=True,problem_type='multi_label_classification',multi_label_delimiter=" ")
    df_split = cv.split()
    print(df_split.head())
    print(df_split.kfold.value_counts())  # ALL 5 K FOLDS are equal values
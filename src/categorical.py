from sklearn import preprocessing


class CategoricalFeatures:
    def __init__(self,df,categorical_feature,encoding_type,handle_na=False):
        """
        """
        self.df = df
        self.cat_feats = categorical_feature
        self.encoding_type = encoding_type
        self,handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        # fill na values as categorical value to prevent error in encoders
        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:,c] = self.df.loc[:,c].fillna('-99999')

        self.output_df = self.df.copy(deep=True)

        
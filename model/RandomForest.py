from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


class RandomForest():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.rf = RandomForestRegressor()

    def forward(self):
        self.rf.fit(self.X_train, self.y_train)

    def predict(self, X_valid, Y_valid):
        x_pred = self.rf.predict(X_valid)
        mse = metrics.mean_squared_error(x_pred, Y_valid)
        rmse = mse**0.5
        return rmse


class MLPR():
    def __init__(self, X_train, y_train):
        # Construct the pipeline with a standard scaler and a small neural network
        self.estimators = []
        self.estimators.append(('standardize', StandardScaler()))
        self.estimators.append(
            ('nn', MLPRegressor(hidden_layer_sizes=(15,), max_iter=1000)))
        self.model = Pipeline(self.estimators)
        self.X_train = X_train
        self.y_train = y_train

    def forward(self):
        # We'll use 5-fold cross validation. That is, a random 80% of the data will be used
        # to train the model, and the prediction score will be computed on the remaining 20%.
        # This process is repeated five times such that the training sets in each "fold"
        # are mutually orthogonal.
        self.kfold = KFold(n_splits=5)
        results = cross_val_score(
            self.model, self.X_train, self.y_train, cv=self.kfold, scoring='neg_mean_squared_error')
        return results

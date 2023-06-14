from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import metrics


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

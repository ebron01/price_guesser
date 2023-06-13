from sklearn.ensemble import RandomForestClassifier

class RandomForest():
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.rf = RandomForestClassifier(max_depth = 5, n_estimators=100, random_state = 42)
        
    def forward(self):
        return self.rf.fit(self.X_train, self.y_train)
    
     
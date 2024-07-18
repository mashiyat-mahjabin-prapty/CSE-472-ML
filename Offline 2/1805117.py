import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def telco_preprocess():
    df = pd.read_csv(
    'F:\\4-2\CSE 472\Offline 2\Dataset\WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # check for unique values in each categorical variable
    for col in df.columns:
        if df[col].dtype == 'object':
            print(col, ':', df[col].nunique())
    # check if any of the entries of TotalCharges are not float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].isnull().sum()

    # replace the null values with the mean of TotalCharges
    df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)

    # check if any of the entries of TotalCharges are not float
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].isnull().sum()

    X = df.iloc[:, 1:-1]
    y = df.iloc[:, -1]

    # apply transformation to the target variable
    y = y.map({'Yes': 1, 'No': 0})

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8,
                                                    random_state=1,
                                                    stratify=y)
    
    scaler = StandardScaler()

    num_X = X_train.select_dtypes(include=['int64', 'float64'])

    X_train[num_X.columns] = scaler.fit_transform(num_X)

    X_train = pd.get_dummies(X_train, drop_first=True, prefix_sep='_')
    
    # apply the same transformation to the test set
    num_X = X_test.select_dtypes(include=['int64', 'float64'])

    X_test[num_X.columns] = scaler.transform(num_X)

    X_test = pd.get_dummies(X_test, drop_first=True, prefix_sep='_')

    return X_train, y_train, X_test, y_test



def credit_preprocess():
    # Read the data
    data = pd.read_csv('F:\\4-2\CSE 472\Offline 2\Dataset\creditcard.csv\creditcard.csv')

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=0,
                                                    stratify=y)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = pd.DataFrame(X_scaled, columns=X_train.columns.values)
    X_scaled.head()

    # sample from the dataset
    X_train_0 = X_train[y_train == 0]
    X_train_1 = X_train[y_train == 1]

    X_train_0 = X_train.sample(10000, random_state=1)

    X_train = pd.concat([X_train_0, X_train_1])
    y_train = y_train[X_train.index]

    X_test_0 = X_test[y_test == 0]
    X_test_1 = X_test[y_test == 1]

    X_test_0 = X_test.sample(10000, random_state=1)

    X_test = pd.concat([X_test_0, X_test_1])
    y_test = y_test[X_test.index]

    return X_train, y_train, X_test, y_test

def adult_preprocess():
    data = pd.read_csv('F:\\4-2\CSE 472\Offline 2\Dataset\\adult\\adult.data', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 
                             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])

    # find if there are any '?' in the data
    for col in data.columns:
        if data[col].dtype == object:
            print(col, data[col][data[col] == ' ?'].count())
    
    # find the number of rows having ' ?' in both workclass and occupation
    data[(data['workclass'] == ' ?') & (data['occupation'] == ' ?')].shape[0]

    # drop the rows having ' ?' in both workclass and occupation
    data = data[~((data['workclass'] == ' ?') & (data['occupation'] == ' ?'))]
    
    # replace '?' with 'Unknown' in occupation and native-country columns
    data['occupation'] = data['occupation'].replace(' ?', ' Unknown_occupation')
    data['native-country'] = data['native-country'].replace(' ?', ' Unknown_native-country')

    X_train = data.drop('income', axis=1)
    y_train = data['income']

    scaler = StandardScaler()

    num_X = X_train.select_dtypes(include=['int64', 'float64'])

    X_train[num_X.columns] = scaler.fit_transform(num_X)

    X_train = pd.get_dummies(X_train, drop_first=True, prefix_sep='_')

    # read the test data
    test_data = pd.read_csv('F:\\4-2\CSE 472\Offline 2\Dataset\\adult\\adult.test', names=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 
                             'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'])
    test_data.drop(0, axis=0, inplace=True)

    # check for missing values
    print(test_data.isnull().sum())

    # check data types
    print(test_data.dtypes)

    # find if there are any '?' in the data
    for col in test_data.columns:
        if test_data[col].dtype == object:
            print(col, test_data[col][test_data[col] == ' ?'].count())

    # find the number of rows having ' ?' in both workclass and occupation
    test_data[(test_data['workclass'] == ' ?') & (test_data['occupation'] == ' ?')].shape[0]

    # drop the rows having ' ?' in both workclass and occupation
    test_data = test_data[~((test_data['workclass'] == ' ?') & (test_data['occupation'] == ' ?'))]

    # print the value count of workclass, occupation, native-country columns
    print(test_data['workclass'].value_counts())
    print(test_data['occupation'].value_counts())
    print(test_data['native-country'].value_counts())

    # replace '?' with 'Unknown' in occupation and native-country columns
    test_data['occupation'] = test_data['occupation'].replace(' ?', ' Unknown_occupation')
    test_data['native-country'] = test_data['native-country'].replace(' ?', ' Unknown_native-country')

    # print the value count of workclass, occupation, native-country columns
    print(test_data['workclass'].value_counts())
    print(test_data['occupation'].value_counts())
    print(test_data['native-country'].value_counts())

    # force the data type int to age column
    test_data['age'] = test_data['age'].astype('int64')

    # see if there is any null value in the data
    test_data.isnull().sum()

    X_test = test_data.drop('income', axis=1)
    y_test = test_data['income']

    X_test.head()

    # replace <=50K and >50K with 0 and 1 in y_train and y_test
    y_train = y_train.replace(' <=50K', 0)
    y_train = y_train.replace(' >50K', 1)

    y_test = y_test.replace(' <=50K.', 0)
    y_test = y_test.replace(' >50K.', 1)

    # apply the same transformation to the test set

    num_X = X_test.select_dtypes(include=['int64', 'float64'])

    X_test[num_X.columns] = scaler.transform(num_X)

    X_test = pd.get_dummies(X_test, drop_first=True, prefix_sep='_')

    return X_train, y_train, X_test, y_test

class LogisticRegression():
    def __init__(self, learning_rate=0.01, num_iterations=1000, error_threshold=0.5):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.error_threshold = error_threshold
        self.theta = None
        self.b = None
    
    def entropy(self, y):
        cnt = y.value_counts(normalize=True)
        return -np.sum(cnt * np.log2(cnt))

    def information_gain(self, X, y, feature):
        # calculate information gain of X[feature]
        target_entropy = self.entropy(y)

        feature_entropy = []
        weight = []

        for values in X[feature].unique():
            weight.append(len(X[X[feature] == values]) / len(X))
            feature_entropy.append(self.entropy(X[X[feature] == values]))
        
        remainder = np.sum(np.array(feature_entropy) * np.array(weight))

        gain = target_entropy - remainder
        print("Gain of " + feature + ":", gain)

        return gain

    def k_best_features(self, X, y, k):
        # return the names of the k best features
        features = X.columns.values
        gains = []

        for feature in features:
            gains.append([feature, self.information_gain(X, y, feature)])
        
        best_features = sorted(gains, key=lambda x: x[1], reverse=True)[:k]

        return [feature[0] for feature in best_features]

    
    def sigmoid(self, z):
        # print(z)
        return 1 / (1 + np.exp(-z))

    def logLoss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def gradient_dw(self):
        z = np.array(np.dot(self.X, self.theta) + self.b, dtype=np.float64)
        dw = np.dot(self.X.T, (self.sigmoid(z) - self.y)) / len(self.y)
        return dw

    def gradient_db(self):
        z = np.array(np.dot(self.X, self.theta) + self.b, dtype=np.float64)
        db = self.y - self.sigmoid(z)
        return np.sum(db) / len(self.y)

    def logistic_regression(self, X, y, num_features=None):
        self.X = X
        self.y = y
        if num_features is None:
            self.num_features = X.shape[1]
        elif num_features > 1:
            self.num_features = num_features
        else:
            self.num_features = int(X.shape[1] * num_features)
        
        # initialize theta with very small values
        self.theta = np.random.randn(self.num_features)
        self.b = 0
        
        # modify X to include only the top k features
        self.best_features = self.k_best_features(self.X, self.y, self.num_features)
        self.X = self.X[self.best_features]

        for iteration in range(self.num_iterations):
            # grad_w = self.gradient_dw()
            # grad_b = self.gradient_db()
            z = np.array(np.dot(self.X, self.theta) + self.b, dtype=np.float64) 
            y_pred = self.sigmoid(z)
            self.theta = self.theta - self.learning_rate * np.dot(self.X.T, (y_pred - self.y)) / len(self.y)
            self.b = self.b - self.learning_rate * np.sum(y_pred - self.y) / len(self.y)

            error = self.logLoss(self.y, y_pred)
            
            print("Iteration:", iteration, "Error:", error)
            if self.error_threshold > 0 and np.max(np.abs(error)) < self.error_threshold:
                break
        
        return self.theta, self.b, error
        
    def predict(self, X):
        # print('here1')
        X = X[self.best_features]
        # print('here2')
        # print(X)
        z = self.sigmoid(np.array(np.dot(X, self.theta) + self.b, dtype=np.float64))
        y_pred = (z >= 0.5).astype(int)
        return y_pred
    
    def accuracy(self, y_pred, y):
        return np.mean(y_pred == y)
    
    def recall(self, y_pred, y):
        tp = np.sum(y_pred * y)
        fn = np.sum(y * (1 - y_pred))
        return tp / (tp + fn)
    
    def precision(self, y_pred, y):
        tp = np.sum(y_pred * y)
        fp = np.sum((1 - y) * y_pred)
        return tp / (tp + fp)
    
    def f1_score(self, y_pred, y):
        recall = self.recall(y_pred, y)
        precision = self.precision(y_pred, y)
        return 2 * recall * precision / (recall + precision)
    
    def specificity(self, y_pred, y):
        tn = np.sum((1 - y_pred) * (1 - y))
        fp = np.sum((1 - y) * y_pred)
        return tn / (tn + fp)
    
    def false_discovery_rate(self, y_pred, y):
        tp = np.sum(y_pred * y)
        fp = np.sum((1 - y) * y_pred)
        return fp / (tp + fp)

class AdaBoost:
    def __init__(self, base_learner, n_estimators=10):
        self.base_learner = base_learner
        self.n_estimators = n_estimators
        self.weighted_sum = None
        self.weighted_b = None       

    def weighted_majority(self, hypotheses, z):
        weighted_sum = np.zeros(len(hypotheses[0][0]))
        weighted_b = 0

        for i in range(len(hypotheses)):
            weighted_sum = weighted_sum + z[i] * hypotheses[i][0]
            weighted_b = weighted_b + z[i] * hypotheses[i][1]
        
        return weighted_sum, weighted_b
        
    def fit(self, examples, y):
        # weakLearner: a function that takes examples and returns a hypothesis
        # K: the number of iterations
        # return: a list of hypotheses
        # concatenate X and y as a dataframe
        
        # initialize weights
        N = len(examples)
        self.weights = np.full(N, 1 / N)
        print(self.weights)

        self.hypotheses = []
        self.z = []

        for k in range(self.n_estimators):
            # resample data
            indices = np.random.choice(len(examples), len(examples), p=self.weights)

            X_t = examples.iloc[indices]
            y_t = y.iloc[indices]
            # train a weak learner
            h = self.base_learner.logistic_regression(X_t, y_t)
            print(h)
            # calculate error
            error = 0
            pred = self.base_learner.predict(X_t)
            print(pred)
            print(pred.shape, y_t.shape)
            # count the unequal predictions
            for p, yt, weight in zip(pred, y_t, self.weights):
                if p != yt:
                    error += weight
            print(error)

            
            print('error in the middle:', error)
            if error > 0.5:
                continue

            # update weights
            for i, (p, y_val) in enumerate(zip(pred, y)):
                if p == y_val:
                    self.weights[i] *= error / (1 - error)

            
            # normalize weights
            self.weights /= np.sum(self.weights)

            self.hypotheses.append(h)

            self.z.append(np.log((1 - error) / error))

            print("Iteration within adaboost:", k, "Error:", error)
        
        # return weighted majority of hypotheses and z
        return self.weighted_majority(self.hypotheses, self.z)


    def predict(self, X):
        # normalize X
        X = (X - X.mean()) / X.std()

        weighted_sum, weighted_b = self.weighted_majority(self.hypotheses, self.z)

        z = np.array(np.dot(X, weighted_sum) + weighted_b, dtype=np.float64)
        y_pred = self.base_learner.sigmoid(z)
        y_pred = [1 if y >= 0.5 else 0 for y in y_pred]

        return y_pred

    def accuracy(self, y_pred, y):
        return np.mean(y_pred == y)
    
    def recall(self, y_pred, y):
        tp = np.sum(np.array(y_pred) * np.array(y))
        fn = np.sum(np.array(y) * (1 - np.array(y_pred)))
        return tp / (tp + fn)
    
    def precision(self, y_pred, y):
        tp = np.sum(np.array(y_pred) * np.array(y))
        fp = np.sum((1 - np.array(y)) * np.array(y_pred))
        return tp / (tp + fp)
    
    def f1_score(self, y_pred, y):
        recall = self.recall(y_pred, y)
        precision = self.precision(y_pred, y)
        return 2 * recall * precision / (recall + precision)
    
    def specificity(self, y_pred, y):
        tn = np.sum((1 - np.array(y_pred)) * (1 - np.array(y)))
        fp = np.sum((1 - np.array(y)) * np.array(y_pred))
        return tn / (tn + fp)
    
    def false_discovery_rate(self, y_pred, y):
        tp = np.sum(np.array(y_pred) * np.array(y))
        fp = np.sum((1 - np.array(y)) * np.array(y_pred))
        return fp / (tp + fp)
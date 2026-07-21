import seaborn as sns
from sklearn.model_selection import train_test_split

class Dataset:
    def __init__(self, test_size=0.2, random_state=42):
        self._x_train = None
        self._y_train = None
        self._x_test = None
        self._y_test = None
        self.test_size = test_size
        self.random_state = random_state

    def load_data(self):
        iris = sns.load_dataset('iris')
        x = iris.drop('species', axis=1)
        y = iris['species']

        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
        return self  # Erlaubt "Method Chaining"

    def get_train_data(self):
        if self._x_train is None:
            self.load_data()
        return self._x_train, self._y_train

    def get_test_data(self):
        if self._x_test is None:
            self.load_data()
        return self._x_test, self._y_test

    def split_data(self, x, y):
        # splits data into training and testing sets
        self._x_train, self._x_test, self._y_train, self._y_test = train_test_split(
            x, y, test_size=self.test_size, random_state=self.random_state
        )
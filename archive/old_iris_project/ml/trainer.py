class Trainer:
    def __init__(self, model, dataset, metrics_list=None, epochs=1):
        self.model = model
        self.dataset = dataset
        """
        if metrics_list is not None:
            self.metrics_list = metrics_list
        else:
            self.metrics_list = []
        """
        self.metrics_list = metrics_list if metrics_list is not None else []
        self.epochs = epochs

    def train(self):
        x_train, y_train = self.dataset.get_train_data()
        # Delegation an das Modell
        self.model.fit(x_train, y_train)
        print("Training completed.")

    def evaluate(self):
        x_test, y_test = self.dataset.get_test_data()
        predictions = self.model.predict(x_test)

        results = {}
        for m in self.metrics_list:
            # Polymorphie: Jede Metrik berechnet sich selbst
            results[m.name] = m.calculate(y_test, predictions)

        return results

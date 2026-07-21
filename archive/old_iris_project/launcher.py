from ml.dataset import Dataset
from ml.models.model_factory import ModelFactory
from ml.trainer import Trainer
from ml.metrics.accuracy import Accuracy
from ml.metrics.confusion_matrix import ConfusionMatrix

"""
Training goal:
Blumen (Schwertlilien) anhand von 4 Merkmalen (Länge/Breite von Kelch- und Kronblättern)
in 3 verschiedene Arten zu klassifizieren.
"""

def get_model_choice():
    """User selects a model from available options."""
    available_models = [
        ("svm_linear", "SVM with a linear kernel – separates classes with a straight hyperplane."),
        ("svm_rbf", "SVM with RBF kernel – maps data into higher dimensions for non-linear separation."),
        ("svm_poly", "SVM with polynomial kernel – separates classes using polynomial decision boundaries."),
        ("svm_sigmoid", "SVM with sigmoid kernel – approximates a neural network-like decision boundary."),
        ("decision_tree", "Splits data recursively by feature thresholds to form a tree of decisions."),
        ("knn", "Classifies a sample by majority vote of its k nearest neighbors."),
        ("random_forest", "Ensemble of decision trees – combines predictions to reduce overfitting."),
        ("lda", "Projects data onto a lower-dimensional space that maximizes class separability."),
    ]

    print("\n--- Available Models ---")
    for i, (model_name, description) in enumerate(available_models, 1):
        print(f"{i}. {model_name}: {description}")

    while True:
        try:
            choice = int(input("\nSelect a model (1-8): "))
            if 1 <= choice <= len(available_models):
                return available_models[choice - 1]
            else:
                print(f"Invalid choice. Please enter 1-{len(available_models)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_hyperparameters():
    """User enters hyperparameters."""
    while True:
        try:
            test_size = float(input("\nEnter test size (0.0-1.0, default: 0.3): ") or "0.3")
            if 0.0 < test_size < 1.0:
                break
            else:
                print("Test size must be between 0.0 and 1.0.")
        except ValueError:
            print("Invalid input. Please enter a decimal number.")

    while True:
        try:
            epochs = int(input("Enter number of epochs (default: 1): ") or "1")
            if epochs > 0:
                break
            else:
                print("Epochs must be greater than 0.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    while True:
        try:
            random_state = int(input("Enter random state for reproducibility (default: 42): ") or "42")
            break
        except ValueError:
            print("Invalid input. Please enter an integer.")

    return test_size, epochs, random_state

def main():
    repeat = True
    while repeat:
        print("--- Starting Machine Learning Pipeline ---")

        # User inputs
        model_name = get_model_choice()
        test_size, epochs, random_state = get_hyperparameters()

        print(f"\n✓ Configuration:")
        print(f"  Model: {model_name}")
        print(f"  Test size: {test_size}")
        print(f"  Epochs: {epochs}")
        print(f"  Random state: {random_state}")

        # 1. Prepare data
        # Using the Dataset handler with the Iris dataset
        print("\n1. Loading data...")
        ds = Dataset(test_size=test_size, random_state=random_state)
        ds.load_data()
        print("✓ Data loaded successfully.")

        # 2. Get model from factory (strategy pattern)
        # You can easily switch between 'svm_linear', 'svm_rbf', etc.
        print("\n2. Creating model...")
        model = ModelFactory.get_model(model_name, input_shape=4, output_shape=3)
        print(f"✓ Model '{model_name}' created.")

        # 3. Define metrics
        # Using classes that implement the abstract base class
        print("\n3. Initializing metrics...")
        metrics = [Accuracy(), ConfusionMatrix()]
        print(f"✓ {len(metrics)} metrics initialized.")

        train(model, ds, metrics, epochs)

        # Ask user if they want to run another training session
        response = input("\n\nDo you want to run another training? (yes/no default: yes): ").strip().lower()
        repeat = response not in ['no', 'n']


def train(model, dataset, metrics, epochs):
    # 4. Initialize trainer (orchestration)
    # All components come together here
    print("\n4. Initializing trainer...")
    trainer = Trainer(model=model, dataset=dataset, metrics_list=metrics, epochs=epochs)

    # 5. Execute training
    print("5. Executing training...")
    trainer.train()

    predict(trainer)

def predict(trainer):
    # 6. Evaluation
    print("\n6. Performing evaluation...")
    results = trainer.evaluate()

    # 7. Display results
    print("\n--- Evaluation Results ---")
    for name, value in results.items():
        print(f"\n{name}:")
        print(value)
        print("-" * 40)

if __name__ == "__main__":
    main()

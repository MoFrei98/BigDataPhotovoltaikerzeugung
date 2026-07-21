import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

np.random.seed(123)
iris = sns.load_dataset('iris')
indices = np.random.permutation(len(iris))
test = iris.iloc[indices[:15]]
train = iris.iloc[indices[15:]]
features = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
target = 'species'
clf = DecisionTreeClassifier(random_state=123)
clf.fit(train[features], train[target])
y_pred = clf.predict(test[features])
error_rate = np.mean(y_pred != test[target])
print("Fehlklassifikationsrate:", error_rate)
cm = confusion_matrix(test[target], y_pred)
labels = sorted(set(test[target]))
cm_df = pd.DataFrame(
    cm,
    index=labels,      # Reference (wahre Klassen)
    columns=labels     # Prediction (vorhergesagte Klassen)
)
print(cm_df)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm_df,
    annot=True,              # Zahlen anzeigen
    fmt="d",
    cmap="Blues",            # ähnlich zu white → skyblue
    cbar=True,
    linewidths=0.5,
    linecolor="white"
)
plt.title("Konfusionsmatrix: ")
plt.xlabel("Vorhergesagte Klasse")
plt.ylabel("Wahre Klasse")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
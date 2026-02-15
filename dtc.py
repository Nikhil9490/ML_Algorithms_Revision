#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
# %%

df = load_iris()
df
# %%

X = pd.DataFrame(df['data'], columns = ['sepal length (cm)', 'petal width (cm)', 'sepal width (cm)', 'petal length (cm)'])
y = pd.DataFrame(df['target'], columns = ['target'])
# %%
X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)
# %%
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)
# %%)
y_pred = dtc.predict(X_test)
accuracy_score = accuracy_score(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)
confusion_matrix = confusion_matrix(y_test, y_pred)
print(f"Accuracy: {accuracy_score:.2f}")
print("Classification Report:", classification_report)
print("Confusion Matrix:\n", confusion_matrix)
# %%

plt.figure(figsize=(10, 6))
tree.plot_tree(dtc, filled=True)
# %%
from sklearn.metrics import accuracy_score
dtc2 = DecisionTreeClassifier(max_depth=2)
dtc2.fit(X_train,y_train)
y_train_pred = dtc2.predict(X_train)
print(accuracy_score(y_train, y_train_pred))
# %%

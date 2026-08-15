from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data #shape (150,4)
y = iris.target #shape (150,4)
print(iris.feature_names, iris.target_names)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #42 is deterministic

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train) #trains the model with data

import joblib
joblib.dump(model,r"C:\Users\Meji\OneDrive\AI Projects\Iris Classifier\outputs\model.joblib")

y_pred = model.predict(X_test) #returns 30 samples
#Returns 0, 1, 2 where 0 = Sentosa, 1 = versicolour, 2 = Virginica

print("predictions:", y_pred[:5])
print("True labels:", y_test[:5])

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#Saves confusion matrix to .png file
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix

cm=confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")

plt.savefig("confusion_matrix.png", dpi=300, bbox_inches="tight") #saves file to unspecified location
plt.savefig(r"C:\Users\Meji\OneDrive\AI Projects\Iris Classifier\outputs\confusion_matrix.png", dpi=300, bbox_inches="tight") #saves file to specified location
plt.close()


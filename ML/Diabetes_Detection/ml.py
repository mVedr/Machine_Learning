import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

diabetes_dataset = pd.read_csv('diabetes.csv')
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']


scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)

X = standardized_data
Y = diabetes_dataset['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

classifiers = {
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(random_state=2),
    "AdaBoost": AdaBoostClassifier(random_state=2),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Logistic Regression": LogisticRegression(random_state=2),
    "SVM": SVC(kernel='linear', random_state=2)
}


for name, clf in classifiers.items():
    clf.fit(X_train, Y_train)

    train_predictions = clf.predict(X_train)
    train_accuracy = accuracy_score(train_predictions, Y_train)
    print(f'{name} - Training accuracy: {train_accuracy:.2f}')

    test_predictions = clf.predict(X_test)
    test_accuracy = accuracy_score(test_predictions, Y_test)
    print(f'{name} - Testing accuracy: {test_accuracy:.2f}')

    input_data = (5, 166, 72, 19, 175, 25.8, 0.587, 51)
    arr = np.asarray(input_data)
    nums = arr.reshape(1, -1)
    std_data = scaler.transform(nums)

    prediction = clf.predict(std_data)
    if prediction[0] == 0:
        print(f'{name} - The person is not diabetic')
    else:
        print(f'{name} - The person is diabetic')

    print()

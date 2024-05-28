import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data from CSV
data = pd.read_csv('tennisdata.csv')
print("The first 5 values of data are:\n", data.head())

# Obtain Train data and Train output
X = data.iloc[:, :-1]
print("\nThe first 5 values of train data are:\n", X.head())

y = data.iloc[:, -1]
print("\nThe first 5 values of train output are:\n", y.head())

# Convert categorical features to numerical values
le_outlook = LabelEncoder()
X['Outlook'] = le_outlook.fit_transform(X['Outlook'])

le_temperature = LabelEncoder()
X['Temperature'] = le_temperature.fit_transform(X['Temperature'])

le_humidity = LabelEncoder()
X['Humidity'] = le_humidity.fit_transform(X['Humidity'])

le_windy = LabelEncoder()
X['Windy'] = le_windy.fit_transform(X['Windy'])

print("\nNow the train data is:\n", X.head())

le_play_tennis = LabelEncoder()
y = le_play_tennis.fit_transform(y)
print("\nNow the train output is:\n", y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Train the model
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is:", accuracy)

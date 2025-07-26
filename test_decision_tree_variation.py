import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
data_path = r"../../../AppData/Local/Temp/940cb257-4331-4493-90e9-54d09fa8e413_bank-additional.zip.bank-additional.zip/bank-additional/bank-additional.csv"
df = pd.read_csv(data_path, sep=';')

# Preprocessing
df.replace('unknown', np.nan, inplace=True)
df.dropna(inplace=True)

# Encode categorical variables
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    if col != 'y':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# Encode target variable
df['y'] = df['y'].map({'yes': 1, 'no': 0})

# Features and target
X = df.drop('y', axis=1)
y = df['y']

# Split data with different test size and random state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

# Train decision tree classifier
clf = DecisionTreeClassifier(random_state=24)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy with test_size=0.3 and random_state=24: {accuracy:.4f}")

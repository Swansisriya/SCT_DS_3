import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load dataset
data_path = r"../../../AppData/Local/Temp/9ef4c6bd-6988-43f3-b5d7-f9783ffc69d6_bank-additional.zip.bank-additional.zip/bank-additional/bank-additional-full.csv"
df = pd.read_csv("data_path sep=';'")

# Preprocessing
# Replace unknown with NaN
df.replace('unknown', np.nan, inplace=True)

# Drop rows with missing values for simplicity
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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Predict and evaluate
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print("Classification Report:")
print(report)

# Plot decision tree and save as PDF with improved design
plt.figure(figsize=(30,15))  # Increased figure size for spaciousness
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True, fontsize=14, proportion=True)
plt.title("Decision Tree Classifier", fontsize=20)
plt.tight_layout(pad=5.0)  # Add padding for clarity
plt.savefig("decision_tree_plot.pdf")
plt.close()

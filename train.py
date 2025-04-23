import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

def load_data(file_path):
    """Load the dataset from a CSV file."""
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    """Preprocess the data by splitting into features and target variable."""
    X = data.drop(columns=['DEATH_EVENT'])
    y = data['DEATH_EVENT']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

# Handing outliers
df= load_data('heart_failure_clinical_records_dataset.csv')
df['age'] = df['age'].astype(int)
outlier_colms = ['creatinine_phosphokinase', 'ejection_fraction', 'platelets', 'serum_creatinine', 'serum_sodium']
df1 = df.copy()

def handle_outliers(df, colm):
    '''Change the values of outlier to upper and lower whisker values '''
    q1 = df.describe()[colm].loc["25%"]
    q3 = df.describe()[colm].loc["75%"]
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    for i in range(len(df)):
        if df.loc[i,colm] > upper_bound:
            df.loc[i,colm]= upper_bound
        if df.loc[i,colm] < lower_bound:
            df.loc[i,colm]= lower_bound
    return df

for colm in outlier_colms:
    df1 = handle_outliers(df1, colm)

X,y=preprocess_data(df1)
X_train, X_test, y_train, y_test = split_data(X, y)

def train_model(X_train, y_train):
    """Train the XGBoost model."""
    xgb_clf = XGBClassifier(n_estimators=200, max_depth=4, max_leaves=5, random_state=42)
    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred)
    f1 = f1_score(y_train, y_pred, average='weighted')
    print(f"Training Accuracy: {accuracy:.4f}")

    return xgb_clf

def evaluate_model(model, X_test, y_test):
    """Evaluate the model on the test set."""
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    return accuracy, f1

model = train_model(X_train, y_train)
accuracy, f1 = evaluate_model(model, X_test, y_test)

def save_model(model, file_path):
    """Save the trained model to a file."""
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

save_file_name = "api/app/xgboost-model.pkl"

joblib.dump(model, save_file_name)
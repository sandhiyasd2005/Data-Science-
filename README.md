from google.colab import drive
drive.mount('/content/drive
import pandas as pd

# Load the dataset
df = pd.read_csv("csv file.csv")

# Show the first 5 rows
print(df.head())

# Summary info
print(df.info())

# Check for missing values
print(df.isnull().sum())
from IPython import get_ipython
from IPython.display import display
# %%
from google.colab import drive
drive.mount('/content/drive')
# %%
import pandas as pd

# Load the dataset
# Replace "csv file.csv" with the actual path to your file if it's not in the same directory
df = pd.read_csv("csv file.csv")

# Show the first 5 rows
print(df.head())

# Summary info
print(df.info())

# Check for missing values
print(df.isnull().sum())

# %%
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split # Import train_test_split

# Assuming 'your_actual_target_column_name' is the name of your target variable column in df
# Replace 'your_actual_target_column_name' with the actual name of your target column
target_column_name = 'your_actual_target_column_name' # <--- REPLACE THIS WITH YOUR ACTUAL TARGET COLUMN NAME

# Check if the target column exists in the DataFrame
if target_column_name not in df.columns:
    print(f"Error: Target column '{target_column_name}' not found in the DataFrame.")
    # You might want to add code here to list available columns or exit
else:
    X = df.drop(target_column_name, axis=1) # Features are all columns except the target
    y = df[target_column_name]             # Target variable

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Adjust test_size as needed

    # Initialize the model
    model = RandomForestClassifier(random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# %%
# This cell was a duplicate and is not needed.
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import classification_report, confusion_matrix

# # Initialize the model
# model = RandomForestClassifier(random_state=42)

# # Train the model
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Evaluate the model
# print("Confusion Matrix:")
# print(confusion_matrix(y_test, y_pred))
# print("\nClassification Report:")
# print(classification_report(y_test, y_pred))

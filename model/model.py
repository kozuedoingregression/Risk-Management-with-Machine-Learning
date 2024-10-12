# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:37.784320Z","iopub.execute_input":"2024-10-11T12:43:37.785201Z","iopub.status.idle":"2024-10-11T12:43:37.791399Z","shell.execute_reply.started":"2024-10-11T12:43:37.785130Z","shell.execute_reply":"2024-10-11T12:43:37.789736Z"},"jupyter":{"outputs_hidden":false}}
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:37.796237Z","iopub.execute_input":"2024-10-11T12:43:37.796689Z","iopub.status.idle":"2024-10-11T12:43:37.822515Z","shell.execute_reply.started":"2024-10-11T12:43:37.796651Z","shell.execute_reply":"2024-10-11T12:43:37.821101Z"},"jupyter":{"outputs_hidden":false}}
df= pd.read_csv('dataset/german_credit_data.csv')
df = df.drop(columns=['Unnamed: 0'])
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:37.824638Z","iopub.execute_input":"2024-10-11T12:43:37.825130Z","iopub.status.idle":"2024-10-11T12:43:37.848487Z","shell.execute_reply.started":"2024-10-11T12:43:37.825077Z","shell.execute_reply":"2024-10-11T12:43:37.846942Z"},"jupyter":{"outputs_hidden":false}}
# Handle missing values by filling with a placeholder or a statistical value
df = df.copy()  # Ensure df is a copy and avoid chained assignment issues
df['Saving accounts'] = df['Saving accounts'].fillna('unknown')
df['Checking account'] = df['Checking account'].fillna('unknown')

# Encode categorical variables
label_encoders = {}
for column in ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Display the cleaned and encoded dataset
df.head()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:37.850040Z","iopub.execute_input":"2024-10-11T12:43:37.850440Z","iopub.status.idle":"2024-10-11T12:43:39.598575Z","shell.execute_reply.started":"2024-10-11T12:43:37.850402Z","shell.execute_reply":"2024-10-11T12:43:39.597355Z"},"jupyter":{"outputs_hidden":false}}
# Ensure that any infinite values are converted to NaN before analysis
df = df.replace([np.inf, -np.inf], np.nan)

# Age distribution
sns.histplot(df['Age'], kde=True)
plt.title('Age Distribution')
plt.show()

# Credit amount distribution
sns.histplot(df['Credit amount'], kde=True)
plt.title('Credit Amount Distribution')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:39.601620Z","iopub.execute_input":"2024-10-11T12:43:39.602174Z","iopub.status.idle":"2024-10-11T12:43:39.620687Z","shell.execute_reply.started":"2024-10-11T12:43:39.602094Z","shell.execute_reply":"2024-10-11T12:43:39.619239Z"},"jupyter":{"outputs_hidden":false}}
# Define features and target
X = df.drop(columns=['Credit amount'])  # Assuming 'Credit amount' as target
y = df['Credit amount'] > df['Credit amount'].median()  # Binary classification (High/Low Credit Amount)

b=X.columns

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:39.622129Z","iopub.execute_input":"2024-10-11T12:43:39.622556Z","iopub.status.idle":"2024-10-11T12:43:39.631896Z","shell.execute_reply.started":"2024-10-11T12:43:39.622515Z","shell.execute_reply":"2024-10-11T12:43:39.630513Z"},"_kg_hide-input":true,"jupyter":{"outputs_hidden":false}}
b

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:39.633435Z","iopub.execute_input":"2024-10-11T12:43:39.633898Z","iopub.status.idle":"2024-10-11T12:43:39.871210Z","shell.execute_reply.started":"2024-10-11T12:43:39.633853Z","shell.execute_reply":"2024-10-11T12:43:39.870171Z"},"jupyter":{"outputs_hidden":false}}
# Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:39.872912Z","iopub.execute_input":"2024-10-11T12:43:39.873283Z","iopub.status.idle":"2024-10-11T12:43:39.892673Z","shell.execute_reply.started":"2024-10-11T12:43:39.873247Z","shell.execute_reply":"2024-10-11T12:43:39.891368Z"},"jupyter":{"outputs_hidden":false}}
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T12:43:39.894696Z","iopub.execute_input":"2024-10-11T12:43:39.895303Z","iopub.status.idle":"2024-10-11T12:43:39.967735Z","shell.execute_reply.started":"2024-10-11T12:43:39.895248Z","shell.execute_reply":"2024-10-11T12:43:39.966508Z"},"jupyter":{"outputs_hidden":false}}
import joblib
joblib.dump(model, "creditRisks.pkl")

# %% [code] {"execution":{"iopub.status.busy":"2024-10-11T13:25:35.452225Z","iopub.execute_input":"2024-10-11T13:25:35.453705Z","iopub.status.idle":"2024-10-11T13:25:35.460024Z","shell.execute_reply.started":"2024-10-11T13:25:35.453643Z","shell.execute_reply":"2024-10-11T13:25:35.458518Z"}}
import sklearn
print(sklearn.__version__)
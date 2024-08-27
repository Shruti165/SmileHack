# In your Jupyter Notebook or Python script

from .models import AIResult

# After the model makes predictions, store the results
predictions = model.predict(X_test)

# Assuming X_test has a corresponding list of supplier names
for i, prediction in enumerate(predictions):
    AIResult.objects.create(
        supplier_name=supplier_names[i],
        prediction=prediction
    )


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('supplier_data.csv')

# Convert 'Avg. Cost($)' to numeric values
df['Avg. Cost($)'] = df['Avg. Cost($)'].str.replace('k', '').astype(float) * 1000

# Encode categorical variables
label_encoders = {}
for column in ['Supplier Name', 'Region', 'Country', 'Function', 'Service', 'Delivery Frequency', 'Geographical Coverage', 'Technology Used']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Convert other columns to numeric as needed
df['Number of Escalations'] = df['Number of Escalations'].astype(int)
df['Average Delivery Time (days)'] = df['Average Delivery Time (days)'].astype(float)

# Define features and target
X = df[['Supplier Name', 'Region', 'Country', 'Function', 'Service', 'Avg. Cost($)', 'Average Delivery Time (days)', 'Number of Escalations', 'Lead Time (days)', 'Total Shipments', 'On-Time Delivery Rate (%)', 'Customer Satisfaction Score', 'Return Rate (%)', 'Contract Duration (months)', 'Geographical Coverage', 'Technology Used']]
y = df['On-Time Delivery Rate (%)'] > 85  # Example target variable for classification: On-Time Delivery Rate > 85%

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Print classification report and accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Display feature importances
importances = rf_model.feature_importances_
feature_names = X.columns
feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

import matplotlib.pyplot as plt
plt.figure(figsize=(12, 8))
feature_importances.plot(kind='bar')
plt.title('Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.show()

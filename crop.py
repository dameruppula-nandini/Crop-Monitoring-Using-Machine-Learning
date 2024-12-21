import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv(r'C:\Users\varun\Desktop\AD1\CROP\crop_recommendation (2).csv') 
df.head()

# Define features and target
target_column = 'label' 
features = df.drop(columns=[target_column])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(features, df[target_column])

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train) 

# Predict on test set and evaluate
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) 
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))

# Data visualization: Climate bar plot (Temperature vs Humidity)
sns.barplot(x='temperature', y='humidity', data=df)
plt.title('Climate') 
plt.xlabel('Temperature')
plt.ylabel('Humidity')					
plt.show()

# Create bins for rainfall to better visualize the line plot
df['rainfall_bin'] = pd.cut(df['rainfall'], bins=5, labels=['Very Low', 'Low', 'Moderate', 'High', 'Very High'])

# Line plot: Temperature vs Humidity with Rainfall categories
plt.figure(figsize=(10, 6))  # Increase plot size for better visibility
sns.lineplot(x='temperature', y='humidity', data=df, hue='rainfall_bin', marker='o', linewidth=2, palette='coolwarm')
plt.title('Temperature vs Humidity with Rainfall') 
plt.xlabel('Temperature')
plt.ylabel('Humidity')					
plt.show()

# Scatter plot: Temperature vs Rainfall with Humidity
sns.scatterplot(x='temperature', y='rainfall', data=df, hue='humidity')
plt.title('Temperature vs Rainfall with Humidity') 
plt.xlabel('Temperature')
plt.ylabel('Rainfall')					
plt.show()

# Further analysis and visualization can be added based on your project needs

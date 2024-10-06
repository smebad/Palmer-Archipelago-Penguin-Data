# 🐧 Palmer Archipelago (Antarctica) Penguin Data

Author: [Syed Muhammad Ebad](https://www.kaggle.com/syedmuhammadebad)  
Date: 23-June-2024  
[Send me an email](mailto:mohammadebad1@hotmail.com) | [Visit my GitHub profile](https://github.com/smebad)

---

## 📖 Introduction:
This project uses the Palmer Archipelago (Antarctica) penguin data to build a machine learning model that classifies the species of penguins based on features like bill length, bill depth, flipper length, and more.  
We will use an **SVM classifier** and handle categorical features using **OneHotEncoder**.

---

## 🔢 Project Steps:

1. **Data Loading**: Load and analyze the dataset.
2. **Handling Missing Values**: Drop rows with missing values or apply forward fill to handle missing data.
3. **Data Visualization**: Use pairplot to visualize the relationships between features.
4. **Correlation Matrix**: Plot the correlation matrix for numeric features.
5. **Data Splitting**: Split the dataset into training and testing sets.
6. **Categorical Encoding**: Apply OneHotEncoding to categorical features.
7. **Pipeline Creation**: Build a pipeline to preprocess the data and train the SVM model.
8. **Model Training & Evaluation**: Train the model and evaluate its accuracy.

---

## 🧰 Libraries Used:

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
```
## 📊 Data Loading and Initial Analysis:
```
# Load the datasets
df = pd.read_csv('penguins_size.csv')
df2 = pd.read_csv('penguins_lter.csv')

# Display the first few rows
print(df.head(5))
print(df2.head(5))

# Dataset Information
df.info()

# Value counts for 'island' and 'species'
print(df['island'].value_counts())
print(df['species'].value_counts())
```
## 🚨 Handling Missing Values:
```
# Drop rows with missing values
df = df.dropna()

# Forward fill for rows with missing data
df = df.ffill()

# Check data after handling missing values
df.info()
```
## 🎨 Data Visualization:
```
# Pairplot for data visualization
sns.pairplot(df, hue='species', height=2.5, diag_kind='hist')
plt.show()

# Correlation Matrix
numeric_df = df.select_dtypes(include=[np.number])
corr_matrix = numeric_df.corr()

# Heatmap for correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(corr_matrix, annot=True, cmap='RdBu')
plt.show()
```
## 🧪 Data Preparation:
```
# Split the data into training and testing sets
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Separate features and target variable
train_y = train['species']
test_y = test['species']
train_x = train.drop(['species'], axis=1)
test_x = test.drop(['species'], axis=1)

# Categorical Columns
categorical_features = train_x.select_dtypes(include=['object']).columns
```
## 🏗️ Building the Model Pipeline:
```
# OneHotEncoder for categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Pipeline: preprocessing + SVM classifier
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', svm.SVC(kernel='linear'))
])

# Cross-validation
cv_scores = cross_val_score(model_pipeline, train_x, train_y, cv=5)
print("Cross-validation scores:", cv_scores)
print("Mean cross-validation score:", cv_scores.mean())

# Fit the model
model_pipeline.fit(train_x, train_y)
```
## 🧮 Model Evaluation:
```
# Predictions
predictions = model_pipeline.predict(test_x)

# Accuracy Calculation
accuracy = metrics.accuracy_score(predictions, test_y)
print("Model Accuracy:", accuracy)
```
## 🎯 Conclusion:
* The SVM classifier was trained on the Palmer Archipelago penguin dataset.
* The model achieved an accuracy of 100% on the test set, indicating perfect performance on this dataset.
* While a perfect accuracy score is impressive, it's essential to ensure the model generalizes well to new, unseen data to avoid overfitting.
* Cross-validation was used to validate the model’s performance, and further testing can be done to verify its robustness.



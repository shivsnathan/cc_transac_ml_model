import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from geopy.distance import geodesic
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('credit_card_transactions.csv')

# Drop rows with missing values for simplicity
df = df.dropna(subset=['dob', 'merch_lat', 'merch_long'])

# Date & Time Features
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5
df['is_night'] = (df['hour'] < 6) | (df['hour'] > 20)
df['log_amt'] = np.log1p(df['amt'])

# Age Feature
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
df['age'] = (pd.to_datetime('now') - df['dob']).dt.days // 365
df['age'] = df['age'].fillna(df['age'].median())

# Distance Feature
df['distance'] = df.apply(lambda row: geodesic((row['lat'], row['long']),
                                               (row['merch_lat'], row['merch_long'])).km, axis=1)

# Reduce merchant cardinality
top_merchants = df['merchant'].value_counts().nlargest(20).index
df['merchant'] = df['merchant'].apply(lambda x: x if x in top_merchants else 'other')

# Select columns
df = df[['log_amt', 'merchant', 'hour', 'day_of_week', 'is_weekend', 'is_night',
         'distance', 'age', 'gender', 'job', 'city_pop', 'category']]

# Encode target
df['category'] = df['category'].astype('category')
category_mapping = dict(enumerate(df['category'].cat.categories))
df['category'] = df['category'].cat.codes

# One-hot encoding for categorical features
categorical_cols = ['merchant', 'gender', 'job']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Split features/target
X = df.drop('category', axis=1)
y = df['category']

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Train model
model = RandomForestClassifier(n_estimators=100, max_depth=12, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train_res, y_train_res)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

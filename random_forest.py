import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Load dataset
df = pd.read_csv('credit_card_transactions.csv')

# Feature selection
df = df[['amt', 'merchant', 'trans_date_trans_time', 'category']]

# Date & Time Features
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5
df['is_night'] = (df['hour'] < 6) | (df['hour'] > 20)
df['log_amt'] = np.log1p(df['amt'])  # More robust to skewed data
df = df.drop(['trans_date_trans_time', 'amt'], axis=1)

# Reduce merchant cardinality
top_merchants = df['merchant'].value_counts().nlargest(20).index
df['merchant'] = df['merchant'].apply(lambda x: x if x in top_merchants else 'other')

# One-hot encoding for 'merchant'
df = pd.get_dummies(df, columns=['merchant'], drop_first=True)

# Encode target variable
df['category'] = df['category'].astype('category')
category_mapping = dict(enumerate(df['category'].cat.categories))
df['category'] = df['category'].cat.codes

# Split features and target
X = df.drop('category', axis=1)
y = df['category']

# Stratified train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train a Random Forest with class_weight and basic hyperparameter tuning
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))


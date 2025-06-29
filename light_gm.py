import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier

# Load dataset
df = pd.read_csv('credit_card_transactions.csv').sample(n=5000, random_state=42)

# === Feature Engineering ===
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['is_weekend'] = df['day_of_week'] >= 5
df['is_night'] = (df['hour'] < 6) | (df['hour'] > 20)
df['log_amt'] = np.log1p(df['amt'])
df = df.drop(['trans_date_trans_time', 'amt'], axis=1)

# Reduce merchant cardinality
top_merchants = df['merchant'].value_counts().nlargest(20).index
df['merchant'] = df['merchant'].apply(lambda x: x if x in top_merchants else 'other')
df = pd.get_dummies(df, columns=['merchant'], drop_first=True)

# Encode target variable
df['category'] = df['category'].astype('category')
category_mapping = dict(enumerate(df['category'].cat.categories))
df['category'] = df['category'].cat.codes

# === Features and Target ===
X = df.drop('category', axis=1)
y = df['category']

# Ensure all features are numeric
non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
if len(non_numeric_cols) > 0:
    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)

# Sanitize column names for LightGBM
X.columns = X.columns.str.replace('[^A-Za-z0-9_]+', '_', regex=True)

# === Train-test split (stratified) ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("NaNs in X_train before SMOTE:", X_train.isnull().sum().sum())  # Add this line for debugging

# Impute missing values in features
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# If you still have NaNs, fill with 0 as a last resort:
X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

# === SMOTE for Class Balancing ===
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# === Train LightGBM ===
model = LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y)),
    class_weight='balanced',
    n_estimators=300,
    max_depth=10,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train_res, y_train_res)

# === Evaluate ===
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=category_mapping.values())
disp.plot(xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# === Feature Importance Plot ===
importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = importances.sort_values(ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 20 Feature Importances (LightGBM)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()

# === Save Category Mapping ===
with open('category_mapping.json', 'w') as f:
    json.dump(category_mapping, f)
print("Category mapping saved to 'category_mapping.json'")

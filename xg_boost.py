import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_csv('credit_card_transactions.csv').sample(n=5000, random_state=42)

# Feature engineering
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

# Encode target
df['category'] = df['category'].astype('category')
category_mapping = dict(enumerate(df['category'].cat.categories))
df['category'] = df['category'].cat.codes

# Features and target
X = df.drop('category', axis=1)
y = df['category']

# Ensure all features are numeric
non_numeric_cols = X.select_dtypes(include=['object', 'category']).columns
if len(non_numeric_cols) > 0:
    X = pd.get_dummies(X, columns=non_numeric_cols, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Impute missing values in features
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())

# SMOTE for class balancing
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train XGBoost model
model = XGBClassifier(
    n_estimators=20,
    max_depth=5,
    learning_rate=0.1,
    objective='multi:softmax',
    num_class=len(np.unique(y)),
    use_label_encoder=False,
    eval_metric='mlogloss',
    tree_method='hist',
    random_state=42
)
model.fit(X_train_res, y_train_res)

# Predict & evaluate
y_pred = model.predict(X_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

# Feature importance plot
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
top_features = feat_importances.sort_values(ascending=False).head(20)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 20 Feature Importances (XGBoost)")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()


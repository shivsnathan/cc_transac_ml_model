import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical

# Load dataset
df = pd.read_csv('credit_card_transactions.csv')

# Feature selection & engineering
df = df[['amt', 'merchant', 'trans_date_trans_time', 'category']]

# Convert datetime
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df = df.drop('trans_date_trans_time', axis=1)

# Log transform skewed 'amt'
df['log_amt'] = np.log1p(df['amt'])
df = df.drop('amt', axis=1)

# Reduce merchant cardinality
top_merchants = df['merchant'].value_counts().nlargest(20).index
df['merchant'] = df['merchant'].apply(lambda x: x if x in top_merchants else 'other')

# One-hot encode merchant
df = pd.get_dummies(df, columns=['merchant'], drop_first=True)

# Encode target label
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])

# Prepare features and labels
X = df.drop('category', axis=1).values
y = to_categorical(df['category'].values)  # one-hot encode for multi-class

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df['category']
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build simple neural network
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(y.shape[1], activation='softmax')  # output layer for multi-class
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train
model.fit(X_train, y_train, epochs=20, batch_size=256, validation_split=0.1)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.4f}")


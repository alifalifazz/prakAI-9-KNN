# knn_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("Iris.csv")

# Pisahkan fitur dan label
X = df.drop(columns='species')
y = df['species']

# Normalisasi fitur
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Buat model KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Evaluasi akurasi
y_pred = knn.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Akurasi: {acc:.2f}")

# Simpan model dan scaler
joblib.dump(knn, 'model_knn.pkl')
joblib.dump(scaler, 'scaler.pkl')
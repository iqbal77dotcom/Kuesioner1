# ===============================
# Import Library
# ===============================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

# ===============================
# Step 1: Buat Data
# ===============================
np.random.seed(42)
data_list = []

def generate_row(mhs_id, total_target):
    while True:
        q_values = np.random.randint(1, 6, size=10)  # 10 pertanyaan (skor 1–5)
        total = q_values.sum()
        if total_target[0] <= total <= total_target[1]:
            return [f"MHS{mhs_id:03d}"] + q_values.tolist() + [total]

# 33 mahasiswa kategori Rendah
for i in range(1, 34):
    data_list.append(generate_row(i, (10, 25)))

# 34 mahasiswa kategori Sedang
for i in range(34, 68):
    data_list.append(generate_row(i, (26, 40)))

# 33 mahasiswa kategori Tinggi
for i in range(68, 101):
    data_list.append(generate_row(i, (41, 50)))

# ===============================
# Step 2: Buat DataFrame
# ===============================
columns = ["ID"] + [f"Q{i}" for i in range(1, 11)] + ["Total"]
df = pd.DataFrame(data_list, columns=columns)

# Tambahkan kolom Label kategori
def kategori(total):
    if total <= 25:
        return "Rendah"
    elif total <= 40:
        return "Sedang"
    else:
        return "Tinggi"

df["Label"] = df["Total"].apply(kategori)

print(df.head())
print(df["Label"].value_counts())

# ===============================
# Step 3: Persiapan Data (X, y)
# ===============================
X = df[[f"Q{i}" for i in range(1, 11)]].values
y = df[["Label"]].values

# One-hot encoding label
enc = OneHotEncoder(sparse=False)
y_encoded = enc.fit_transform(y)

# Normalisasi fitur
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y
)

# ===============================
# Step 4: Bangun Model Backpropagation
# ===============================
model = Sequential([
    Input(shape=(10,)),                   # 10 pertanyaan
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(3, activation="softmax")        # 3 kelas: Rendah, Sedang, Tinggi
])

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ===============================
# Step 5: Training Model
# ===============================
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    verbose=1
)

# ===============================
# Step 6: Evaluasi Model
# ===============================
# Plot akurasi dan loss
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Akurasi')

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss')
plt.show()

# Confusion Matrix
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)
y_true = y_test.argmax(axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=enc.categories_[0],
            yticklabels=enc.categories_[0],
            cmap="Blues")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=enc.categories_[0]))

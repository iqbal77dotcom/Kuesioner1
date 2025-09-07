# ===============================
# 📊 Visualisasi Distribusi
# ===============================
plt.figure(figsize=(6,4))
sns.countplot(x="Kategori Kejenuhan", data=df, order=["Rendah", "Sedang", "Tinggi"], palette="Set2")
plt.title("Distribusi Tingkat Kejenuhan Mahasiswa ITAF Kupang")
plt.xlabel("Kategori Kejenuhan")
plt.ylabel("Jumlah Mahasiswa")
plt.tight_layout()
plt.show()

# ===============================
# 🥧 Visualisasi Pie Chart
# ===============================
kategori_counts = df["Kategori Kejenuhan"].value_counts().reindex(["Rendah", "Sedang", "Tinggi"])
plt.figure(figsize=(6,6))
plt.pie(kategori_counts, labels=kategori_counts.index, autopct='%1.1f%%',
        colors=["skyblue", "orange", "salmon"])
plt.title("Persentase Tingkat Kejenuhan Mahasiswa ITAF Kupang")
plt.show()

# ===============================
# ⚙️ Persiapan Data
# ===============================
X = df.iloc[:, 1:11].values  # Q1–Q10
y = df["Kategori Kejenuhan"].values

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)  # pastikan sparse_output=False
y_encoded = encoder.fit_transform(y.reshape(-1,1))

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42
)

# ===============================
# 🧠 Model Backpropagation
# ===============================
model = Sequential([
    Dense(12, input_shape=(10,), activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

# ===============================
# 🚀 Kompilasi & Training Model
# ===============================
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=100,
    batch_size=8,
    verbose=1
)

# ===============================
# 📊 Visualisasi Hasil Training
# ===============================
plt.figure(figsize=(12,5))

# Akurasi
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Akurasi Model')
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Model')
plt.legend()

plt.tight_layout()
plt.show()

# ===============================
# 📈 Evaluasi Model
# ===============================
from sklearn.metrics import classification_report, confusion_matrix

# Prediksi
y_pred_prob = model.predict(X_test)
y_pred = y_pred_prob.argmax(axis=1)
y_true = y_test.argmax(axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=encoder.categories_[0],
            yticklabels=encoder.categories_[0],
            cmap="Blues")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix Backpropagation")
plt.show()

# Classification Report
print("=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=encoder.categories_[0]))

# ===============================
# 📂 Simpan Hasil Prediksi ke DataFrame
# ===============================
df_pred = df.copy()
df_pred["Prediksi"] = model.predict(X_scaled).argmax(axis=1)
df_pred["Prediksi"] = df_pred["Prediksi"].map(
    {i: label for i, label in enumerate(encoder.categories_[0])}
)

print(df_pred.head())

# Simpan ke CSV
df_pred.to_csv("hasil_prediksi_kejenuhan.csv", index=False)
print("✅ Hasil prediksi disimpan ke: hasil_prediksi_kejenuhan.csv")

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers

RAW_PATH = "data/raw/WISDM_ar_v1.1_raw.txt"
CLEAN_PATH = "data/processed/WISDM_clean.csv"
WINDOWS_PATH = "data/processed/wisdm_windows.npz"


if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Raw file not found at: {RAW_PATH}")

# STEP 1: Load raw data
df = pd.read_csv(RAW_PATH,
                 header=None,
                 names=["user", "activity", "timestamp", "x", "y", "z"],
                 sep=",",
                 engine="python",
                 on_bad_lines="skip")

print("\n[RAW DATA]")
print("Data type before cleaning:\n", df.dtypes)
print("Shape:", df.shape)
print(df.head(5))
print("\nMissing values:\n", df.isna().sum())
print("\nActivity counts:\n", df["activity"].value_counts())
print("Example raw z values:", df["z"].head(3).tolist())


# STEP 2: Clean
df["z"] = df["z"].astype(str).str.replace(";", "", regex=False)

# Converting numeric columns and bad values become NaN
df["user"] = pd.to_numeric(df["user"], errors="coerce")
df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
df["x"] = pd.to_numeric(df["x"], errors="coerce")
df["y"] = pd.to_numeric(df["y"], errors="coerce")
df["z"] = pd.to_numeric(df["z"], errors="coerce")

# drop invalid rows, duplicates and sort
df = df.dropna()
df = df.drop_duplicates()
df = df.sort_values(["user", "timestamp"]).reset_index(drop=True)

print("\n[CLEAN DATA]")
print("After cleaning shape:", df.shape)
print("Missing values after cleaning:\n", df.isna().sum())
print("\nActivity counts after cleaning:\n", df["activity"].value_counts())
print("\nData type after cleaning:\n", df.dtypes)

# Save cleaned dataset
df.to_csv(CLEAN_PATH, index=False)
print("Saved cleaned dataset:", CLEAN_PATH)


# STEP 3: Preprocessing/Windowing
window_size = 100
step_size = 50
X_list = []
y_list = []

# Create windows per user so data from 2 people never mixes in one window
for user_id, user_df in df.groupby("user"):
    user_df = user_df.reset_index(drop=True)

    data_xyz = user_df[["x", "y", "z"]].values
    labels = user_df["activity"].values

    for start in range(0, len(user_df) - window_size + 1, step_size):
        end = start + window_size

        X_window = data_xyz[start:end]          # shape (100, 3)
        y_window = labels[start:end]        # 100 labels

        # Majority activity label inside the window
        unique_labels, counts = np.unique(y_window, return_counts=True)
        majority_label = unique_labels[np.argmax(counts)]

        X_list.append(X_window)
        y_list.append(majority_label)

# Convert to numpy arrays
X = np.array(X_list, dtype=np.float32)
y = np.array(y_list)

print("\n[WINDOWS]")
print("X shape:", X.shape)
print("y shape:", y.shape)

np.savez(WINDOWS_PATH, X=X, y=y)
print("Saved windows:", WINDOWS_PATH)

# pick a window index you want to visualize
i = 0

plt.figure()
plt.plot(X[i][:, 0], label="x")
plt.plot(X[i][:, 1], label="y")
plt.plot(X[i][:, 2], label="z")
plt.title(f"Example Window Signal (Label: {y[i]})")
plt.xlabel("Time steps (within window)")
plt.ylabel("Acceleration")
plt.legend()
plt.tight_layout()
plt.savefig("data/processed/Window_signal.png", dpi=200)
plt.show()

# STEP 4: Random Forest model (ML Model)
# Random Forest needs 2D input -> flatten 100x3 into 300 features
X_flat = X.reshape(X.shape[0], -1)

label_encode = LabelEncoder()
y_enc = label_encode.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X_flat, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

preds = rf.predict(X_test)

print("\n[RANDOM FOREST]")
print("Accuracy:", round(accuracy_score(y_test, preds), 4))
print(classification_report(y_test, preds, target_names=label_encode.classes_))

joblib.dump(rf, "data/processed/rf_model.joblib")
joblib.dump(label_encode, "data/processed/label_encoder.joblib")
print("Saved: data/processed/rf_model.joblib")
print("Saved: data/processed/label_encoder.joblib")


# Reload model and predict ONE sample
rf_loaded = joblib.load("data/processed/rf_model.joblib")
label_encode_loaded = joblib.load("data/processed/label_encoder.joblib")

sample_X = X_test[0].reshape(1, -1)
sample_true = label_encode_loaded.inverse_transform([y_test[0]])[0]
sample_pred = label_encode_loaded.inverse_transform([rf_loaded.predict(sample_X)[0]])[0]

print("Loaded RF prediction (1 sample) -> True:", sample_true, "| Pred:", sample_pred)

# Confusion matrix plot
ConfusionMatrixDisplay.from_predictions(
    y_test,
    preds,
    display_labels=label_encode.classes_,
    xticks_rotation=45
)

plt.title("Random Forest - Confusion Matrix")
plt.tight_layout()
plt.savefig("data/processed/rf_confusion_matrix.png", dpi=200)
plt.show()

# Confusion matrix plot (Normalized)
ConfusionMatrixDisplay.from_predictions(
    y_test,
    preds,
    display_labels=label_encode.classes_,
    xticks_rotation=45,
    normalize="true"
)
plt.title("Random Forest - Confusion Matrix (Normalized)")
plt.tight_layout()
plt.savefig("data/processed/rf_confusion_matrix_normalized.png", dpi=200)
plt.show()

# recall bar chart
report_dict = classification_report(
    y_test, preds, target_names=label_encode.classes_, output_dict=True)

classes = label_encode.classes_
recall_values = [report_dict[c]["recall"] for c in classes]

plt.figure()
plt.bar(classes, recall_values)
plt.xticks(rotation=45)
plt.ylim(0, 1)
plt.title("Recall by Class (Random Forest)")
plt.ylabel("Recall")
plt.tight_layout()
plt.savefig("data/processed/rf_recall_by_class.png", dpi=200)
plt.show()


# STEP 5: CNN Model (Deep learning)
label_encode = LabelEncoder()
y_enc = label_encode.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)

# simple scaling using training stats
mean = X_train.mean(axis=(0, 1), keepdims=True)
std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

X_train = (X_train - mean) / std
X_test = (X_test - mean) / std

np.save("data/processed/cnn_mean.npy", mean)
np.save("data/processed/cnn_std.npy", std)
joblib.dump(label_encode, "data/processed/label_encoder_cnn.joblib")

num_classes = len(label_encode.classes_)

model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
    layers.Conv1D(32, 5, activation="relu"),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 5, activation="relu"),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation="relu"),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=8,
    batch_size=64,
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n[CNN]")
print("Test Accuracy:", round(float(test_acc), 4))

model.save("data/processed/cnn_model.keras")
print("Saved: data/processed/cnn_model.keras")

# Reload and predict one sample
loaded_model = keras.models.load_model("data/processed/cnn_model.keras")
label_encode2 = joblib.load("data/processed/label_encoder_cnn.joblib")

sample = X_test[0:1]  # already scaled
pred_class = np.argmax(loaded_model.predict(sample, verbose=0), axis=1)[0]

true_label = label_encode2.inverse_transform([y_test[0]])[0]
pred_label = label_encode2.inverse_transform([pred_class])[0]

print("Loaded CNN prediction (1 sample) -> True:", true_label, "| Pred:", pred_label)


# CNN Confusion matrix (normalized)
y_pred = np.argmax(loaded_model.predict(X_test, verbose=0), axis=1)

ConfusionMatrixDisplay.from_predictions(
    y_test, y_pred,
    display_labels=label_encode2.classes_,
    normalize="true",
    xticks_rotation=45
)

plt.title("CNN - Confusion Matrix (Normalized)")
plt.tight_layout()
plt.savefig("data/processed/cnn_confusion_matrix_normalized.png", dpi=200)
plt.show()

print("\nDONE - check data/processed/ for saved models and plots.")
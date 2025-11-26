import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import os
import pandas as pd
from pathlib import Path
import time

np.random.seed(0)
tf.random.set_seed(0)

# Load and prepare workout data
def load_workout_data(data_dir="TrainingDataEAI", window_sec=5, sample_rate=100):
    data_path = Path(data_dir)
    sequences = []
    labels = []

    activities = [
        ("WeightLift_", 0),
        ("Walking_", 1),
        ("Plank_", 2),
        ("JumpingJacks_", 3),
        ("Squats_", 4),
        ("JumpRope_", 5)
    ]

    window_size = int(window_sec * sample_rate)

    for prefix, label in activities:
        dirs = sorted([d for d in data_path.iterdir()
                      if d.is_dir() and d.name.startswith(prefix)])

        print(f"Found {len(dirs)} {prefix.rstrip('_')} sessions")

        for dir in dirs:
            accel_file = dir / "WatchAccelerometerUncalibrated.csv"
            hr_file = dir / "HeartRate.csv"

            if not accel_file.exists():
                continue

            accel_df = pd.read_csv(accel_file)

            if len(accel_df) < window_size:
                continue

            # Extract accelerometer readings
            features = accel_df[['x', 'y', 'z']].values

            # Include heart rate if available
            if hr_file.exists() and os.path.getsize(hr_file) > 50:
                hr_df = pd.read_csv(hr_file)
                if len(hr_df) > 0:
                    hr = hr_df['bpm'].mean()
                else:
                    hr = 100.0
            else:
                hr = 100.0

            hr_col = np.full((len(features), 1), hr)
            features = np.hstack([features, hr_col])

            # Create sliding windows from the session
            num_windows = min(4, len(features) // window_size)
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                if end <= len(features):
                    sequences.append(features[start:end])
                    labels.append(label)

    return sequences, labels

# Load the workout data
print("Loading workout data...")
seqs, labs = load_workout_data()
print(f"Found {len(seqs)} training sequences")

X = np.array(seqs, dtype=np.float32)
y = np.array(labs, dtype=np.int32)
print(f"Input shape: {X.shape}")

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

workouts = ["WeightLift", "Walking", "Plank", "JumpingJacks", "Squats", "JumpRope"]
print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}\n")

# Visualize the raw sensor data
colors = ['blue', 'orange', 'green', 'red', "purple", "black"]
feature_names = ["Accel X", "Accel Y", "Accel Z", "Heart Rate"]

fig, axes = plt.subplots(2, 2)
axes = axes.ravel()

for i, name in enumerate(feature_names):
    for c in range(len(workouts)):
        idx = np.where(y == c)[0]
        data = X[idx][:, :, i]

        np.random.seed(42 + c)
        samples = np.random.choice(len(data), min(3, len(data)), replace=False)

        for s in samples:
            axes[i].plot(data[s], color=colors[c], alpha=0.3, linewidth=1)

        median = np.median(data, axis=0)
        axes[i].plot(median, label=workouts[c], color=colors[c], linewidth=2.5)

    axes[i].set_title(name, fontweight='bold')
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Analyze frequency domain characteristics
fig, axes = plt.subplots(2, 2)
axes = axes.ravel()

for i, name in enumerate(feature_names):
    for c in range(len(workouts)):
        idx = np.where(y == c)[0]
        data = X[idx][:, :, i]

        ffts = []
        for sample in data:
            centered = sample - np.mean(sample)
            fft = np.abs(np.fft.fft(centered)[:len(centered)//2])
            ffts.append(fft)

        median_fft = np.median(ffts, axis=0)
        freqs = np.fft.fftfreq(X.shape[1], 1/100)[:X.shape[1]//2]

        axes[i].plot(freqs, median_fft, label=workouts[c],
                    color=colors[c], linewidth=2)

    axes[i].set_title(f'{name} - Frequency', fontweight='bold')
    axes[i].set_xlabel('Hz')
    axes[i].set_xlim([0, 10])
    axes[i].legend(fontsize=8)
    axes[i].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Build the classification model (using CNN for edge device compatibility)
def create_workout_model(input_size, hidden_size, num_classes, sequence_length=500):
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, input_size)),

        # First convolutional block
        layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Second convolutional block
        layers.Conv1D(64, kernel_size=5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Third convolutional block
        layers.Conv1D(hidden_size, kernel_size=3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.3),

        # Classification head
        layers.Dense(num_classes)
    ])
    return model

print(f"Using TensorFlow {tf.__version__}\n")

# Model configuration
LOAD_PRETRAINED = False  # Set to True to load existing model instead of training
PRETRAINED_PATH = 'workout_model.keras'

if LOAD_PRETRAINED and os.path.exists(PRETRAINED_PATH):
    print(f"Loading model from {PRETRAINED_PATH}...")
    model = keras.models.load_model(PRETRAINED_PATH)
else:
    print("Building model architecture...")
    model = create_workout_model(4, 32, len(workouts), sequence_length=X.shape[1])
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

# Train the model
if not (LOAD_PRETRAINED and os.path.exists(PRETRAINED_PATH)):
    print("Starting training...\n")
    epochs = 1000

    # Setup training callbacks
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=150,
        restore_best_weights=True
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7
    )

    class PrintProgress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}: Train acc {logs['accuracy']:.3f} | Val acc {logs['val_accuracy']:.3f}")

    start = time.time()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=len(X_train),
        callbacks=[early_stop, reduce_lr, PrintProgress()],
        verbose=0
    )

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed:.1f} seconds")

    train_losses = history.history['loss']
    train_accs = history.history['accuracy']
    val_losses = history.history['val_loss']
    val_accs = history.history['val_accuracy']
    best_val_acc = max(val_accs)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
else:
    print("Skipping training, using pretrained model\n")
    # Evaluate the loaded model
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)

    # Create history for plotting
    train_losses = [train_loss]
    train_accs = [train_acc]
    val_losses = [val_loss]
    val_accs = [val_acc]
    best_val_acc = val_acc

    print(f"Train accuracy: {train_acc:.4f}, Validation accuracy: {val_acc:.4f}")

# Generate predictions on validation set
preds = np.argmax(model.predict(X_val), axis=1)

# Plot training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(train_losses, label='Train')
ax1.plot(val_losses, label='Val')
ax1.set_title("Loss")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(train_accs, label='Train')
ax2.plot(val_accs, label='Val')
ax2.set_title("Accuracy")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.show()

cm = confusion_matrix(y_val, preds)
disp = ConfusionMatrixDisplay(cm, display_labels=workouts)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Validation Confusion Matrix")
plt.show()

print("\nPer-class validation results:")
for i, name in enumerate(workouts):
    total = (y_val == i).sum()
    correct = ((y_val == i) & (preds == i)).sum()
    print(f"  {name}: {correct}/{total} ({correct/total*100:.1f}%)")

# Save the trained model
params = model.count_params()
print(f"\nModel has {params:,} parameters")

model.save('workout_model.keras')
size_keras = os.path.getsize('workout_model.keras') / (1024 * 1024)
print(f"Saved Keras model: {size_keras:.2f} MB")

# Convert to TensorFlow Lite format
print("\nConverting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()

with open('workout_model.tflite', 'wb') as f:
    f.write(tflite_model)

size_tflite_fp32 = os.path.getsize('workout_model.tflite') / (1024 * 1024)
print(f"Saved TFLite FP32: {size_tflite_fp32:.2f} MB ({size_tflite_fp32/size_keras*100:.0f}% of original)")

# Verify the TFLite model works correctly
print("Validating TFLite model...")
interpreter = tf.lite.Interpreter(model_path='workout_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

preds_tflite = []
for i in range(len(X_val)):
    interpreter.set_tensor(input_details[0]['index'], X_val[i:i+1])
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    preds_tflite.append(np.argmax(output))

preds_tflite = np.array(preds_tflite)
acc_tflite = (preds_tflite == y_val).mean()
print(f"TFLite FP32 validation accuracy: {acc_tflite:.4f}")

# Convert to quantized FP16 version for even smaller size
print("\nConverting to quantized FP16...")
converter_fp16 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_fp16.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
converter_fp16.optimizations = [tf.lite.Optimize.DEFAULT]
converter_fp16.target_spec.supported_types = [tf.float16]
tflite_model_fp16 = converter_fp16.convert()

with open('workout_model_fp16.tflite', 'wb') as f:
    f.write(tflite_model_fp16)

size_tflite_fp16 = os.path.getsize('workout_model_fp16.tflite') / (1024 * 1024)
print(f"Saved TFLite FP16: {size_tflite_fp16:.2f} MB ({size_tflite_fp16/size_keras*100:.0f}% of original)")

# Verify quantized model accuracy
print("Validating quantized model...")
interpreter_fp16 = tf.lite.Interpreter(model_path='workout_model_fp16.tflite')
interpreter_fp16.allocate_tensors()

input_details_fp16 = interpreter_fp16.get_input_details()
output_details_fp16 = interpreter_fp16.get_output_details()

preds_fp16 = []
for i in range(len(X_val)):
    interpreter_fp16.set_tensor(input_details_fp16[0]['index'], X_val[i:i+1])
    interpreter_fp16.invoke()
    output = interpreter_fp16.get_tensor(output_details_fp16[0]['index'])
    preds_fp16.append(np.argmax(output))

preds_fp16 = np.array(preds_fp16)
acc_fp16 = (preds_fp16 == y_val).mean()
print(f"TFLite FP16 validation accuracy: {acc_fp16:.4f}")

# Compare all model variants side by side
print("\nGenerating comparison plots...")
cm_tflite = confusion_matrix(y_val, preds_tflite)
cm_fp16 = confusion_matrix(y_val, preds_fp16)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

disp_fp32 = ConfusionMatrixDisplay(cm, display_labels=workouts)
disp_fp32.plot(cmap='Blues', xticks_rotation=45, ax=ax1)
ax1.set_title("Original Keras Model")

disp_tflite = ConfusionMatrixDisplay(cm_tflite, display_labels=workouts)
disp_tflite.plot(cmap='Greens', xticks_rotation=45, ax=ax2)
ax2.set_title("TFLite FP32")

disp_fp16_cm = ConfusionMatrixDisplay(cm_fp16, display_labels=workouts)
disp_fp16_cm.plot(cmap='Oranges', xticks_rotation=45, ax=ax3)
ax3.set_title("TFLite FP16 Quantized")

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("MODEL EXPORT COMPLETE")
print("="*50)
print("\nSaved models:")
print("  • workout_model.keras - Full Keras model")
print("  • workout_model.tflite - TFLite FP32")
print("  • workout_model_fp16.tflite - TFLite FP16 quantized")
print("\nAll models use pure TFLite ops and are ready for edge deployment!")
print("="*50)

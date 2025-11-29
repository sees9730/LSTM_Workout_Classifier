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
def load_workout_data(data_dir="TrainingDataEAI", window_sec=2, sample_rate=100):
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

            # Create sliding windows from the session
            num_windows = len(features) // window_size
            for i in range(num_windows):
                start = i * window_size
                end = start + window_size
                if end <= len(features):
                    sequences.append(features[start:end])
                    labels.append(label)

    return sequences, labels

# Load the workout data
print("Loading workout data...")
seqs, labs = load_workout_data(window_sec=2)
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
feature_names = ["Accel X", "Accel Y", "Accel Z"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

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
def create_workout_model(input_size, hidden_size, num_classes, sequence_length=200):
    model = keras.Sequential([
        layers.Input(shape=(sequence_length, input_size)),

        # Minimal but effective: single separable conv layer
        layers.SeparableConv1D(8, kernel_size=3, activation='relu', padding='same'),
        layers.MaxPooling1D(pool_size=5),
        layers.Dropout(0.2),

        # Global pooling and classification
        layers.GlobalAveragePooling1D(),
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
    model = create_workout_model(3, 8, len(workouts), sequence_length=X.shape[1])
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

# Verify FP16 quantized model accuracy
print("Validating FP16 quantized model...")
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

# Convert to quantized INT8/UINT8 version for maximum compression
print("\nConverting to quantized INT8...")
converter_int8 = tf.lite.TFLiteConverter.from_keras_model(model)
converter_int8.optimizations = [tf.lite.Optimize.DEFAULT]

# Provide representative dataset for INT8 calibration
# This helps determine optimal quantization ranges for each tensor
def representative_dataset():
    for i in range(min(100, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

converter_int8.representative_dataset = representative_dataset

# Force full integer quantization (all ops use int8)
converter_int8.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter_int8.inference_input_type = tf.uint8
converter_int8.inference_output_type = tf.uint8

tflite_model_int8 = converter_int8.convert()

with open('workout_model_int8.tflite', 'wb') as f:
    f.write(tflite_model_int8)

size_tflite_int8 = os.path.getsize('workout_model_int8.tflite') / (1024 * 1024)
print(f"Saved TFLite INT8: {size_tflite_int8:.2f} MB ({size_tflite_int8/size_keras*100:.0f}% of original)")

# Verify INT8 quantized model accuracy
print("Validating INT8 quantized model...")
interpreter_int8 = tf.lite.Interpreter(model_path='workout_model_int8.tflite')
interpreter_int8.allocate_tensors()

input_details_int8 = interpreter_int8.get_input_details()
output_details_int8 = interpreter_int8.get_output_details()

# Get quantization parameters for input
input_scale = input_details_int8[0]['quantization'][0]
input_zero_point = input_details_int8[0]['quantization'][1]

# Get quantization parameters for output
output_scale = output_details_int8[0]['quantization'][0]
output_zero_point = output_details_int8[0]['quantization'][1]

print(f"Input quantization: scale={input_scale:.6f}, zero_point={input_zero_point}")
print(f"Output quantization: scale={output_scale:.6f}, zero_point={output_zero_point}")

preds_int8 = []
for i in range(len(X_val)):
    # Quantize input from float32 to uint8
    input_data = X_val[i:i+1]
    input_quantized = (input_data / input_scale + input_zero_point).astype(np.uint8)

    interpreter_int8.set_tensor(input_details_int8[0]['index'], input_quantized)
    interpreter_int8.invoke()

    # Get quantized output and dequantize
    output_quantized = interpreter_int8.get_tensor(output_details_int8[0]['index'])
    output_dequantized = (output_quantized.astype(np.float32) - output_zero_point) * output_scale

    preds_int8.append(np.argmax(output_dequantized))

preds_int8 = np.array(preds_int8)
acc_int8 = (preds_int8 == y_val).mean()
print(f"TFLite INT8 validation accuracy: {acc_int8:.4f}")

# Retrain if INT8 accuracy drops
if acc_int8 < 1.0:
    print("\n" + "="*50)
    print(f"INT8 accuracy is {acc_int8:.4f} (< 1.0)")
    print("Retraining the model and re-quantizing...")
    print("="*50)

    # Create a fresh model with the same architecture
    retrain_model = create_workout_model(3, 32, len(workouts), sequence_length=X.shape[1])

    # Initialize with the original trained weights
    retrain_model.set_weights(model.get_weights())

    retrain_model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=0.0001, weight_decay=1e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    print("\nFine-tuning model with lower learning rate...")

    # Fine-tune with fewer epochs
    retrain_early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True
    )

    retrain_reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-8
    )

    class RetrainPrintProgress(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}: Train acc {logs['accuracy']:.3f} | Val acc {logs['val_accuracy']:.3f}")

    retrain_history = retrain_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1000,
        batch_size=len(X_train),
        callbacks=[retrain_early_stop, retrain_reduce_lr, RetrainPrintProgress()],
        verbose=0
    )

    retrain_val_acc = max(retrain_history.history['val_accuracy'])
    print(f"\nRetrained validation accuracy: {retrain_val_acc:.4f}")

    # Re-quantize the retrained model to INT8
    print("Re-quantizing retrained model to INT8...")
    converter_retrain = tf.lite.TFLiteConverter.from_keras_model(retrain_model)
    converter_retrain.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_dataset_retrain():
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter_retrain.representative_dataset = representative_dataset_retrain
    converter_retrain.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter_retrain.inference_input_type = tf.uint8
    converter_retrain.inference_output_type = tf.uint8

    tflite_model_retrain = converter_retrain.convert()

    with open('workout_model_int8_retrained.tflite', 'wb') as f:
        f.write(tflite_model_retrain)

    size_tflite_retrain = os.path.getsize('workout_model_int8_retrained.tflite') / (1024 * 1024)
    print(f"Saved TFLite INT8 retrained: {size_tflite_retrain:.2f} MB")

    # Validate retrained INT8 model
    print("Validating retrained INT8 model...")
    interpreter_retrain = tf.lite.Interpreter(model_path='workout_model_int8_retrained.tflite')
    interpreter_retrain.allocate_tensors()

    input_details_retrain = interpreter_retrain.get_input_details()
    output_details_retrain = interpreter_retrain.get_output_details()

    input_scale_retrain = input_details_retrain[0]['quantization'][0]
    input_zero_point_retrain = input_details_retrain[0]['quantization'][1]
    output_scale_retrain = output_details_retrain[0]['quantization'][0]
    output_zero_point_retrain = output_details_retrain[0]['quantization'][1]

    preds_retrain = []
    for i in range(len(X_val)):
        input_data = X_val[i:i+1]
        input_quantized = (input_data / input_scale_retrain + input_zero_point_retrain).astype(np.uint8)

        interpreter_retrain.set_tensor(input_details_retrain[0]['index'], input_quantized)
        interpreter_retrain.invoke()

        output_quantized = interpreter_retrain.get_tensor(output_details_retrain[0]['index'])
        output_dequantized = (output_quantized.astype(np.float32) - output_zero_point_retrain) * output_scale_retrain

        preds_retrain.append(np.argmax(output_dequantized))

    preds_retrain = np.array(preds_retrain)
    acc_retrain = (preds_retrain == y_val).mean()
    print(f"TFLite INT8 retrained validation accuracy: {acc_retrain:.4f}")

    # Update to use retrained model if it's better
    if acc_retrain > acc_int8:
        print(f"\nRetraining improved accuracy: {acc_int8:.4f} -> {acc_retrain:.4f}")
        acc_int8 = acc_retrain
        preds_int8 = preds_retrain
        size_tflite_int8 = size_tflite_retrain
        # Replace original INT8 model with retrained version
        import shutil
        shutil.copy('workout_model_int8_retrained.tflite', 'workout_model_int8.tflite')
        print("Updated workout_model_int8.tflite with retrained model")
    else:
        print(f"\nRetraining did not improve accuracy, keeping original INT8 model")
else:
    print("\nINT8 accuracy is perfect (1.0)")


# Compare all model variants side by side
print("\nGenerating comparison plots...")
cm_tflite = confusion_matrix(y_val, preds_tflite)
cm_fp16 = confusion_matrix(y_val, preds_fp16)
cm_int8 = confusion_matrix(y_val, preds_int8)
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

disp_fp32 = ConfusionMatrixDisplay(cm, display_labels=workouts)
disp_fp32.plot(cmap='Blues', xticks_rotation=45, ax=ax1)
ax1.set_title(f"Original Keras Model\nAcc: {best_val_acc:.4f}")

disp_tflite = ConfusionMatrixDisplay(cm_tflite, display_labels=workouts)
disp_tflite.plot(cmap='Greens', xticks_rotation=45, ax=ax2)
ax2.set_title(f"TFLite FP32\nAcc: {acc_tflite:.4f}")

disp_fp16_cm = ConfusionMatrixDisplay(cm_fp16, display_labels=workouts)
disp_fp16_cm.plot(cmap='Oranges', xticks_rotation=45, ax=ax3)
ax3.set_title(f"TFLite FP16 Quantized\nAcc: {acc_fp16:.4f}")

disp_int8_cm = ConfusionMatrixDisplay(cm_int8, display_labels=workouts)
disp_int8_cm.plot(cmap='Reds', xticks_rotation=45, ax=ax4)
ax4.set_title(f"TFLite INT8 Quantized\nAcc: {acc_int8:.4f}")

plt.tight_layout()
plt.show()

print("\n" + "="*50)
print("MODEL EXPORT COMPLETE")
print("="*50)
print(f"\nModel Parameters: {params:,}")
print("\nSaved models:")
print(f"  • workout_model.keras - Full Keras model ({size_keras:.2f} MB)")
print(f"  • workout_model.tflite - TFLite FP32 ({size_tflite_fp32:.2f} MB)")
print(f"  • workout_model_fp16.tflite - TFLite FP16 quantized ({size_tflite_fp16:.2f} MB)")
print(f"  • workout_model_int8.tflite - TFLite INT8 quantized ({size_tflite_int8:.2f} MB)")
print(f"\nModel Size in KB:")
print(f"  • Keras:  {size_keras * 1024:.2f} KB")
print(f"  • FP32:   {size_tflite_fp32 * 1024:.2f} KB")
print(f"  • FP16:   {size_tflite_fp16 * 1024:.2f} KB")
print(f"  • INT8:   {size_tflite_int8 * 1024:.2f} KB")
print("\nModel Accuracy Comparison:")
print(f"  • Original Keras: {best_val_acc:.4f}")
print(f"  • TFLite FP32:    {acc_tflite:.4f}")
print(f"  • TFLite FP16:    {acc_fp16:.4f}")
print(f"  • TFLite INT8:    {acc_int8:.4f}")
print("\nAll models use pure TFLite ops and are ready for edge deployment!")
print("INT8 model is optimized for STM32 and other microcontrollers.")
print("="*50)

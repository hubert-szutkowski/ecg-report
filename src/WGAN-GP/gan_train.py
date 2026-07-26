import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import sys

from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from tslearn.metrics import dtw

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTICLASS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "MultiClass")
if MULTICLASS_DIR not in sys.path:
    sys.path.insert(0, MULTICLASS_DIR)

from multiclass_preprocessing import get_record_ids, get_multiclass_data


from gan_model import WGANGP, build_generator, build_critic

parser = argparse.ArgumentParser(description="WGAN-GP Training Pipeline for ECG Augmentation")
parser.add_argument("--data-dir", type=str, required=True, help="Path to raw ECG data")
parser.add_argument("--target-class", type=str, required=True, default="V", help="Class label to train the GAN on (e.g., 'V', 'S', or '2')")
parser.add_argument("--selected-samples", type=int, default=20, help="Number of ECG records to load")
parser.add_argument("--window", type=int, default=400, help="Fixed window size around the R-peak") 
parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs (GANs need more!)")
parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--start-learning-rate", type=float, default=5e-5, help="Learning rate for WGAN-GP (default 0.00005)")
parser.add_argument("--latent-dim", type=int, default=100, help="Dimension of the latent noise vector")
args = parser.parse_args()

mlflow.log_params({
    "window_size": args.window,
    "epochs": args.epochs,
    "target_class": args.target_class,
    "selected_samples": args.selected_samples,
    "random_seed": args.random_seed,
    "batch_size": args.batch_size,
    "latent_dim": args.latent_dim,
    "learning_rate": args.start_learning_rate,
    "model_type": "WGAN-GP_1D"
})


class WGANMonitor(tf.keras.callbacks.Callback):
    def __init__(self, real_samples, num_samples=4, latent_dim=100, save_dir='outputs/wgan_samples', class_name=""):
        super().__init__()
        self.num_samples = num_samples
        self.latent_dim = latent_dim
        self.save_dir = save_dir
        self.class_name = class_name
        
        # Pick fixed real samples
        np.random.seed(42)
        idx = np.random.choice(len(real_samples), self.num_samples, replace=False)
        self.real_samples = real_samples[idx]
        
        # Build fixed latent vectors
        self.fixed_latent_vectors = tf.random.normal(shape=(num_samples, latent_dim))
        
        os.makedirs(self.save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % 10 == 0:
            # Generate samples
            generated_signals = self.model.generator(self.fixed_latent_vectors, training=False).numpy()
            
            # Draw a 2 x num_samples grid
            fig, axes = plt.subplots(2, self.num_samples, figsize=(15, 6))
            fig.suptitle(f'ECG comparison (Class: {self.class_name}) - Epoch {epoch + 1}', fontsize=14)
            
            for i in range(self.num_samples):
                # Top row: real ECG
                axes[0, i].plot(self.real_samples[i, :, 0], color='green')
                axes[0, i].set_title(f"Real {i+1}")
                axes[0, i].set_ylim([-4, 4]) 
                axes[0, i].grid(True, linestyle='--', alpha=0.6)
                
                # Bottom row: generated ECG
                axes[1, i].plot(generated_signals[i, :, 0], color='blue')
                axes[1, i].set_title(f"Generated {i+1}")
                axes[1, i].set_ylim([-4, 4]) 
                axes[1, i].grid(True, linestyle='--', alpha=0.6)
                
            plt.tight_layout()
            
            filepath = os.path.join(self.save_dir, f'epoch_{epoch+1:03d}.png')
            plt.savefig(filepath)
            plt.close(fig)


class MLflowWGANCallback(tf.keras.callbacks.Callback):
    """Log WGAN metrics to MLflow."""
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metric("d_loss", logs.get("d_loss", 0), step=epoch)
            mlflow.log_metric("g_loss", logs.get("g_loss", 0), step=epoch)


class EpochWeightsSaver(tf.keras.callbacks.Callback):
    def __init__(self, filepath, interval=50):
        super().__init__()
        self.filepath = filepath
        self.interval = interval
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            # separate paths for generator and critic weights
            gen_path = self.filepath.format(epoch=epoch + 1).replace('.weights.h5', '_generator.weights.h5')
            crit_path = self.filepath.format(epoch=epoch + 1).replace('.weights.h5', '_critic.weights.h5')
            
            # Save weights separately
            self.model.generator.save_weights(gen_path)
            self.model.critic.save_weights(crit_path)
            
            print(f"\n[Checkpoint] Saved (Generator, Critic) weights for epoch {epoch + 1}")

class ECGSampleSaver(tf.keras.callbacks.Callback):
    def __init__(self, latent_dim, output_dir="checkpoints/plots", interval=50):
        super().__init__()
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.interval = interval
        self.fixed_noise = tf.random.normal([4, self.latent_dim]) 
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            generated_signals = self.model.generator(self.fixed_noise, training=False)
            generated_signals = generated_signals.numpy()
            
            fig, axs = plt.subplots(4, 1, figsize=(10, 8))
            for i in range(4):
                axs[i].plot(generated_signals[i, :, 0], color='blue')
                axs[i].set_ylim([-3, 3]) 
            
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/epoch_{epoch+1}.png")
            plt.close()

def plot_wgan_losses(history, target_class):
    """Plot generator and critic loss."""
    d_loss = history.history['d_loss']
    g_loss = history.history['g_loss']
    epochs = range(1, len(d_loss) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, d_loss, 'b-', label='Critic Loss (d_loss)')
    plt.plot(epochs, g_loss, 'r-', label='Generator Loss (g_loss)')
    plt.title(f'WGAN-GP Learning Curve - Class {target_class}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'outputs/wgan_loss_curve_class_{target_class}.png')
    plt.close()

def evaluate_distribution_pca(real_signals, fake_signals):
    
    real_flat = real_signals.reshape(real_signals.shape[0], -1)
    fake_flat = fake_signals.reshape(fake_signals.shape[0], -1)
    
    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_flat)
    fake_pca = pca.transform(fake_flat) 
    
    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real')
    plt.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5, label='Generated')
    plt.legend()
    plt.title("PCA: Real vs Generated ECG")
    plt.savefig("outputs/pca_evaluation.png")
    plt.close()

def calculate_mean_dtw(real_signals, fake_signals, n_samples=100):
    """
    Computing mean DTW distance between random pairs of real and generated signals.
    """
    distances = []
    for i in range(n_samples):

        idx_real = np.random.randint(0, real_signals.shape[0])
        idx_fake = np.random.randint(0, fake_signals.shape[0])
        d = dtw(real_signals[idx_real, :, 0], fake_signals[idx_fake, :, 0])
        distances.append(d)
        
    return np.mean(distances)  

def load_all_multiclass_data(data_dir: str, num_records: int, random_seed: int):
    # Load data with IQR downsampling.
    X_all, y_all, groups_all = [], [], []
    record_ids = get_record_ids(data_dir)
    records_to_process = min(num_records, len(record_ids))
    
    for i in range(records_to_process):
        X_record, y_record = get_multiclass_data(data_dir, sample_select=i)
        if len(X_record) > 0:
            X_all.append(X_record)
            y_all.append(y_record)
            groups_all.extend([record_ids[i]] * len(X_record))
            
    X_master = np.vstack(X_all) if X_all else np.array([])
    y_master = np.concatenate(y_all) if y_all else np.array([])
    groups_master = np.array(groups_all)
    
    if len(y_master) == 0:
        return X_master, y_master, groups_master

    rng = np.random.default_rng(random_seed)
    valid_indices = []
    
    for cls in np.unique(y_master):
        cls_mask = (y_master == cls)
        patients_with_cls = groups_master[cls_mask]
        unique_patients, counts = np.unique(patients_with_cls, return_counts=True)
        
        q1 = np.percentile(counts, 25)
        q3 = np.percentile(counts, 75)
        iqr = q3 - q1
        
        dynamic_limit = int(np.ceil(q3 + 1.5 * iqr))
        dynamic_limit = max(dynamic_limit, 30) 
        
        for patient in unique_patients:
            patient_cls_idx = np.where((y_master == cls) & (groups_master == patient))[0]
            if len(patient_cls_idx) > dynamic_limit:
                patient_cls_idx = rng.choice(patient_cls_idx, size=dynamic_limit, replace=False)
            valid_indices.extend(patient_cls_idx)

    valid_indices = np.sort(valid_indices)
    return X_master[valid_indices], y_master[valid_indices], groups_master[valid_indices]


os.makedirs('outputs/wgan_samples', exist_ok=True)

# Load and parse data
X_DATA, Y_LABELS, GROUPS = load_all_multiclass_data(args.data_dir, args.selected_samples, args.random_seed)
print(X_DATA.shape, Y_LABELS.shape, GROUPS.shape)

# Align label types
encoder = LabelEncoder()
Y_ENCODED = encoder.fit_transform(Y_LABELS)
joblib.dump(encoder, 'outputs/label_encoder_wgan.pkl')

print(f"\nTotal extracted windows in dataset: {len(X_DATA)}")
print(f"Detected classes: {encoder.classes_}")

# Filter the target class
try:
    # Resolve class index safely
    if args.target_class.isdigit():
        target_encoded_idx = int(args.target_class)
    else:
        target_encoded_idx = encoder.transform([args.target_class])[0]
except ValueError:
    raise ValueError(f"Class '{args.target_class}' not found in dataset classes: {encoder.classes_}")

target_mask = (Y_ENCODED == target_encoded_idx)
X_minority = X_DATA[target_mask]
print(f"Isolated {len(X_minority)} samples for target class: {args.target_class} (Encoded: {target_encoded_idx})")

if len(X_minority) == 0:
    raise ValueError("No samples found for the target class!")

# Scale the data
scaler = StandardScaler()
X_minority_flattened = X_minority.reshape(-1,1)
X_minority_scaled = scaler.fit_transform(X_minority_flattened)
joblib.dump(scaler, f'outputs/wgan_scaler_class_{args.target_class}.pkl')

# Reshape for conv layers
X_dataset_w = X_minority_scaled.reshape(-1, args.window, 1).astype(np.float32)

# Build models
print("Building Generator and Critic...")
generator = build_generator(latent_dim=args.latent_dim)
critic = build_critic(input_shape=(args.window, 1))

wgan = WGANGP(
    generator=generator, 
    critic=critic, 
    latent_dim=args.latent_dim,
    d_steps=5,
    gp_weight=10.0
)

# Use WGAN Adam settings
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=args.start_learning_rate, beta_1=0.0, beta_2=0.9)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=3*args.start_learning_rate, beta_1=0.0, beta_2=0.9)

wgan.compile(
    d_optimizer=critic_optimizer,
    g_optimizer=generator_optimizer
)


# Configure callbacks
callbacks = [
    # Save sample plots
    WGANMonitor(num_samples=4, latent_dim=args.latent_dim, save_dir='outputs/wgan_samples', class_name=args.target_class, real_samples=X_dataset_w),
    MLflowWGANCallback(),
    ECGSampleSaver(latent_dim=args.latent_dim, output_dir='outputs/wgan_samples', interval=50),
    EpochWeightsSaver(filepath="checkpoints/weights/wgan_epoch_{epoch:03d}.weights.h5", interval=50)
]

print(f"\n{'='*50}")
print(f"STARTING WGAN-GP TRAINING FOR CLASS: {args.target_class}")
print(f"{'='*50}")

history = wgan.fit(
    X_dataset_w,
    batch_size=args.batch_size,
    epochs=args.epochs,
    callbacks=callbacks,
    verbose=1
)

test_noise = np.random.normal(0, 1, size=(1000, args.latent_dim))
X_synthetic = wgan.generator.predict(test_noise, batch_size=args.batch_size)


print("Computing evaluation metrics for generated samples...")

# PCA evaluation
evaluate_distribution_pca(X_dataset_w, X_synthetic)

# DTW evaluation
mean_dtw = calculate_mean_dtw(X_dataset_w, X_synthetic, n_samples=100)

# Register metrics in MLflow
mlflow.log_metric("final_mean_dtw", mean_dtw)

print(f"Final mean DTW: {mean_dtw:.4f}")

# Save outputs
plot_wgan_losses(history, args.target_class)

# Save generator only
generator_path = f"./outputs/wgan_generator_class_{args.target_class}.keras"
wgan.generator.save(generator_path)

print(f"\n{'='*50}")
print("WGAN TRAINING COMPLETE")
print(f"Generator saved to: {generator_path}")
print(f"Scaler saved to: outputs/wgan_scaler_class_{args.target_class}.pkl")
print(f"View loss curves in 'outputs/' and generated samples in 'outputs/wgan_samples/'")
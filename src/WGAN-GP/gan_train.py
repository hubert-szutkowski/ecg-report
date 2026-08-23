import os
import csv
import argparse
import numpy as np
import pandas as pd
import matplotlib
if os.environ.get("WGAN_INTERACTIVE_MONITOR") == "1":
    matplotlib.use("TkAgg")
else:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import joblib
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import sys
import time

from itertools import combinations
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from tslearn.metrics import dtw
from fastdtw import fastdtw
from dtaidistance import dtw

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MULTICLASS_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "MultiClass")
if MULTICLASS_DIR not in sys.path:
    sys.path.insert(0, MULTICLASS_DIR)

from multiclass_preprocessing import get_record_ids, get_multiclass_data, get_global_window_size


from gan_model import WGANGP, build_generator, build_critic

N_FOLD_SPLITS = 5


def _prepare(signals: np.ndarray) -> np.ndarray:
    return np.asarray([s.squeeze().astype(np.float64) for s in signals])


def dtw_within_set(signals: np.ndarray, max_samples: int = 100, window: int | None = 20, random_state: int = 42) -> dict:
    """
    Mean/median/std of pairwise DTW within one set (real-real or synthetic-synthetic).

    Parameters:
        - signals: np.ndarray of shape (n_samples, n_timesteps, 1)
        - max_samples: int, maximum number of samples to consider for pairwise DTW
        - window: int or None, the Sakoe-Chiba window size for DTW
        - random_state: seed for the subsample drawn when len(signals) > max_samples. Seeded by
          default because this function's output is compared ACROSS runs (diversity_ratio, and
          the dtw_diversity_ratio metric a HyperDrive sweep selects on). With an unseeded draw,
          the same unchanged real data gave 5.87 and 7.67 on two runs of class S - a 31% swing
          that is pure subsampling noise, large enough to swamp real differences between runs.
          Sets smaller than max_samples are unaffected either way (class F, 72 windows, was
          already reproducible to 15 decimal places).
    Returns:
        - dict with keys: mean, median, std, n_pairs
    """
    signals = _prepare(signals)
    if len(signals) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(signals), max_samples, replace=False)
        signals = signals[idx]

    dist_matrix = dtw.distance_matrix_fast(signals, window=window, parallel=True)
    upper_triangle = dist_matrix[np.triu_indices(len(signals), k=1)]

    return {
        "mean": float(np.mean(upper_triangle)),
        "median": float(np.median(upper_triangle)),
        "std": float(np.std(upper_triangle)),
        "n_pairs": len(upper_triangle),
    }


def dtw_cross_set(real_signals: np.ndarray, synthetic_signals: np.ndarray, max_samples: int = 100, window: int | None = 20, random_state: int = 42) -> dict:
    """
    Mean/median/std of pairwise DTW between two different sets (real vs synthetic).

    Parameters:
        - real_signals: np.ndarray of shape (n_real_samples, n_timesteps, 1)
        - synthetic_signals: np.ndarray of shape (n_synthetic_samples, n_timesteps, 1)
        - max_samples: int, maximum number of samples to consider for pairwise DTW
        - window: int or None, the Sakoe-Chiba window size for DTW
        - random_state: seed for the subsamples drawn when a set exceeds max_samples - see
          dtw_within_set's docstring for why this must not be left unseeded
    Returns:
        - dict with keys: mean, median, std, n_pairs
    """
    real = _prepare(real_signals)
    synth = _prepare(synthetic_signals)

    rng = np.random.default_rng(random_state)
    if len(real) > max_samples:
        real = real[rng.choice(len(real), max_samples, replace=False)]
    if len(synth) > max_samples:
        synth = synth[rng.choice(len(synth), max_samples, replace=False)]

    combined = np.vstack([real, synth])
    n_real = len(real)

    full_matrix = dtw.distance_matrix_fast(combined, window=window, parallel=True)
    cross_block = full_matrix[:n_real, n_real:]  # real rows x synthetic columns

    return {
        "mean": float(np.mean(cross_block)),
        "median": float(np.median(cross_block)),
        "std": float(np.std(cross_block)),
        "n_pairs": cross_block.size,
    }


class InteractiveWGANMonitor(tf.keras.callbacks.Callback):
    """Display a live ECG dashboard during WGAN-GP training."""

    def __init__(self, real_samples, latent_dim, every_n_epochs=50,
                 num_samples=4, class_name="", metric_max_samples=100,
                 dtw_window=20):
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1")

        self.every_n_epochs = every_n_epochs
        self.latent_dim = latent_dim
        self.class_name = class_name
        self.metric_max_samples = metric_max_samples
        self.dtw_window = dtw_window
        self.start_time = None
        self.loss_history = {"g_loss": [], "d_loss": []}
        self.metric_history = []

        rng = np.random.default_rng(42)
        self.num_samples = min(num_samples, len(real_samples))
        real_indices = rng.choice(len(real_samples), self.num_samples, replace=False)
        self.real_samples = real_samples[real_indices]
        self.fixed_noise = tf.random.normal(shape=(1, latent_dim), seed=42)

        self.figure, self.axes = plt.subplots(2, 3, figsize=(16, 8), num="WGAN-GP ECG monitor")
        self.status_text = self.figure.text(0.5, 0.015, "Status: oczekiwanie na start...", ha="center", fontsize=11)
        plt.ion()
        self.figure.show()

    def on_train_begin(self, logs=None):
        self.start_time = time.perf_counter()
        self.total_epochs = self.params.get("epochs")

    def _elapsed_seconds(self):
        return time.perf_counter() - self.start_time if self.start_time else 0.0

    def _update_status(self, epoch):
        current_epoch = epoch + 1
        total_epochs = self.total_epochs or current_epoch
        progress = min(current_epoch / total_epochs, 1.0)
        bar_width = 30
        filled = int(bar_width * progress)
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = self._elapsed_seconds()
        remaining = elapsed * (1.0 - progress) / progress if progress > 0 else 0.0
        status = "zakonczony" if current_epoch >= total_epochs else "trening"
        self.status_text.set_text(
            f"Epoki: {current_epoch}/{total_epochs} [{bar}] {progress * 100:.1f}% | "
            f"Czas: {elapsed:.0f} s | Pozostalo: {remaining:.0f} s | Status: {status}"
        )

    def _update_dashboard(self, epoch, generated_samples, metrics, generated_fixed=None):
        axes = self.axes.ravel()
        for axis in axes:
            axis.clear()

        fixed_sample = generated_samples[0] if generated_fixed is None else generated_fixed[0]
        axes[0].plot(fixed_sample[:, 0], color="tab:blue")
        axes[0].set_title("Aktualny synthetic beat (fixed noise)")
        axes[1].plot(np.mean(self.real_samples[:, :, 0], axis=0), color="tab:green")
        axes[1].set_title("Sredni real beat")
        axes[2].plot(np.mean(generated_samples[:, :, 0], axis=0), color="tab:blue")
        axes[2].set_title("Sredni synthetic beat")

        epochs = np.arange(1, len(self.loss_history["g_loss"]) + 1)
        axes[3].plot(epochs, self.loss_history["g_loss"], color="tab:red", label="g_loss")
        axes[3].set_title("Generator loss")
        axes[3].set_xlabel("Epoka")
        axes[3].legend()
        axes[4].plot(epochs, self.loss_history["d_loss"], color="tab:orange", label="d_loss")
        axes[4].set_title("Critic loss")
        axes[4].set_xlabel("Epoka")
        axes[4].legend()

        axes[5].axis("off")
        axes[5].text(
            0.02, 0.95,
            f"Epoka: {epoch + 1}\n"
            f"Czas treningu: {self._elapsed_seconds():.1f} s\n\n"
            f"DTW real-real: {metrics['dtw_real_real']:.3f}\n"
            f"DTW synth-synth: {metrics['dtw_synth_synth']:.3f}\n"
            f"DTW real-synth: {metrics['dtw_real_synth']:.3f}\n"
            f"Diversity ratio: {metrics['diversity_ratio']:.3f}",
            va="top", fontsize=12,
        )
        self._update_status(epoch)

        for axis in axes[:5]:
            axis.grid(True, linestyle="--", alpha=0.5)
        for axis in axes[:3]:
            axis.set_ylim([-4, 4])
        self.figure.suptitle(f"WGAN-GP | klasa {self.class_name} | epoka {epoch + 1}")
        self.figure.tight_layout(rect=[0, 0.055, 1, 0.95])
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()
        plt.pause(0.001)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.loss_history["g_loss"].append(float(logs.get("g_loss", np.nan)))
        self.loss_history["d_loss"].append(float(logs.get("d_loss", np.nan)))
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        generated_fixed = self.model.generator(self.fixed_noise, training=False).numpy()
        metric_noise = tf.random.normal(shape=(self.num_samples, self.latent_dim))
        generated_samples = self.model.generator(metric_noise, training=False).numpy()
        real_stats = dtw_within_set(self.real_samples, max_samples=self.metric_max_samples, window=self.dtw_window)
        synth_stats = dtw_within_set(generated_samples, max_samples=self.metric_max_samples, window=self.dtw_window)
        cross_stats = dtw_cross_set(self.real_samples, generated_samples, max_samples=self.metric_max_samples, window=self.dtw_window)
        metrics = {
            "dtw_real_real": real_stats["mean"],
            "dtw_synth_synth": synth_stats["mean"],
            "dtw_real_synth": cross_stats["mean"],
            "diversity_ratio": synth_stats["mean"] / (real_stats["mean"] + 1e-8),
        }
        self.metric_history.append(metrics)
        self._update_dashboard(epoch, generated_samples, metrics, generated_fixed=generated_fixed)

    def on_train_end(self, logs=None):
        print(f"[GAN] Interactive monitor: operation time: {self._elapsed_seconds():.2f} seconds")
        self.figure.canvas.draw_idle()
        self.figure.canvas.flush_events()


class SavedSamplesCallback(tf.keras.callbacks.Callback):
    """Save four fixed real beats once and four generated beats every N epochs."""

    def __init__(self, real_samples, latent_dim, output_dir, every_n_epochs=100,
                 num_samples=4, class_name=""):
        super().__init__()
        if every_n_epochs < 1:
            raise ValueError("every_n_epochs must be at least 1")

        self.every_n_epochs = every_n_epochs
        self.latent_dim = latent_dim
        self.output_dir = output_dir
        self.class_name = class_name
        self.num_samples = min(num_samples, len(real_samples))
        rng = np.random.default_rng(42)
        self.real_samples = real_samples[rng.choice(len(real_samples), self.num_samples, replace=False)]
        self.fixed_noise = tf.random.normal(shape=(self.num_samples, latent_dim), seed=42)
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_samples(self, samples, path, title, color):
        figure = Figure(figsize=(10, 2.2 * self.num_samples))
        FigureCanvasAgg(figure)
        axes = figure.subplots(self.num_samples, 1, squeeze=False)
        for axis, sample in zip(axes.ravel(), samples):
            axis.plot(sample[:, 0], color=color)
            axis.set_xlabel("Sample index in ECG beat")
            axis.set_ylabel("Amplitude")
            axis.grid(True, alpha=0.3)
        figure.suptitle(title)
        figure.tight_layout()
        figure.savefig(path, dpi=150, bbox_inches="tight")

    def on_train_begin(self, logs=None):
        path = os.path.join(self.output_dir, f"real_samples_{self.class_name}.png")
        self._save_samples(
            self.real_samples,
            path,
            f"Real ECG beats | class {self.class_name}",
            "tab:green",
        )
        print(f"[Samples] Saved 4 real beats to {path}")

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        if current_epoch % self.every_n_epochs != 0:
            return

        generated_samples = self.model.generator(self.fixed_noise, training=False).numpy()
        prefix = f"{self.class_name}_epoch_{current_epoch:04d}"
        self._save_samples(
            generated_samples,
            os.path.join(self.output_dir, f"synthetic_samples_{prefix}.png"),
            f"Synthetic ECG beats | class {self.class_name} | epoch {current_epoch}",
            "tab:blue",
        )
        print(f"[Samples] Saved 4 synthetic beats for epoch {current_epoch}")


class MLflowWGANCallback(tf.keras.callbacks.Callback):
    """Log WGAN metrics to MLflow, namespaced so parallel per-class/per-fold runs don't collide."""
    def __init__(self, metric_prefix=""):
        super().__init__()
        self.metric_prefix = metric_prefix

    def on_epoch_end(self, epoch, logs=None):
        if logs:
            mlflow.log_metric(f"{self.metric_prefix}d_loss", logs.get("d_loss", 0), step=epoch)
            mlflow.log_metric(f"{self.metric_prefix}g_loss", logs.get("g_loss", 0), step=epoch)


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


class LearningRateDecayCallback(tf.keras.callbacks.Callback):
    def __init__(self, decay_epoch=5000, decay_factor=10.0):
        super().__init__()
        self.decay_epoch = decay_epoch
        self.decay_factor = decay_factor
        self.has_decayed = False

    def on_epoch_end(self, epoch, logs=None):
        if self.has_decayed or epoch + 1 != self.decay_epoch:
            return

        generator_lr = self.model.g_optimizer.learning_rate / self.decay_factor
        critic_lr = self.model.d_optimizer.learning_rate / self.decay_factor
        self.model.g_optimizer.learning_rate.assign(generator_lr)
        self.model.d_optimizer.learning_rate.assign(critic_lr)
        self.has_decayed = True

        print(
            f"\n[Learning rate] Reduced by {self.decay_factor:g}x after epoch {self.decay_epoch}: "
            f"generator={float(generator_lr):.3e}, critic={float(critic_lr):.3e}"
        )


class DTWDiversityCallback(tf.keras.callbacks.Callback):
    """
    Every `every_n_epochs`, generates synthetic samples from the current
    generator and computes real-real, synth-synth, and real-synth DTW
    distance statistics, logging them to MLflow.
    """

    def __init__(self, real_signals: np.ndarray, latent_dim: int, every_n_epochs: int = 10,
                 n_generate: int = 100, max_samples: int = 100, window: int | None = 20, metric_prefix: str = ""):
        super().__init__()
        self.real_signals = real_signals
        self.latent_dim = latent_dim
        self.every_n_epochs = every_n_epochs
        self.n_generate = n_generate
        self.max_samples = max_samples
        self.window = window
        self.metric_prefix = metric_prefix
        self.best_ratio = -np.inf

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs != 0:
            return

        z = tf.random.normal(shape=(self.n_generate, self.latent_dim))
        synthetic = self.model.generator(z, training=False).numpy()

        real_stats = dtw_within_set(self.real_signals, max_samples=self.max_samples, window=self.window)
        synth_stats = dtw_within_set(synthetic, max_samples=self.max_samples, window=self.window)
        cross_stats = dtw_cross_set(self.real_signals, synthetic, max_samples=self.max_samples, window=self.window)

        diversity_ratio = synth_stats["mean"] / (real_stats["mean"] + 1e-8)
        self.best_ratio = max(self.best_ratio, diversity_ratio)

        metrics = {
            f"{self.metric_prefix}dtw_real_real_mean": real_stats["mean"],
            f"{self.metric_prefix}dtw_synth_synth_mean": synth_stats["mean"],
            f"{self.metric_prefix}dtw_real_synth_mean": cross_stats["mean"],
            f"{self.metric_prefix}dtw_diversity_ratio": diversity_ratio,
        }
        mlflow.log_metrics(metrics, step=epoch)

        print(
            f"[Epoch {epoch + 1}] real-real={real_stats['mean']:.2f} "
            f"synth-synth={synth_stats['mean']:.2f} real-synth={cross_stats['mean']:.2f} "
            f"diversity_ratio={diversity_ratio:.3f}"
        )

    def on_train_end(self, logs=None):
        if self.best_ratio > -np.inf:
            mlflow.log_metric(f"{self.metric_prefix}best_dtw_diversity_ratio", self.best_ratio)


def plot_wgan_losses(history, target_class, output_dir="outputs"):
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
    plt.savefig(f'{output_dir}/wgan_loss_curve_class_{target_class}.png')
    plt.close()

def evaluate_distribution_pca(real_signals, fake_signals, output_dir="outputs"):

    real_flat = real_signals.reshape(real_signals.shape[0], -1)
    fake_flat = fake_signals.reshape(fake_signals.shape[0], -1)

    pca = PCA(n_components=2)
    real_pca = pca.fit_transform(real_flat)
    fake_pca = pca.transform(fake_flat)

    plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.5, label='Real')
    plt.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.5, label='Generated')
    plt.legend()
    plt.title("PCA: Real vs Generated ECG")
    plt.savefig(f"{output_dir}/pca_evaluation.png")
    plt.close()


def load_all_multiclass_data(data_dir: str, num_records: int, random_seed: int):
    # Load data with IQR downsampling.
    X_all, y_all, groups_all = [], [], []
    record_ids = get_record_ids(data_dir)
    records_to_process = min(num_records, len(record_ids))
    global_distance = get_global_window_size(data_dir, record_ids)
    for i in range(records_to_process):
        X_record, y_record = get_multiclass_data(data_dir, sample_select=i, window_size=global_distance)
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
    return X_master[valid_indices], y_master[valid_indices], groups_master[valid_indices], global_distance


def train_gan_and_generate(
    X_real_class: np.ndarray,
    groups_real_class: np.ndarray,
    n_samples_to_generate: int,
    window_size: int,
    target_class_name: str,
    fold: int,
    epochs: int = 500,
    d_steps: int = 3,
    gp_weight: float = 10.0,
    latent_dim: int = 32,
    batch_size: int = 32,
    critic_lr_multiplier: float = 3.0,
    start_learning_rate: float = 5e-5,
    lr_decay_factor: float = 10.0,
    random_seed: int = 42,
    output_dir: str = "outputs",
    interactive_monitor: bool = False,
) -> np.ndarray:
    """
    Trains a WGAN-GP on real ECG windows of a single AAMI class and returns
    synthetic windows in the original (unscaled) amplitude space, ready to be
    concatenated with real training data.

    X_real_class / groups_real_class must already be restricted to the
    current fold's TRAINING patients only (never pass validation-fold
    patients here) so synthetic samples never leak validation morphology
    into training.

    Parameters:
        - X_real_class: np.ndarray (n, window_size, 1), raw real windows for one class, train-fold patients only
        - groups_real_class: np.ndarray (n,), patient/record id per row in X_real_class
        - n_samples_to_generate: how many synthetic windows to produce
        - window_size, target_class_name, fold: context used for outputs/metrics naming
        - epochs, d_steps, gp_weight, latent_dim, batch_size, critic_lr_multiplier, start_learning_rate: GAN hyperparameters
        - lr_decay_factor: divide both learning rates by this factor after epoch 5000 (5 or 10)
        - random_seed: shuffling/reproducibility seed
        - output_dir: base directory for sample plots and loss curves
    Returns:
        - np.ndarray (n_samples_to_generate, window_size, 1) of synthetic windows, unscaled
    """
    if len(X_real_class) == 0:
        raise ValueError(f"No real samples provided for class '{target_class_name}' (fold {fold}).")

    n_patients = len(np.unique(groups_real_class))
    print(f"[GAN] class={target_class_name} fold={fold}: {len(X_real_class)} windows from {n_patients} unique patients (train-fold only)")

    run_label = f"{target_class_name}_fold{fold}"
    class_dir = os.path.join(output_dir, "wgan_samples", run_label)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_real_class.reshape(-1, 1)).reshape(-1, window_size, 1).astype(np.float32)

    train_dataset = (
        tf.data.Dataset.from_tensor_slices(X_scaled)
        .shuffle(buffer_size=len(X_scaled), seed=random_seed, reshuffle_each_iteration=True)
        .batch(batch_size)
    )

    generator = build_generator(window_size=window_size, latent_dim=latent_dim)
    critic = build_critic(input_shape=(window_size, 1))

    wgan = WGANGP(generator=generator, critic=critic, latent_dim=latent_dim, d_steps=d_steps, gp_weight=gp_weight)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate=start_learning_rate, beta_1=0.0, beta_2=0.9)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr_multiplier * start_learning_rate, beta_1=0.0, beta_2=0.9)
    wgan.compile(d_optimizer=critic_optimizer, g_optimizer=generator_optimizer)

    metric_prefix = f"gan_{run_label}_"
    callbacks = [
        MLflowWGANCallback(metric_prefix=metric_prefix),
        SavedSamplesCallback(
            real_samples=X_scaled,
            latent_dim=latent_dim,
            output_dir=class_dir,
            every_n_epochs=100,
            num_samples=4,
            class_name=target_class_name,
        ),
        EpochWeightsSaver(filepath=os.path.join(class_dir, "wgan_epoch_{epoch:03d}.weights.h5"), interval=50),
        LearningRateDecayCallback(decay_factor=lr_decay_factor),
        DTWDiversityCallback(real_signals=X_scaled, latent_dim=latent_dim, every_n_epochs=10, metric_prefix=metric_prefix),
    ]
    if interactive_monitor:
        callbacks.insert(0, InteractiveWGANMonitor(
            real_samples=X_scaled,
            latent_dim=latent_dim,
            every_n_epochs=5,
            num_samples=4,
            class_name=target_class_name,
        ))

    print(f"\n{'='*50}\nTraining WGAN-GP | class={target_class_name} fold={fold}\n{'='*50}")
    operation_start = time.perf_counter()
    history = wgan.fit(train_dataset, epochs=epochs, callbacks=callbacks, verbose=1)
    plot_wgan_losses(history, run_label, output_dir=output_dir)

    noise = tf.random.normal(shape=(n_samples_to_generate, latent_dim))
    X_synthetic_scaled = wgan.generator.predict(noise, batch_size=batch_size)
    X_synthetic = scaler.inverse_transform(X_synthetic_scaled.reshape(-1, 1)).reshape(-1, window_size, 1).astype(np.float32)

    print(f"[GAN] Generated {len(X_synthetic)} synthetic windows for class {target_class_name} (fold {fold})")
    print(f"[GAN] Operation time: {time.perf_counter() - operation_start:.2f} seconds")
    return X_synthetic


def build_parser():
    parser = argparse.ArgumentParser(description="WGAN-GP Training Pipeline for ECG Augmentation")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to raw ECG data")
    parser.add_argument("--aami-class", type=str, required=True, help="AAMI class label to train the GAN on (e.g., 'S', 'V', 'F', 'Q')")
    parser.add_argument("--fold", type=int, required=True, help=f"Fold index (0-{N_FOLD_SPLITS - 1}) from StratifiedGroupKFold to train on")
    parser.add_argument("--selected-samples", type=int, default=20, help="Number of ECG records to load")
    parser.add_argument("--n-generate", type=int, default=1000, help="Number of synthetic samples to generate")
    parser.add_argument("--epochs", type=int, default=500, help="Number of training epochs (GANs need more!)")
    parser.add_argument("--d-steps", type=int, default=3, help="Number of critic updates per generator update")
    parser.add_argument("--gp-weight", type=float, default=10.0, help="Gradient penalty weight")
    parser.add_argument("--latent-dim", type=int, default=32, help="Dimension of the latent noise vector")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--critic-lr-multiplier", type=float, default=3.0, help="Critic learning rate as a multiple of --start-learning-rate")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--start-learning-rate", type=float, default=5e-5, help="Generator learning rate for WGAN-GP (default 0.00005)")
    parser.add_argument("--lr-decay-factor", type=float, choices=[5.0, 10.0], default=10.0,
                        help="Divide both learning rates by this factor after epoch 5000 (default: 10)")
    return parser


def main():
    args = build_parser().parse_args()

    mlflow.log_params({
        "epochs": args.epochs,
        "aami_class": args.aami_class,
        "fold": args.fold,
        "selected_samples": args.selected_samples,
        "random_seed": args.random_seed,
        "d_steps": args.d_steps,
        "gp_weight": args.gp_weight,
        "batch_size": args.batch_size,
        "latent_dim": args.latent_dim,
        "learning_rate": args.start_learning_rate,
        "lr_decay_epoch": 5000,
        "lr_decay_factor": args.lr_decay_factor,
        "critic_lr_multiplier": args.critic_lr_multiplier,
        "model_type": "WGAN-GP_1D"
    })

    os.makedirs('outputs/wgan_samples', exist_ok=True)

    # Load and parse data
    X_DATA, Y_LABELS, GROUPS, window_size = load_all_multiclass_data(args.data_dir, args.selected_samples, args.random_seed)
    print(X_DATA.shape, Y_LABELS.shape, GROUPS.shape)
    mlflow.log_metrics({"window_size": window_size})

    # Align label types
    encoder = LabelEncoder()
    Y_ENCODED = encoder.fit_transform(Y_LABELS)
    joblib.dump(encoder, 'outputs/label_encoder_wgan.pkl')

    print(f"\nTotal extracted windows in dataset: {len(X_DATA)}")
    print(f"Detected classes: {encoder.classes_}")
    print(f"Window size used: {window_size}")

    # Split into folds (StratifiedGroupKFold, grouped by patient/record)
    if not (0 <= args.fold < N_FOLD_SPLITS):
        raise ValueError(f"--fold must be in [0, {N_FOLD_SPLITS}), got {args.fold}")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLD_SPLITS, shuffle=True, random_state=args.random_seed)
    fold_splits = list(sgkf.split(X_DATA, Y_ENCODED, GROUPS))
    train_idx, _ = fold_splits[args.fold]

    X_fold_train = X_DATA[train_idx]
    Y_fold_train = Y_ENCODED[train_idx]
    GROUPS_fold_train = GROUPS[train_idx]
    print(f"Fold {args.fold}: {len(train_idx)} windows / {len(np.unique(GROUPS_fold_train))} unique patients in training partition")

    try:
        if args.aami_class.isdigit():
            target_encoded_idx = int(args.aami_class)
        else:
            target_encoded_idx = encoder.transform([args.aami_class])[0]
    except ValueError:
        raise ValueError(f"Class '{args.aami_class}' not found in dataset classes: {encoder.classes_}")

    target_mask = (Y_fold_train == target_encoded_idx)
    X_minority = X_fold_train[target_mask]
    groups_minority = GROUPS_fold_train[target_mask]
    print(f"Isolated {len(X_minority)} samples for AAMI class: {args.aami_class} (Encoded: {target_encoded_idx}, fold {args.fold})")

    if len(X_minority) == 0:
        raise ValueError("No samples found for the target class in this fold!")

    X_synthetic = train_gan_and_generate(
        X_real_class=X_minority,
        groups_real_class=groups_minority,
        n_samples_to_generate=args.n_generate,
        window_size=window_size,
        target_class_name=args.aami_class,
        fold=args.fold,
        epochs=args.epochs,
        d_steps=args.d_steps,
        gp_weight=args.gp_weight,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        critic_lr_multiplier=args.critic_lr_multiplier,
        start_learning_rate=args.start_learning_rate,
        lr_decay_factor=args.lr_decay_factor,
        random_seed=args.random_seed,
    )

    print("Computing evaluation metrics for generated samples...")
    evaluate_distribution_pca(X_minority, X_synthetic)

    np.save(f"outputs/wgan_synthetic_class_{args.aami_class}_fold_{args.fold}.npy", X_synthetic)

    print(f"\n{'='*50}")
    print("WGAN TRAINING COMPLETE")
    print(f"Synthetic samples saved to: outputs/wgan_synthetic_class_{args.aami_class}_fold_{args.fold}.npy")
    print(f"View loss curves and generated samples in 'outputs/'")


if __name__ == "__main__":
    main()

import tensorflow as tf
from tensorflow.keras import layers, models, Model, Sequential
import numpy as np


class MinibatchStdDev(layers.Layer):
    def call(self, inputs):
        mean = tf.reduce_mean(inputs, axis=0, keepdims=True)
        variance = tf.reduce_mean(tf.square(inputs - mean), axis=0, keepdims=True)
        stddev = tf.sqrt(variance + 1e-8)
        mean_stddev = tf.reduce_mean(stddev)
        shape = tf.shape(inputs)
        feature_map = tf.fill([shape[0], shape[1], 1], mean_stddev)
        return tf.concat([inputs, feature_map], axis=-1)



def build_generator(window_size=216, latent_dim=100):
    upsampling_factor = 2 ** 3
    base_length = int(np.ceil(window_size / upsampling_factor))
    generated_length = base_length * upsampling_factor

    z = layers.Input(shape=(latent_dim,))

    x = layers.Dense(base_length * 32)(z)
    x = layers.Reshape((base_length, 32))(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv1DTranspose(64, kernel_size=8, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1DTranspose(32, kernel_size=6, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1DTranspose(16, kernel_size=4, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    if generated_length > window_size:
        crop_size = generated_length - window_size
        x = layers.Cropping1D(cropping=(0, crop_size))(x)

    output = layers.Conv1D(1, kernel_size=3, padding="same", activation="linear")(x)
    return Model(z, output, name="dynamic_slim_generator")


def build_critic(input_shape=(216, 1)):
    inp = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=7, strides=2, padding="same")(inp)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv1D(64, kernel_size=5, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv1D(128, kernel_size=3, strides=2, padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    x = layers.Dropout(0.3)(x)

    out = layers.Dense(1, activation="linear")(x)
    return Model(inp, out, name="slim_critic")

class WGANGP(Model):
    def __init__(self, generator, critic, latent_dim, d_steps=5, gp_weight=10.0):
        super(WGANGP, self).__init__()
        self.generator = generator
        self.critic = critic
        self.latent_dim = latent_dim
        self.d_steps = d_steps 
        self.gp_weight = gp_weight 

    def compile(self, d_optimizer, g_optimizer):
        super(WGANGP, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        
        self.d_loss_metric = tf.keras.metrics.Mean(name="d_loss")
        self.g_loss_metric = tf.keras.metrics.Mean(name="g_loss")

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]

    def gradient_penalty(self, batch_size, real_samples, fake_samples):
        alpha = tf.random.normal([batch_size, 1, 1], 0.0, 1.0)
        diff = fake_samples - real_samples
        interpolated = real_samples + alpha * diff

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated)
            pred = self.critic(interpolated, training=True)

        grads = gp_tape.gradient(pred, [interpolated])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2]))
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    def train_step(self, real_samples):
        if isinstance(real_samples, tuple):
            real_samples = real_samples[0]
            
        batch_size = tf.shape(real_samples)[0]

        for _ in range(self.d_steps):
            random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
            
            with tf.GradientTape() as tape:
                fake_samples = self.generator(random_latent_vectors, training=True)
                
                fake_logits = self.critic(fake_samples, training=True)
                real_logits = self.critic(real_samples, training=True)

                c_wass_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)
                c_gp = self.gradient_penalty(batch_size, real_samples, fake_samples)
                c_loss = c_wass_loss + c_gp * self.gp_weight

            c_gradient = tape.gradient(c_loss, self.critic.trainable_variables)
            self.d_optimizer.apply_gradients(zip(c_gradient, self.critic.trainable_variables))

        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        with tf.GradientTape() as tape:
            fake_samples = self.generator(random_latent_vectors, training=True)
            gen_logits = self.critic(fake_samples, training=True)

            g_loss = -tf.reduce_mean(gen_logits)

        gen_gradient = tape.gradient(g_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_gradient, self.generator.trainable_variables))

        self.d_loss_metric.update_state(c_loss)
        self.g_loss_metric.update_state(g_loss)
        
        return {
            "d_loss": self.d_loss_metric.result(),
            "g_loss": self.g_loss_metric.result()
        }
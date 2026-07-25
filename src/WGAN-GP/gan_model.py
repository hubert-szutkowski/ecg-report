import tensorflow as tf
from tensorflow.keras import layers, models, Model, Sequential



def _inception_block(x, filters):

    b1 = layers.Conv1D(filters // 4, 3, padding="same")(x)

    b2 = layers.Conv1D(filters // 4, 7, padding="same")(x)

    b3 = layers.Conv1D(filters // 4, 15, padding="same")(x)

    b4 = layers.Conv1D(filters // 4, 31, padding="same")(x)

    x = layers.Concatenate()([b1, b2, b3, b4])

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    return x


def _transformer_block(x,
                       embed_dim,
                       num_heads=4,
                       ff_dim=None):

    if ff_dim is None:
        ff_dim = embed_dim * 4

    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=embed_dim // num_heads,
    )(x, x)

    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(ff_dim, activation="gelu")(x)
    ff = layers.Dense(embed_dim)(ff)

    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    return x

def build_generator(latent_dim=100):

    z = layers.Input(shape=(latent_dim,))

    x = layers.Dense(50 * 256)(z)
    x = layers.Reshape((50, 256))(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # 50
    x = _inception_block(x, 256)
    x = _transformer_block(x, 256)

    # 100
    x = layers.UpSampling1D(2)(x)

    x = layers.Conv1D(
        128,
        kernel_size=21,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = _inception_block(x, 128)
    x = _transformer_block(x, 128)

    # 200
    x = layers.UpSampling1D(2)(x)

    x = layers.Conv1D(
        64,
        kernel_size=15,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = _inception_block(x, 64)
    x = _transformer_block(x, 64)

    # 400
    x = layers.UpSampling1D(2)(x)

    x = layers.Conv1D(
        32,
        kernel_size=11,
        padding="same"
    )(x)

    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = _inception_block(x, 32)

    output = layers.Conv1D(
        1,
        kernel_size=11,
        padding="same",
        activation="linear"
    )(x)

    return Model(z, output, name="generator")

def build_critic(input_shape=(400, 1)):

    inp = layers.Input(shape=input_shape)

    # 400 -> 200

    x = _inception_block(inp, 64)

    x = layers.Conv1D(
        64,
        kernel_size=31,
        strides=2,
        padding="same"
    )(x)

    x = layers.LeakyReLU(0.2)(x)

    # 200 -> 100

    x = _inception_block(x, 128)

    x = layers.Conv1D(
        128,
        kernel_size=21,
        strides=2,
        padding="same"
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = _transformer_block(x, 128)

    # 100 -> 50

    x = _inception_block(x, 256)

    x = layers.Conv1D(
        256,
        kernel_size=15,
        strides=2,
        padding="same"
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = _transformer_block(x, 256)

    # 50 -> 25

    x = layers.Conv1D(
        512,
        kernel_size=9,
        strides=2,
        padding="same"
    )(x)

    x = layers.LayerNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.GlobalAveragePooling1D()(x)

    out = layers.Dense(
        1,
        activation="linear"
    )(x)

    return Model(inp, out, name="critic")

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
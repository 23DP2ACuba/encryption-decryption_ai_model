import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# Clear any previous session to avoid variable conflicts
tf.keras.backend.clear_session()

# ------------- config -------------
plaintext_size = 16
key_size = 16
batch_size = 512
# ----------------------------------

plaintexts = tf.random.uniform((batch_size, plaintext_size), minval=65, maxval=91, dtype=tf.int32)
keys = tf.random.uniform((batch_size, key_size), minval=65, maxval=91, dtype=tf.int32)

def encryption_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(plaintext_size + key_size,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(plaintext_size, activation="tanh"),
    ])
    return model

def decryption_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation="relu", input_shape=(plaintext_size + key_size,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(plaintext_size, activation="tanh"),
    ])
    return model

encr = encryption_model()
dcr1 = decryption_model()
dcr2 = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=(plaintext_size,)),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(plaintext_size, activation="tanh"),
])

optimizer = tf.keras.optimizers.Adam()

def step(plaintexts, keys):
    with tf.GradientTape(persistent=True) as tape:
        plaintexts = tf.cast(plaintexts, tf.float32)
        keys = tf.cast(keys, tf.float32)

        combined_input = tf.concat([plaintexts, keys], axis=1)
        ciphertexts = encr(combined_input)

        dcr1_input = tf.concat([ciphertexts, keys], axis=1)
        dcr1_output = dcr1(dcr1_input)

        dcr2_output = dcr2(ciphertexts)

        dcr1_loss = tf.reduce_mean(tf.square(plaintexts - dcr1_output))
        dcr2_loss = tf.reduce_mean(tf.square(plaintexts - dcr2_output))

        encr_dcr1_loss = dcr1_loss + (1.0 - dcr2_loss)

    grads_ab = tape.gradient(encr_dcr1_loss, encr.trainable_variables + dcr1.trainable_variables)
    grads_eve = tape.gradient(dcr2_loss, dcr2.trainable_variables)

    optimizer.apply_gradients(zip(grads_ab, encr.trainable_variables + dcr1.trainable_variables))
    optimizer.apply_gradients(zip(grads_eve, dcr2.trainable_variables))

    return dcr1_loss, dcr2_loss

for epoch in range(10000):
    decr1_loss, decr2_loss = step(plaintexts, keys)
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Bob loss = {decr1_loss.numpy():.4f}, Eve loss = {decr2_loss.numpy():.4f}")

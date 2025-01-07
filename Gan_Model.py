""" THIS IS A GAN MODEL FOR CREATING MIMICED DATA """

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle


data = pd.read_csv('ML-EdgeIIoT-dataset.csv', low_memory=False)

drop_columns = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
    "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp",
    "http.request.uri.query", "tcp.options", "tcp.payload", "tcp.srcport",
    "tcp.dstport", "udp.port", "arp.opcode", "mqtt.msg", "icmp.unused",
    "http.tls_port", 'dns.qry.type', 'dns.retransmit_request_in', "mqtt.msg_decoded_as",
    "mbtcp.trans_id", "mbtcp.unit_id", "http.request.method", "http.referer",
    "http.request.version", "dns.qry.name.len", "mqtt.conack.flags",
    "mqtt.protoname", "mqtt.topic"
]
data.drop(drop_columns, axis=1, inplace=True)
data.dropna(inplace=True)

X = data.drop(columns=['Attack_label', 'Attack_type'])
scaler = MinMaxScaler()
X_scaled = shuffle(scaler.fit_transform(X))



def build_generator(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_dim=input_dim),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(input_dim, activation='sigmoid')
    ])
    return model



def build_discriminator(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_dim=input_dim),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    return model



def build_gan(generator, discriminator):
    discriminator.trainable = False
    gan_input = tf.keras.Input(shape=(generator.input_shape[1],))
    generated_data = generator(gan_input)
    gan_output = discriminator(generated_data)
    gan = tf.keras.Model(gan_input, gan_output)
    gan.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='binary_crossentropy')
    return gan



input_dim = X_scaled.shape[1]
generator = build_generator(input_dim)
discriminator = build_discriminator(input_dim)
discriminator.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])
gan = build_gan(generator, discriminator)


epochs = 1000
batch_size = 100
half_batch = batch_size // 2

generator_losses, discriminator_losses = [], []

for epoch in range(epochs):
    
    idx = np.random.randint(0, X_scaled.shape[0], half_batch)
    real_samples = X_scaled[idx]
   
    noise = np.random.normal(0, 1, (half_batch, generator.input_shape[1]))
    fake_samples = generator.predict(noise)
    X_combined = np.vstack((real_samples, fake_samples))
    y_real = np.ones(half_batch) 
    y_fake = np.zeros(half_batch) 
    y_combined = np.hstack((y_real, y_fake))
   
    indices = np.random.permutation(X_combined.shape[0])
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
   
    d_loss = discriminator.train_on_batch(X_combined, y_combined)
    
    
    
    
    
    noise = np.random.normal(0, 1, (batch_size, input_dim))
    misleading_labels = np.ones(batch_size) 
    g_loss = gan.train_on_batch(noise, misleading_labels)

    generator_losses.append(g_loss)
    discriminator_losses.append(d_loss[0])

    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Generator Loss: {g_loss}, Discriminator Loss: {d_loss[0]}")


plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label="Generator Loss")
plt.plot(discriminator_losses, label="Discriminator Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("GAN Losses")
plt.show()

noise = np.random.normal(0, 1, (100, input_dim))
mimiced_sample=generator(noise)



from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from keras import backend as K
from matplotlib import colors as mcolors

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

batch_size = 10

# original_dim = 7*5
original_dim = 8*8

latent_dim = 2
# intermediate_dim = 25
intermediate_dim = 45
epochs = 350
epsilon_std = 0.1

def sampling(args: tuple):
    z_mean, z_log_var = args
    print(z_mean)
    print(z_log_var)
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0.,
                              stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2) * epsilon  # h(z)

x = Input(shape=(original_dim,), name="input")
h = Dense(intermediate_dim, activation='relu', name="encoding")(x)
z_mean = Dense(latent_dim, name="mean")(h)
z_log_var = Dense(latent_dim, name="log-variance")(h)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
encoder = Model(x, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

input_decoder = Input(shape=(latent_dim,), name="decoder_input")
decoder_h = Dense(intermediate_dim, activation='relu', name="decoder_h")(input_decoder)
x_decoded = Dense(original_dim, activation='sigmoid', name="flat_decoded")(decoder_h)
decoder = Model(input_decoder, x_decoded, name="decoder")
decoder.summary()

output_combined = decoder(encoder(x)[2])
vae = Model(x, output_combined)
vae.summary()

import tensorflow as tf
from keras import metrics


def vae_loss(x: tf.Tensor, x_decoded_mean: tf.Tensor):
  xent_loss = original_dim * metrics.binary_crossentropy(x, x_decoded_mean) # x-^X
  kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
  vae_loss = K.mean(xent_loss + kl_loss)
  return vae_loss


vae.compile(loss=vae_loss)
vae.summary()

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
from tp3.loader import CSVLoader


# data_column = ["x" + str(i) for i in range(1, 36)]
# data, exp = CSVLoader.load("../ej1/fonts.csv", False, data_column, None, False)
data_column = ["x" + str(i) for i in range(1, 65)]
data, exp = CSVLoader.load("../ej1/other.csv", False, data_column, None, False)
data = data[:, 1:]
# x_train = data[:32].reshape(32, 7, 5)
print(data.shape)
# x_train = data[:224].reshape(224, 8, 8)
x_train = data[:192].reshape(192, 8, 8)
x_test = data
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)
# y_test = list(colors.values())[:32]
# y_test = list(colors.values())[:224]
y_test = list(colors.values())[:192]

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))


vae.fit(x_train, x_train,
        shuffle=True,
        epochs=epochs,
        verbose=0,
        batch_size=batch_size)

x_test_encoded = encoder.predict(x_test, batch_size=batch_size)[0]
# plt.figure(figsize=(6, 6))
# plt.scatter(x_test_encoded[:,0], x_test_encoded[:,1], c=y_test, cmap='viridis')
# plt.colorbar()
# plt.show()

# y = 7
# x = 5
y = 8
x = 8
n = 15
figure = np.zeros((y * n, x * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(y, x)
        figure[i * y: (i + 1) * y,
               j * x: (j + 1) * x] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()

print(grid_x)

print(z_sample)

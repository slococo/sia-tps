import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers, losses
from keras.models import Model

from tp3.loader import CSVLoader

data_column = ["x" + str(i) for i in range(1, 36)]
data, exp = CSVLoader.load("../ej1/fonts.csv", False, data_column, None, False)
data = data[:, 1:]
x_train = data.reshape(32, 7, 5)
x_test = data.reshape(32, 7, 5)

print(x_train)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
print(x_train)

print(x_train.shape)
print(x_test.shape)

latent_dim = 2


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                layers.Flatten(),
                layers.Dense(25, activation="sigmoid"),
                layers.Dense(17, activation="sigmoid"),
                layers.Dense(latent_dim, activation="sigmoid"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                layers.Dense(17, activation="sigmoid"),
                layers.Dense(25, activation="sigmoid"),
                layers.Dense(35, activation="sigmoid"),
                layers.Reshape((7, 5)),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(latent_dim)

autoencoder.compile(optimizer="adam", loss=losses.MeanSquaredError())

autoencoder.fit(
    x_train,
    x_train,
    epochs=50000,
    shuffle=False,
    verbose=0,
    validation_data=(x_test, x_test),
)

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 32
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses, Input
from keras.datasets import fashion_mnist
from keras.models import Model

from tp3.loader import CSVLoader

data_column = ["x" + str(i) for i in range(1, 36)]
data, exp = CSVLoader.load("../ej1/fonts.csv", False, data_column, None, False)
data = data[:, 1:]
x_train = data.reshape(32, 7, 5)
x_test = data.reshape(32, 7, 5)

print(x_train)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
print(x_train)

print(x_train.shape)
print(x_test.shape)

latent_dim = 2


class Autoencoder(Model):
    def __init__(self, latent_dim):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(25, activation='sigmoid'),
            layers.Dense(17, activation='sigmoid'),
            layers.Dense(latent_dim, activation='sigmoid'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(17, activation='sigmoid'),
            layers.Dense(25, activation='sigmoid'),
            layers.Dense(35, activation='sigmoid'),
            layers.Reshape((7, 5))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


autoencoder = Autoencoder(latent_dim)

# input_img = Input(shape=(35,))
# encoded = layers.Dense(latent_dim, activation='relu')(input_img)
# decoded = layers.Dense(35, activation='sigmoid')(encoded)
# autoencoder = Model(input_img, decoded)
# encoder = Model(input_img, encoded)
# encoded_input = Input(shape=(latent_dim,))
# decoder_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=50000,
                shuffle=False,
                verbose=0,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
# encoded_imgs = encoder(x_test).numpy()
# decoded_imgs = decoder(encoded_imgs).numpy()

n = 32
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    # plt.imshow(x_test[i].reshape(7, 5))
    # plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    # plt.imshow(decoded_imgs[i].reshape(7, 5))
    # plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

"""## Second example: Image denoising


![Image denoising results](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/images/image_denoise_fmnist_results.png?raw=1)

An autoencoder can also be trained to remove noise from images. In the following section, you will create a noisy version of the Fashion MNIST dataset by applying random noise to each image. You will then train an autoencoder using the noisy image as input, and the original image as the target.

Let's reimport the dataset to omit the modifications made earlier.
"""

# (x_train, _), (x_test, _) = fashion_mnist.load_data()
#
# x_train = x_train.astype('float32') / 255.
# x_test = x_test.astype('float32') / 255.
#
# x_train = x_train[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]
#
# print(x_train.shape)
#
# """Adding random noise to the images"""
#
# noise_factor = 0.2
# x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
# x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
#
# x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
# x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)
#
# """Plot the noisy images.
#
# """
#
# n = 10
# plt.figure(figsize=(20, 2))
# for i in range(n):
#     ax = plt.subplot(1, n, i + 1)
#     plt.title("original + noise")
#     plt.imshow(tf.squeeze(x_test_noisy[i]))
#     plt.gray()
# plt.show()
#
# """### Define a convolutional autoencoder
#
# In this example, you will train a convolutional autoencoder using  [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D) layers in the `encoder`, and [Conv2DTranspose](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2DTranspose) layers in the `decoder`.
# """
#
#
# class Denoise(Model):
#     def __init__(self):
#         super(Denoise, self).__init__()
#         self.encoder = tf.keras.Sequential([
#             layers.Input(shape=(28, 28, 1)),
#             layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),
#             layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)])
#
#         self.decoder = tf.keras.Sequential([
#             layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),
#             layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),
#             layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#
# autoencoder = Denoise()
#
# autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
#
# autoencoder.fit(x_train_noisy, x_train,
#                 epochs=10,
#                 shuffle=True,
#                 validation_data=(x_test_noisy, x_test))
#
# """Let's take a look at a summary of the encoder. Notice how the images are downsampled from 28x28 to 7x7."""
#
# autoencoder.encoder.summary()
#
# """The decoder upsamples the images back from 7x7 to 28x28."""
#
# autoencoder.decoder.summary()
#
# """Plotting both the noisy images and the denoised images produced by the autoencoder."""
#
# encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
# decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()
#
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original + noise
#     ax = plt.subplot(2, n, i + 1)
#     plt.title("original + noise")
#     plt.imshow(tf.squeeze(x_test_noisy[i]))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     bx = plt.subplot(2, n, i + n + 1)
#     plt.title("reconstructed")
#     plt.imshow(tf.squeeze(decoded_imgs[i]))
#     plt.gray()
#     bx.get_xaxis().set_visible(False)
#     bx.get_yaxis().set_visible(False)
# plt.show()
#
# """## Third example: Anomaly detection
#
# ## Overview
#
#
# In this example, you will train an autoencoder to detect anomalies on the [ECG5000 dataset](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000). This dataset contains 5,000 [Electrocardiograms](https://en.wikipedia.org/wiki/Electrocardiography), each with 140 data points. You will use a simplified version of the dataset, where each example has been labeled either `0` (corresponding to an abnormal rhythm), or `1` (corresponding to a normal rhythm). You are interested in identifying the abnormal rhythms.
#
# Note: This is a labeled dataset, so you could phrase this as a supervised learning problem. The goal of this example is to illustrate anomaly detection concepts you can apply to larger datasets, where you do not have labels available (for example, if you had many thousands of normal rhythms, and only a small number of abnormal rhythms).
#
# How will you detect anomalies using an autoencoder? Recall that an autoencoder is trained to minimize reconstruction error. You will train an autoencoder on the normal rhythms only, then use it to reconstruct all the data. Our hypothesis is that the abnormal rhythms will have higher reconstruction error. You will then classify a rhythm as an anomaly if the reconstruction error surpasses a fixed threshold.
#
# ### Load ECG data
#
# The dataset you will use is based on one from [timeseriesclassification.com](http://www.timeseriesclassification.com/description.php?Dataset=ECG5000).
# """
#
# # Download the dataset
# dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
# raw_data = dataframe.values
# dataframe.head()
#
# # The last element contains the labels
# labels = raw_data[:, -1]
#
# # The other data points are the electrocadriogram data
# data = raw_data[:, 0:-1]
#
# train_data, test_data, train_labels, test_labels = train_test_split(
#     data, labels, test_size=0.2, random_state=21
# )
#
# """Normalize the data to `[0,1]`.
#
# """
#
# min_val = tf.reduce_min(train_data)
# max_val = tf.reduce_max(train_data)
#
# train_data = (train_data - min_val) / (max_val - min_val)
# test_data = (test_data - min_val) / (max_val - min_val)
#
# train_data = tf.cast(train_data, tf.float32)
# test_data = tf.cast(test_data, tf.float32)
#
# """You will train the autoencoder using only the normal rhythms, which are labeled in this dataset as `1`. Separate the normal rhythms from the abnormal rhythms."""
#
# train_labels = train_labels.astype(bool)
# test_labels = test_labels.astype(bool)
#
# normal_train_data = train_data[train_labels]
# normal_test_data = test_data[test_labels]
#
# anomalous_train_data = train_data[~train_labels]
# anomalous_test_data = test_data[~test_labels]
#
# """Plot a normal ECG. """
#
# plt.grid()
# plt.plot(np.arange(140), normal_train_data[0])
# plt.title("A Normal ECG")
# plt.show()
#
# """Plot an anomalous ECG."""
#
# plt.grid()
# plt.plot(np.arange(140), anomalous_train_data[0])
# plt.title("An Anomalous ECG")
# plt.show()
#
# """### Build the model"""
#
#
# class AnomalyDetector(Model):
#     def __init__(self):
#         super(AnomalyDetector, self).__init__()
#         self.encoder = tf.keras.Sequential([
#             layers.Dense(32, activation="relu"),
#             layers.Dense(16, activation="relu"),
#             layers.Dense(8, activation="relu")])
#
#         self.decoder = tf.keras.Sequential([
#             layers.Dense(16, activation="relu"),
#             layers.Dense(32, activation="relu"),
#             layers.Dense(140, activation="sigmoid")])
#
#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded
#
#
# autoencoder = AnomalyDetector()
#
# autoencoder.compile(optimizer='adam', loss='mae')
#
# """Notice that the autoencoder is trained using only the normal ECGs, but is evaluated using the full test set."""
#
# history = autoencoder.fit(normal_train_data, normal_train_data,
#                           epochs=20,
#                           batch_size=512,
#                           validation_data=(test_data, test_data),
#                           shuffle=True)
#
# plt.plot(history.history["loss"], label="Training Loss")
# plt.plot(history.history["val_loss"], label="Validation Loss")
# plt.legend()
#
# """You will soon classify an ECG as anomalous if the reconstruction error is greater than one standard deviation from the normal training examples. First, let's plot a normal ECG from the training set, the reconstruction after it's encoded and decoded by the autoencoder, and the reconstruction error."""
#
# encoded_data = autoencoder.encoder(normal_test_data).numpy()
# decoded_data = autoencoder.decoder(encoded_data).numpy()
#
# plt.plot(normal_test_data[0], 'b')
# plt.plot(decoded_data[0], 'r')
# plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
# plt.legend(labels=["Input", "Reconstruction", "Error"])
# plt.show()
#
# """Create a similar plot, this time for an anomalous test example."""
#
# encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
# decoded_data = autoencoder.decoder(encoded_data).numpy()
#
# plt.plot(anomalous_test_data[0], 'b')
# plt.plot(decoded_data[0], 'r')
# plt.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
# plt.legend(labels=["Input", "Reconstruction", "Error"])
# plt.show()
#
# """### Detect anomalies
#
# Detect anomalies by calculating whether the reconstruction loss is greater than a fixed threshold. In this tutorial, you will calculate the mean average error for normal examples from the training set, then classify future examples as anomalous if the reconstruction error is higher than one standard deviation from the training set.
#
# Plot the reconstruction error on normal ECGs from the training set
# """
#
# reconstructions = autoencoder.predict(normal_train_data)
# train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)
#
# plt.hist(train_loss[None, :], bins=50)
# plt.xlabel("Train loss")
# plt.ylabel("No of examples")
# plt.show()
#
# """Choose a threshold value that is one standard deviations above the mean."""
#
# threshold = np.mean(train_loss) + np.std(train_loss)
# print("Threshold: ", threshold)
#
# """Note: There are other strategies you could use to select a threshold value above which test examples should be classified as anomalous, the correct approach will depend on your dataset. You can learn more with the links at the end of this tutorial.
#
# If you examine the reconstruction error for the anomalous examples in the test set, you'll notice most have greater reconstruction error than the threshold. By varing the threshold, you can adjust the [precision](https://developers.google.com/machine-learning/glossary#precision) and [recall](https://developers.google.com/machine-learning/glossary#recall) of your classifier.
# """
#
# reconstructions = autoencoder.predict(anomalous_test_data)
# test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
#
# plt.hist(test_loss[None, :], bins=50)
# plt.xlabel("Test loss")
# plt.ylabel("No of examples")
# plt.show()
#
# """Classify an ECG as an anomaly if the reconstruction error is greater than the threshold."""
#
#
# def predict(model, data, threshold):
#     reconstructions = model(data)
#     loss = tf.keras.losses.mae(reconstructions, data)
#     return tf.math.less(loss, threshold)
#
#
# def print_stats(predictions, labels):
#     print("Accuracy = {}".format(accuracy_score(labels, predictions)))
#     print("Precision = {}".format(precision_score(labels, predictions)))
#     print("Recall = {}".format(recall_score(labels, predictions)))
#
#
# preds = predict(autoencoder, test_data, threshold)
# print_stats(preds, test_labels)
#

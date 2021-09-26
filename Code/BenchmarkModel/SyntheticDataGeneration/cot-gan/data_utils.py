#!/usr/bin/env python

# Classes to create the data for training model
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import tensorflow as tf
import os, glob
import cmath
import re
import sys
import io
import math


class DataProcessor:
    def __init__(self, path, seq_len, channels):
        self.training_path = path
        self.sequence_length = seq_len
        self.channels = channels

    def get_dataset_from_path(self, buffer):
        read_data = tf.data.Dataset.list_files(self.training_path)
        dataset = read_data.repeat().shuffle(buffer_size=buffer)
        dataset = dataset.interleave(tf.data.TFRecordDataset, cycle_length=4, block_length=16)
        return dataset

    def provide_video_data(self, buffer, batch_size, height, width):
        '''
        :return: tf dataset
        '''
        def read_tfrecord(serialized_example):
            features = {'x': tf.io.FixedLenFeature([height * width * self.sequence_length * self.channels, ],
                                                   dtype=tf.float32)}
            example = tf.io.parse_single_example(serialized_example, features)
            return example['x']

        dataset = self.get_dataset_from_path(buffer)
        dataset = dataset.map(read_tfrecord, num_parallel_calls=4)
        dataset = dataset.batch(batch_size)
        return dataset


class AROne:
    '''
    :param D: dimension of x
    :param T: sequence length
    :param phi: parameters for AR model
    :param s: parameter that controls the magnitude of covariance matrix
    '''
    def __init__(self, D, T, phi, s, burn=10):
        self.D = D
        self.T = T
        self.phi = phi
        self.Sig = np.eye(D) * (1 - s) + s
        self.chol = np.linalg.cholesky(self.Sig)
        self.burn = burn

    def batch(self, N):
        x0 = np.random.randn(N, self.D)
        x = np.zeros((self.T + self.burn, N, self.D))
        x[0, :, :] = x0
        for i in range(1, self.T + self.burn):
            x[i, ...] = self.phi * x[i - 1] + np.random.randn(N, self.D) @ self.chol.T

        x = x[-self.T:, :, :]
        x = np.swapaxes(x, 0, 1)
        return x.astype("float32")


class Gaussian:
    def __init__(self, D=1):
        self.D = D

    def batch(self, batch_size):
        return np.random.randn(batch_size, 1, self.D)


class SineImage(object):
    '''
    :param Dx: dimensionality of of data at each time step
    :param angle: rotation
    :param z0: initial position and velocity
    :param rand_std: gaussian randomness in the latent trajectory
    :param noise_std: observation noise at output
    '''
    def __init__(self, Dx=20, angle=np.pi / 6., z0=None, rand_std=0.0, noise_std=0.0, length=None, amp=1.0):
        super().__init__()
        self.D = 2
        self.Dx = Dx
        self.z0 = z0

        self.A = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        self.rand_std = rand_std
        self.noise_std = noise_std
        self.length = length
        self.amp = amp

    def sample(self, n, T):
        # n: number of samples
        # T: lenght of each sample
        if self.z0 is None:
            z = np.random.randn(n, 2)
            z = z / np.linalg.norm(z, axis=-1, keepdims=True)
        else:
            z = np.tile(self.z0, (n, 1))

        zs = []
        for t in np.arange(T):
            m = self.conditional_param(z)
            z = m + np.random.randn(*m.shape) * self.rand_std
            zs += z,

        zs = np.stack(zs, 1)

        grid = np.linspace(-1.5, 1.5, self.Dx)

        mean = np.exp(- 0.5 * (zs[..., :1] - grid) ** 2 / 0.3 ** 2) * self.amp
        mean = mean.reshape(n, -1)
        xs = mean + np.random.randn(*mean.shape) * self.noise_std

        return zs, xs.reshape(n, T, self.Dx)

    def conditional_param(self, zt):

        slope = 1.0
        r = np.sqrt(np.sum(zt ** 2, -1))
        r_ratio = 1.0 / (np.exp(-slope * 4 * (r - 0.3)) + 1) / r

        ztp1 = zt @ self.A
        ztp1 *= r_ratio[..., None]

        return ztp1

    def batch(self, batch_size):
        return self.sample(batch_size, self.length)[1]


class NPData(object):
    def __init__(self, data, batch_size, nepoch=np.inf, tensor=True):
        self.data = data
        self.N, self.length = data.shape[0:2]
        self.epoch = 0
        self.counter = 0
        np.random.shuffle(self.data)
        self.batch_size = batch_size
        self.nepoch = nepoch
        self.tensor = tensor

    def __iter__(self):
        return self

    def __len__(self):
        return int(np.floor(self.N / self.batch_size))

    def __next__(self):
        if (self.counter + 1) * self.batch_size > self.N:
            self.epoch += 1
            np.random.shuffle(self.data)
            self.counter = 0

        if np.isfinite(self.nepoch) and self.epoch == self.nepoch:
            raise StopIteration

        idx = slice(self.counter * self.batch_size, (self.counter + 1) * self.batch_size)
        batch = self.data[idx]
        self.counter += 1
        if self.tensor:
            batch = tf.cast(batch, tf.float32)
        return batch

    def batch(self, batch_size):
        return self.__next__()


class EEGData(NPData):
    '''
    :param Dx: dimensionality of of data at each time step
    :param length: sequence length
    :param batch size: batch size
    '''

    def __init__(self, Dx, length, batch_size, nepoch=np.inf, tensor=True, seed=0, prefix="", downsample=1):
        # nsubject x n trials x channel x times_steps
        all_data = np.load(prefix + "data/eeg/eeg_data.npy", allow_pickle=True)
        train_data = []
        test_data = []
        sep = 0.75
        np.random.RandomState(seed).shuffle(all_data)
        for sub_data in all_data:
            ntrial = int(sep * len(sub_data))
            train_data += sub_data[:ntrial, :downsample * length:downsample, :Dx],
            test_data += sub_data[ntrial:, :downsample * length:downsample, :Dx],
            assert train_data[-1].shape[1] == length
            assert train_data[-1].shape[2] == Dx

        self.train_data = self.normalize(train_data)
        self.test_data = self.normalize(test_data)
        self.all_data = np.concatenate([self.train_data, self.test_data], 0)
        super().__init__(self.train_data, batch_size, nepoch, tensor)

    def normalize(self, data):
        data = np.concatenate(data, 0)
        m, s = data.mean((0, 1)), data.std((0, 1))
        data = (data - m) / (3 * s)
        data = np.tanh(data)
        return data


def plot_batch(batch_series, iters, saved_file, axis=None):
    '''
    :param batch_series: a batch of sequence
    :param iters: current iteration
    :return: plots up to six sequences on shared axis
    '''
    flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    batch_size = np.shape(batch_series)[0]
    num_seq = np.minimum(len(flatui), batch_size)

    for i in range(0, num_seq):
        data = [_ for _ in enumerate(batch_series[i])]
        sns.lineplot(x=[el[0] for el in data],
                     y=[el[1] for el in data],
                     color=flatui[i % len(flatui)],
                     ax=axis)
    str = "Sample plot after {} iterations".format(iters)
    plt.title(str)
    plt.savefig("./trained/{}/images/{}.png".format(saved_file, str))
    plt.close()


def display_images(x, row, col, batch_size, height, width, iters, saved_file):
    fig, axe = plt.subplots(row, col, figsize=(8, 8))

    for i in range(row):
        for j in range(col):
            axe[i][j].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width]), origin="upper",
                             cmap="gray", interpolation="nearest")
            axe[i][j].set_xticks([])
            axe[i][j].set_yticks([])
    str = "Sample plot after {} iterations".format(iters)
    # plt.title(str)
    plt.savefig("./trained/{}/images/{}.png".format(saved_file, str))
    plt.close()


def display_frames(x, row, batch_size, seq_len, height, width, channels, iters, saved_file):
    fig, axe = plt.subplots(row, figsize=(8, 8))

    for i in range(row):
        if channels > 1:
            axe[i].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width * seq_len, channels]),
                          origin="upper", cmap="gray", interpolation="nearest")
        else:
            axe[i].imshow(np.reshape(x[np.random.randint(0, batch_size), ...], [height, width * seq_len]),
                          origin="upper", cmap="gray", interpolation="nearest")
        axe[i].set_xticks([])
        axe[i].set_yticks([])
    str = "Sample plot after {} iterations".format(iters)
    # plt.title(str)
    plt.savefig("./trained/{}/images/{}.png".format(saved_file, str))
    plt.close()


def check_model_summary(batch_size, seq_len, model, stateful=False):
    if stateful:
        inputs = tf.keras.Input((batch_size, seq_len))
    else:
        inputs = tf.keras.Input((batch_size, seq_len))
    outputs = model.call(inputs)

    model_build = tf.keras.Model(inputs, outputs)
    print(model_build.summary())

from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

import lasagne as nn


import Semi_Supervised_VAE.sampling_layers as sl
from Semi_Supervised_VAE import util

TWO_PI = T.Constant(2 * np.pi, theano.config.floatX)


def build_model(x_sym):
    input_l = nn.layers.InputLayer(
        [None, 784],
        x_sym
    )

    h_mu = nn.layers.DenseLayer(
        input_l,
        500,
        nonlinearity=nn.nonlinearities.softplus
    )

    h_sigma = nn.layers.DenseLayer(
        input_l,
        500,
        nonlinearity=nn.nonlinearities.softplus
    )

    h_pi = nn.layers.DenseLayer(
        input_l,
        500,
        nonlinearity=nn.nonlinearities.softplus
    )

    p_y = nn.layers.DenseLayer(
        h_pi,
        10,
        nonlinearity=nn.nonlinearities.softmax
    )

    y = sl.BernoulliSampleLayer(p_y)

    h_mu_y = nn.layers.ConcatLayer([y, h_mu])

    mu = nn.layers.DenseLayer(
        h_mu_y,
        50,
        nonlinearity=nn.nonlinearities.identity
    )

    log_sigma = nn.layers.DenseLayer(
        h_sigma,
        50,
        nonlinearity=nn.nonlinearities.identity
    )

    z = sl.GaussianSampleLayer(mu, log_sigma)

    yz = nn.layers.ConcatLayer([y, z])

    h_p = nn.layers.DenseLayer(
        yz,
        500,
        nonlinearity=nn.nonlinearities.softplus
    )

    p_x = nn.layers.DenseLayer(
        h_p,
        784,
        nonlinearity=nn.nonlinearities.sigmoid
    )

    return p_x, y, z, p_y, mu, log_sigma


def log_gaussian_pdf(z, mu, log_sigma):
    log_norm = -0.5 * T.log(TWO_PI * T.prod(T.exp(2 * log_sigma), axis=1))
    exp_arg = -0.5 * ((z - mu) ** 2 / T.exp(2 * log_sigma)).sum(axis=1)
    return log_norm + exp_arg


def train(X_unlabelled, X_labelled, y_train, X_test, y_test):
    x_sym = T.matrix("X")
    y_sym = T.vector("y")

    prior_y = T.constant(0.1)

    alpha = 0.1

    layers = build_model(x_sym)
    y_l = layers[1]

    p_x, y, z, p_y, mu, log_sigma = nn.layers.get_output(layers, {y_l: y_sym})

    def build_loss(supervised):
        nll = T.sum(nn.objectives.binary_crossentropy(p_x, x_sym), axis=1)
        p_y = T.log(prior_y)
        p_z = log_gaussian_pdf(z, T.zeros_like(z), T.zeros_like(z))
        q_z = log_gaussian_pdf(z, mu, log_sigma)

        loss = -nll + p_y + p_z - q_z

        if not supervised:
            this_p_y = p_y[T.arange(p_y.shape[0]), y]
            loss = this_p_y * (loss - T.log(this_p_y))
        else:
            loss += alpha * nn.objectives.categorical_crossentropy(p_y, y)

        return loss.mean(axis=0)

    supervised_loss = build_loss(supervised=True)
    unsupervised_loss = build_loss(supervised=False)

    params = nn.layers.get_all_params(layers[0], trainable=True)
    supervised_updates = nn.updates.adam(supervised_loss, params)
    unsupervised_updates = nn.updates.adam(unsupervised_loss, params)

    train_supervised = theano.function([x_sym, y_sym], supervised_loss,
                                       updates=supervised_updates)
    train_unsupervised = theano.function([x_sym, y_sym], unsupervised_loss,
                                         updates=unsupervised_updates)

    n_epochs = 200
    batch_size = 100
    n_labels = 10

    for i in range(n_epochs):
        for j, X_mb in enumerate(util.iterate_minibatches(X_unlabelled,
                                                          batch_size)):
            mb_loss = 0
            y_mb = np.zeros(batch_size, dtype=np.uint8)
            for l in range(n_labels):
                mb_loss += train_unsupervised(X_mb, y_mb)
                y_mb += 1
            print("Epoch {},\tminibatch {}:\t\tUnsupervised loss={}"
                  .format(i+1, j+1, mb_loss))

        for j, (X_mb, y_mb) in enumerate(util.iterate_minibatches(X_labelled,
                                                                  batch_size,
                                                                  y=y_train)):
            mb_loss = train_supervised(X_mb, y_mb)
            print ("Epoch {},\tminibatch {}:\t\tSupervised loss={}"
                   .format(i+1, j+1, mb_loss))

from __future__ import print_function

import numpy as np

import theano
import theano.tensor as T

import lasagne as nn


import Semi_Supervised_VAE.sampling_layers as sl
from Semi_Supervised_VAE import util

TWO_PI = T.constant(2 * np.pi, dtype=theano.config.floatX)
EPS = 1e-5


def build_model(x_sym):
    input_l = nn.layers.InputLayer(
        [None, 784],
        x_sym,
        name="Input"
    )

    h_mu = nn.layers.DenseLayer(
        input_l,
        500,
        nonlinearity=nn.nonlinearities.softplus,
        name="Hidden recognition mean"
    )

    h_sigma = nn.layers.DenseLayer(
        input_l,
        500,
        nonlinearity=nn.nonlinearities.softplus,
        name="Hidden recognition log sd"
    )

    h_pi = nn.layers.DenseLayer(
        input_l,
        500,
        nonlinearity=nn.nonlinearities.softplus,
        name="Hidden y layer"
    )

    def softmax(x):
        return T.exp(x) / T.exp(x).sum(1, keepdims=True)
    p_y = nn.layers.DenseLayer(
        h_pi,
        10,
        nonlinearity=nn.nonlinearities.softmax,
        name="Posterior y probabilities"
    )

    y = sl.BernoulliSampleLayer(p_y, name="y samples")

    h_mu_y = nn.layers.ConcatLayer([h_mu, y], name="Hidden mean, sampled y "
                                                   "concatenation")

    mu = nn.layers.DenseLayer(
        h_mu_y,
        50,
        nonlinearity=nn.nonlinearities.identity,
        name="Recognition mean"
    )

    log_sigma = nn.layers.DenseLayer(
        h_sigma,
        50,
        nonlinearity=nn.nonlinearities.identity,
        name="Recognition log sd"
    )

    z = sl.GaussianSampleLayer(mu, log_sigma, name="Posterior sample")

    yz = nn.layers.ConcatLayer([y, z], name="Sample concatenation")

    h_p = nn.layers.DenseLayer(
        yz,
        500,
        nonlinearity=nn.nonlinearities.softplus,
        name="Hidden reconstruction"
    )

    p_x = nn.layers.DenseLayer(
        h_p,
        784,
        nonlinearity=nn.nonlinearities.sigmoid,
        name="Reconstruction probabilities"
    )

    return p_x, y, z, p_y, mu, log_sigma


TWO = T.constant(2, dtype=theano.config.floatX)


def log_gaussian_pdf(z, mu, log_sigma):
    log_norm = -T.log(TWO_PI * T.prod(T.exp(TWO * log_sigma), axis=1)) / TWO
    exp_arg = -((z - mu) ** TWO / T.exp(TWO * log_sigma)).sum(axis=1) / TWO
    return log_norm + exp_arg


def train(X_unlabelled, X_labelled, y_train, X_test, y_test):
    x_sym = T.matrix("X")
    y_sym = T.matrix("y", dtype="uint8")

    prior_y = T.constant(0.1, dtype=theano.config.floatX)

    alpha = T.constant(0.1, dtype=theano.config.floatX)

    layers = build_model(x_sym)
    y_l = layers[1]

    p_x, y, z, p_y, mu, log_sigma = nn.layers.get_output(layers, {y_l: y_sym})

    def build_loss(supervised):
        bc = nn.objectives.binary_crossentropy
        nll = T.sum(bc(p_x.clip(EPS, 1 - EPS), x_sym), axis=1)
        p_theta_y = T.log(prior_y)
        p_theta_z = log_gaussian_pdf(z, T.zeros_like(z), T.zeros_like(z))
        q_z = log_gaussian_pdf(z, mu, log_sigma)

        loss = -nll + p_theta_y + p_theta_z - q_z

        if not supervised:
            this_p_y = p_y[T.arange(p_y.shape[0]), y_sym.argmax(axis=1)]
            loss = this_p_y * (loss - T.log(this_p_y))
        else:
            cc = nn.objectives.categorical_crossentropy

            loss += alpha * cc(p_y.clip(EPS, 1 - EPS), y_sym)

        return loss.mean(axis=0)

    supervised_loss = -build_loss(supervised=True)
    unsupervised_loss = -build_loss(supervised=False)

    params = nn.layers.get_all_params(layers[0], trainable=True)
    supervised_updates = nn.updates.adam(supervised_loss, params)
    unsupervised_updates = nn.updates.adam(unsupervised_loss, params)

    train_supervised = theano.function([x_sym, y_sym], supervised_loss,
                                       updates=supervised_updates)
    train_unsupervised = theano.function([x_sym, y_sym], unsupervised_loss,
                                         updates=unsupervised_updates)

    test_loss = T.eq(p_y.argmax(axis=1), y_sym).sum()
    test_fn = theano.function([x_sym, y_sym], test_loss)

    n_epochs = 1
    batch_size = 1000
    n_labels = 10

    for i in range(n_epochs):
        for j, X_mb in enumerate(util.iterate_minibatches(X_unlabelled,
                                                          batch_size)):
            mb_loss = 0
            for l in range(n_labels):
                y_mb = np.zeros((batch_size, 10), dtype=np.uint8)
                y_mb[:, l] = 1
                mb_loss += train_unsupervised(X_mb, y_mb)
            print("Epoch {},\tminibatch {}:\t\tUnsupervised loss={}"
                  .format(i+1, j+1, mb_loss))

        for j, (X_mb, y_mb) in enumerate(util.iterate_minibatches(X_labelled,
                                                                  batch_size,
                                                                  y=y_train)):
            mb_loss = train_supervised(X_mb, y_mb)
            print("Epoch {},\tminibatch {}:\t\tSupervised loss={}"
                  .format(i+1, j+1, mb_loss))

    n_correct = 0
    for X_mb, y_mb in util.iterate_minibatches(X_test, batch_size,
                                               y=y_test, shuffle=False):
        n_correct += test_fn(X_mb, y_mb)
    print("{}\% test images correctly classified after "
          "training".format(n_correct / y_test.shape[0] * 100))

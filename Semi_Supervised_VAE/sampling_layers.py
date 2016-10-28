import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne as nn


class GaussianSampleLayer(nn.layers.MergeLayer):

    def __init__(self, mu, log_sd, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams()
        super(GaussianSampleLayer, self).__init__([mu, log_sd], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, log_sd = inputs
        if deterministic:
            return mu
        else:
            shape = (self.input_shapes[0][0] or mu.shape[0],
                     self.input_shapes[0][1] or mu.shape[1])
            return mu + (T.exp(log_sd) *
                         self.rng.normal(shape, dtype=theano.config.floatX))


class BernoulliSampleLayer(nn.layers.Layer):

    def __init__(self, incoming, rng=None, **kwargs):
        self.rng = rng if rng is not None else RandomStreams()
        super(BernoulliSampleLayer, self).__init__(incoming, **kwargs)

    def get_output_shapre_for(self, input_shape):
        return input_shape

    def get_output_for(self, input, deterministic=False, **kwargs):
        probs = input
        if deterministic:
            return probs
        else:
            return self.rng.multinomial(pvals=probs, dtype=theano.config.floatX)

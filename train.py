import theano
import theano.tensor as T
import lasagne as nn
import Semi_Supervised_VAE.sampling_layers as sl


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

    p = nn.layers.DenseLayer(
        h_p,
        784,
        nonlinearity=nn.nonlinearities.sigmoid
    )

    return p, y, z, pi


def train():
    x_sym = T.matrix("X")
    y_sym = T.vector("y")

    p, y, z, pi = build_model(x_sym)

    def build_loss(supervised):
        nll = T.sum(nn.objectives.binary_crossentropy(p, x_sym), axis=1)
        p_y = T.log(pi[y])
        p_z = -0.5
        q_z = None

        loss = -nll + p_y + p_z - q_z

        if not supervised:
            H = -T.sum(p * T.log(p), axis=1)
            loss = p[T.arange(p.shape[0]), y] * (loss + H)
        else:
            loss += alpha * nn.objectives.categorical_crossentropy(pi, y)

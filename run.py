from Semi_Supervised_VAE.train import train
from Semi_Supervised_VAE import util


def main():
    train(*util.load_mnist())


if __name__ == '__main__':
    main()

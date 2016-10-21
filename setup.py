try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Semi-Supervised VAE',
    'author': 'Raza Habib',
    'url': 'URL to get it at.',
    'download_url': 'Where to download it.',
    'author_email': 'raza.habib.15@ucl.ac.uk',
    'version': '0.1',
    'install_requires': ['nose'],
    'packages': ['NAME'],
    'scripts': [],
    'name': 'projectname'
}

setup(**config)
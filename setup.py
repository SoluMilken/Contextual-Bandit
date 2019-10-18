import os
from pathlib import Path

from setuptools import setup


here = os.path.abspath(os.path.dirname(__file__))

readme = Path(__file__).parent.joinpath('README.md')
if readme.exists():
    with readme.open() as f:
        long_description = f.read()
else:
    long_description = '-'

REQUIRED_PACKAGES = [
    'numpy',
]

about = {}
with open(os.path.join(here, 'contextual_bandits', '__version__.py'), 'r') as filep:
    exec(filep.read(), about)


setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    long_description=long_description,
    python_requires='>=3.6',
    packages=[
        'contextual-bandits',
    ],
    install_requires=REQUIRED_PACKAGES,
    author=about['__author__'],
    author_email='s916526000@gmail.com',
    url='',
    license='MIT',
)

"""Setup file for the package"""

from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='python-common-helpers',
    version='1.0.0',
    description='Helper functions to use in Python code',
    long_description=long_description,
    url='https://github.aig.net/mcontrac/python-common-helpers',
    author='Munir Contractor',
    author_email='munir.contractor@aig.com',
    keywords='helpers development',
    packages=find_packages(exclude=['tests', 'docs']),
    install_requires=[],
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Helpers',
        'Programming Language :: Python :: 2.7'
    ]
)

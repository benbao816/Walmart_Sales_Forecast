from setuptools import find_packages, setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='src',
    packages=find_packages(),
    install_requires=requirements,
    version='0.1',
    python_requires='==3.10.12',
    description='Time-Series Forecast - Walmart Sales',
    author='Ben Bao'
)
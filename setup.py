from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Predicting level of damage to buildings caused by the Gorkha earthquale idepending on building location and building construction properties. Data used for the model prediction are based on surveys conducted by Kathmandu Living Labs and the Central Bureau of Statistics.',
    author='QTeam',
    license='MIT',
)

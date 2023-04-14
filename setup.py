import os

from setuptools import setup


def read_requirements(path='./requirements.txt'):
    with open(path) as file:
        install_requires = file.readlines()

    return install_requires


setup(
    name="detectnet_v2",
    version="0.1.0",
    author="Behnam Samadi",
    author_email="behnamsamadi27@gmail.com",
    description=(
        "A package to inference DetectNetV2 TRT Engines with python"
    ),
    packages=[
        'detectnet_v2'
    ],
    install_requires=read_requirements()
)

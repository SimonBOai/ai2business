from distutils.core import setup

import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()
print(required)
setup(
    name="ai2business",
    version="0.1.0",
    packages=setuptools.find_packages(),
    url="https://github.com/AI2Business/ai2business",
    license="Apache License 2.0",
    author="ai2business",
    author_email="ai2business@protonmail.com",
    description="Smart Solutions for Business with AI",
    install_requires=required,
)

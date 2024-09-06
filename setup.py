from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="gmlp_tinygrad",
    version="0.9.1",
    packages=find_packages(),
    install_requires=[
        "tinygrad",
        "einops",
        "numpy"
    ],
    author="Ethan Bennett",
    long_description=long_description,
    long_description_content_type="text/markdown",
)
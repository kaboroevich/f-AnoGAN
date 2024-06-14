from setuptools import setup, find_packages

install_requires = [
    "numpy",
    "matplotlib",
    "torch",
    "lightning",
    "ipywidgets",
    "IPython"
]

setup(
    name="f-AnoGan",
    version="",
    packages=find_packages(),
    url="https://github.com/kaboroevich/f-AnoGan",
    license="GPLv3",
    author="Keith A. Boroevich",
    author_email="kaboroevich@gmail.com",
    description="Implementation of a 1D f-AnoGAN in PyTorch.",
    install_requires=install_requires
)

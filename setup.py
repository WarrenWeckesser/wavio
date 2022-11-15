from setuptools import setup
from os import path


def get_wavio_version():
    """
    Find the value assigned to __version__ in wavio.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in wavio.py.  It returns the string version-string, or None if such a
    line is not found.
    """
    with open("wavio.py", "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]


# Get the long description from README.rst.
_here = path.abspath(path.dirname(__file__))
with open(path.join(_here, 'README.rst')) as f:
    _long_description = f.read()

setup(
    name='wavio',
    version=get_wavio_version(),
    author='Warren Weckesser',
    description=("A Python module for reading and writing WAV files using "
                 "numpy arrays."),
    long_description=_long_description,
    license="BSD",
    url="https://github.com/WarrenWeckesser/wavio",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="wav numpy",
    py_modules=["wavio"],
    install_requires=[
        'numpy >= 1.19.0',
    ],
)

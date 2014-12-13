from setuptools import setup


setup(
    name='wavio',
    version='0.0.1',
    author='Warren Weckesser',
    description=("Read and write 24 bit WAV files with numpy arrays."),
    license="BSD",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    py_modules=["wavio"],
    install_requires=[
        'numpy >= 1.6.0',
    ],
)

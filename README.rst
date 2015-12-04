wavio
=====

``wavio`` is a Python module that defines two functions:

* ``wavio.read`` reads a WAV file and returns an object that holds the sampling
  rate, sample width (in bytes), and a numpy array containing the data.
* ``wavio.write`` writes a numpy array to a WAV file, optionally using a
  specified sample width.

The module uses the ``wave`` module in Python's standard library, so it has the
same limitations as that module.  In particular, it does not support compressed
WAV files, and it does not handle floating point WAV files.  (When floating
point data is passed to ``wavio.write`` it is converted to integers before
being written to the WAV file.)  The functions can read and write 8-, 16-, 24-
and 32-bit integer WAV files.

``wavio`` has been tested with Python versions 2.7, 3.4 and 3.5.

``wavio`` depends on numpy (http://www.numpy.org).  It has been tested with
versions 1.8.1, 1.9.0 and 1.10.1, and will likely work with older versions.

The package has a suite of unit tests, but it should still be considered
prototype-quality software.  There may be backwards-incompatible API changes
between releases.

Example
~~~~~~~

The following code (also found in the docstring of ``wavio.write``) writes
a three second 440 Hz sine wave to a 24-bit WAV file::

    import numpy as np
    import wavio

    rate = 22050  # samples per second
    T = 3         # sample duration (seconds)
    f = 440.0     # sound frequency (Hz)
    t = np.linspace(0, T, T*rate, endpoint=False)
    x = np.sin(2*np.pi * f * t)
    wavio.write("sine24.wav", x, rate, sampwidth=3)


-----

:Author:     Warren Weckesser
:Repository: https://github.com/WarrenWeckesser/wavio
:License:    BSD 2-clause (http://opensource.org/licenses/BSD-2-Clause)

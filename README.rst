wavio
=====

``wavio`` is a Python module that defines two functions:

* ``wavio.read`` reads a WAV file and returns an object that holds the
  sampling rate, sample width (in bytes), and a numpy array containing the
  data.
* ``wavio.write`` writes a numpy array to a WAV file, optionally using a
  specified sample width.

The functions can read and write 8-, 16-, 24- and 32-bit integer WAV files.

The module uses the ``wave`` module in Python's standard library, so it has
the same limitations as that module.  In particular, the ``wave`` module
does not support compressed WAV files, and it does not handle floating
point WAV files.  When floating point data is passed to ``wavio.write`` it
is converted to integers before being written to the WAV file.

``wavio`` requires Python 3.6 or later.

``wavio`` depends on numpy (http://www.numpy.org).  NumPy version 1.19.0 or
later is required.    The unit tests in ``wavio`` require ``pytest``.

The API of the functions in ``wavio`` should not be considered stable.  There
may be backwards-incompatible API changes between releases.

*Important notice*

In version 0.0.5 (not released yet), the data handling has been changed in a
backwards-incompatible way.  The API in 0.0.4 was a flexible interface that
only its creator could love.  The new API is simpler, and it is hoped that it
does the right thing by default in most cases.  In particular:

* When the input data is an integer type, the values are not scaled or
  shifted.  The only changed that might happen is the data might be clipped
  if the values do not fit in the output integer type.
* By default, floating point input is scaled to the full width of the
  output integer type, with the constraint that 0 in the input is mapped
  to the midpoint of the output integer type.  The ```scale``` parameter allows
  that behavior to be changed--it gives the upper bound of the float values
  that are mapped to the maximum of the output integer type.  Regardless of
  the value of ``scale``, the float input 0.0 is always mapped to the midpoint
  of the output type.
* A warning is now generated if any data values are clipped.  A parameter
  allows the generation of the warning to be disabled or converted to an
  exception.

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

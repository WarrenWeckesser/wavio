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

``wavio`` requires Python 3.7 or later.

``wavio`` depends on numpy (http://www.numpy.org).  NumPy version 1.19.0 or
later is required.    The unit tests in ``wavio`` require ``pytest``.

The API of the functions in ``wavio`` should not be considered stable.  There
may be backwards-incompatible API changes between releases.

*Important notice*

In version 0.0.5, the data handling in ``wavio.write`` has been changed in
a backwards-incompatible way.  The API for scaling the input in 0.0.4 was
a flexible interface that only its creator could love.  The new API is
simpler, and it is hoped that it does the right thing by default in
most cases.  In particular:

* When the input data is an integer type, the values are not scaled or
  shifted.  The only change that might happen is the data will be clipped
  if the values do not fit in the output integer type.
* If the input data is a floating point type, ``sampwidth`` must be given.
  The default behavior is to scale input values in the range [-1.0, 1.0]
  to the output range [min_int+1, max_int], where min_int and max_int are
  the minimum and maximum values of the output data type determined by
  ``sampwidth``.  See the description of ``scale`` in the docstring of
  ``wavio.write`` for more options.  Regardless of the value of ``scale``,
  the float input 0.0 is always mapped to the midpoint of the output type;
  ``wavio.write`` will not translate the values up or down.
* A warning is now generated if any data values are clipped.  A parameter
  allows the generation of the warning to be disabled or converted to an
  exception.

Examples
--------

The following examples are also found in the docstring of ``wavio.write``.

Create a 3 second 440 Hz sine wave, and save it in a 24-bit WAV file.

    >>> import numpy as np
    >>> import wavio

    >>> rate = 22050           # samples per second
    >>> T = 3                  # sample duration (seconds)
    >>> n = int(rate*T)        # number of samples
    >>> t = np.arange(n)/rate  # grid of time values

    >>> f = 440.0              # sound frequency (Hz)
    >>> x = np.sin(2*np.pi * f * t)

`x` is a single sine wave with amplitude 1, so we can use the default
`scale`.

    >>> wavio.write("sine24.wav", x, rate, sampwidth=3)

Create a file that contains the 16 bit integer values -10000 and 10000
repeated 100 times.  Use a sample rate of 8000.

    >>> x = np.empty(200, dtype=np.int16)
    >>> x[::2] = -10000
    >>> x[1::2] = 10000
    >>> wavio.write("foo.wav", x, 8000)

Check that the file contains what we expect.  The values are checked
for exact equality.  The input was an integer array, so the values are
not scaled.

    >>> w = wavio.read("foo.wav")
    >>> np.all(w.data[:, 0] == x)
    True

Write floating point data to a 16 bit WAV file.  The floating point
values are assumed to be within the range [-2, 2], and we want the
values 2 and -2 to correspond to the full output range, even if the
actual values in the data do not fill this range.  We do that by
specifying `scale=2`.

`T`, `rate` and `t` are from above.  The data is the sum of two
sinusoids, with frequencies 440 and 880 Hz, modulated by a parabolic
curve that is zero at the start and end of the data.

    >>> envelope = (4/T**2)*(t * (T - t))
    >>> omega1 = 2*np.pi*440
    >>> omega2 = 2*np.pi*880
    >>> y = envelope*(np.sin(omega1*t) + 0.3*np.sin(omega2*t + 0.2))
    >>> y.min(), y.max()
    (-1.1745469775555515, 1.093833464065767)

Write the WAV file, with `scale=2`.

    >>> wavio.write('harmonic.wav', y, rate, sampwidth=2, scale=2)

Check the minimum and maximum integers that were actually written
to the file:

    >>> w = wavio.read("harmonic.wav")
    >>> w.data.min(), w.data.max()
    (-19243, 17921)

If we want the WAV file to use as much of the range of the output
integer type as possible (while still mapping 0.0 in the input to 0 in
the output), we set `scale="auto"`.

    >>> wavio.write('harmonic_full.wav', y, rate, sampwidth=2, scale="auto")

    >>> w = wavio.read('harmonic_full.wav')
    >>> w.data.min(), w.data.max()
    (-32768, 30517)

-----

Author:     Warren Weckesser

Repository: https://github.com/WarrenWeckesser/wavio

License:    BSD 2-clause (http://opensource.org/licenses/BSD-2-Clause)

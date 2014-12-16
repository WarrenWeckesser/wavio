**wavio**

`wavio` is a Python module that defines two functions:

* `readwav` reads a WAV file and returns the sampling rate, sample width
  (in bytes), and a numpy array containing the data.
* `writewav24` writes a numpy array to a 24 bit WAV file.

The module uses the `wave` module in Python's standard library, so it has the
same limitations as that module.  In particular, it does not support compressed
WAV files.

The `wavio` module provides an alternative to the SciPy module `scipy.io.wavfile`.
As of version 0.15.0 of scipy, the functions in `scipy.io.wavfile` do not support
24 bit sample widths.  The function in this module, `wavio.readwav`, can read
24 bit files.  When the sample depth is 24 bits, the data is returned in a 32 bit
numpy array.

`wavio` has been tested with Python versions 2.7 and 3.4.  It will likely
work with older versions.

`wavio` depends on numpy (http://www.numpy.org).  It has been tested with versions 1.8.1
and 1.9.0, and will likely work with older versions.

-----

_Author_:     Warren Weckesser  <br />
_Repository_: https://github.com/WarrenWeckesser/wavio  <br />
_License_:    BSD 3-clause (http://opensource.org/licenses/BSD-3-Clause)     <br />

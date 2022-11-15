"""
The wavio module defines the functions:

read(file)
    Read a WAV file and return a `wavio.Wav` object, with attributes
    `data`, `rate` and `sampwidth`.

write(filename, data, rate, scale=None, sampwidth=None)
    Write a numpy array to a WAV file.


-----
Author: Warren Weckesser
License: BSD 2-Clause:
Copyright (c) 2015-2022, Warren Weckesser
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import warnings as _warnings
import wave as _wave
import numpy as _np


__version__ = "0.0.7"


class ClippedDataWarning(UserWarning):

    def __init__(self, message=None):
        if message is None:
            message = ('Some data values were clipped when converted to the '
                       'output format.')
        self.args = (message,)


class ClippedDataError(RuntimeError):

    def __init__(self, message=None):
        if message is None:
            message = ('Some data values were clipped when converted to the '
                       'output format.')
        self.args = (message,)


def _wav2array(nchannels, sampwidth, data):
    """data must be the string containing the bytes from the wav file."""
    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = _np.empty((num_samples, nchannels, 4), dtype=_np.uint8)
        raw_bytes = _np.frombuffer(data, dtype=_np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = _np.frombuffer(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)
    return result


def _array2wav(a, sampwidth):
    """
    Convert the input array `a` to a string of WAV data.

    a.dtype must be one of uint8, int16 or int32.  Allowed sampwidth
    values are:
        dtype    sampwidth
        uint8        1
        int16        2
        int32      3 or 4
    When sampwidth is 3, the *low* bytes of `a` are assumed to contain
    the values to include in the string.
    """
    if sampwidth == 3:
        # `a` must have dtype int32
        if a.ndim == 1:
            # Convert to a 2D array with a single column.
            a = a.reshape(-1, 1)
        # By shifting first 0 bits, then 8, then 16, the resulting output
        # is 24 bit little-endian.
        a8 = (a.reshape(a.shape + (1,)) >> _np.array([0, 8, 16])) & 255
        wavdata = a8.astype(_np.uint8).tobytes()
    else:
        # Make sure the array is little-endian, and then convert using
        # tobytes().
        a = a.astype('<' + a.dtype.str[1:], copy=False)
        wavdata = a.tobytes()
    return wavdata


class Wav(object):
    """
    Object returned by `wavio.read`.  Attributes are:

    data : numpy array
        The array of data read from the WAV file. The shape of the array
        is (num_samples, num_channels).  num_channels is the number of audio
        channels (1 for mono, 2 for stereo).   The data type of the array
        (i.e. data.dtype) is determined by `sampwidth`::

                sampwidth      dtype
                    1          numpy.uint8
                    2          numpy.int16
                    3          numpy.int32
                    4          numpy.int32

    rate : int
        The sampling frequency (i.e. frame rate or sample rate) of the
        WAV file.
    sampwidth : int
        The sample width (i.e. number of bytes per sample) of the WAV file.
        For example, `sampwidth == 3` is a 24 bit WAV file.

    """

    def __init__(self, data, rate, sampwidth):
        self.data = data
        self.rate = rate
        self.sampwidth = sampwidth

    def __repr__(self):
        s = ("Wav(data.shape=%s, data.dtype=%s, rate=%r, sampwidth=%r)" %
             (self.data.shape, self.data.dtype, self.rate, self.sampwidth))
        return s


def read(file):
    """
    Read a WAV file.

    Parameters
    ----------
    file : string or file object
        Either the name of a file or an open file pointer.

    Returns
    -------
    wav : wavio.Wav() instance
        The return value is an instance of the class `wavio.Wav`,
        with the following attributes:

            data : numpy array
                The array of data read from the WAV file.  The shape of the
                array is (num_samples, num_channels).  num_channels is the
                number of audio channels (1 for mono, 2 for stereo).  The
                data type of the array (i.e. data.dtype) is determined by
                `sampwidth`::

                    sampwidth      dtype
                        1          numpy.uint8
                        2          numpy.int16
                        3          numpy.int32
                        4          numpy.int32

            rate : int
                The sampling frequency (i.e. frame rate or sample rate) of the
                WAV file.
            sampwidth : int
                The sample width (i.e. number of bytes per sample) of the
                WAV file.  For example, `sampwidth == 3` is a 24 bit WAV file.

    Notes
    -----
    This function uses the `wave` module of the Python standard libary
    to read the WAV file, so it has the same limitations as that library.
    In particular, the function does not read compressed WAV files, and
    it does not read files with floating point data.

    The array returned by `wavio.read` is always two-dimensional.  If the
    WAV data is mono, the array will have shape (num_samples, 1).

    `wavio.read()` does not scale or normalize the data.  The data in the
    array `wav.data` is the data that was in the file.  When the file
    contains 24 bit samples, the resulting numpy array is 32 bit integers,
    with values that have been sign-extended.
    """
    wav = _wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)
    w = Wav(data=array, rate=rate, sampwidth=sampwidth)
    return w


_sampwidth_dtypes = {1: _np.uint8,
                     2: _np.int16,
                     3: _np.int32,
                     4: _np.int32}
_sampwidth_minmax = {1: (0, 255),
                     2: (-2**15, 2**15 - 1),
                     3: (-2**23, 2**23 - 1),
                     4: (-2**31, 2**31 - 1)}


def _float_to_integer(x, sampwidth, scale=None, clip="warn"):

    # For a given sampwidth and scale, the actual allowed
    # interval for float input is [-(1 + 1/c)*scale, scale],
    # where c = 2**(8*sampwidth - 1) - 0.5.  Values outside
    # that interval will result in clipping.

    nbits = 8*sampwidth
    c = 2**(nbits - 1) - 0.5

    if scale == "auto":
        scale = max(_np.max(_np.r_[x[x > 0], 0]),
                    _np.max(_np.r_[-x[x < 0], 0])/(1 + 1/c))
    elif scale is None:
        scale = 1.0

    if sampwidth == 1:
        int_min = 0
        midpoint = 128
        int_max = 255
    else:
        int_min = -2**(nbits - 1)
        midpoint = 0
        int_max = 2**(nbits - 1) - 1

    scaled_x = x / scale
    if _np.any(scaled_x > 1) or _np.any(scaled_x < -1 - 1/c):
        msg = (f'Some data values have been clipped.  With scale={scale}, the '
               'interval of input values that will not be clipped '
               f'is [{-(1 + 1/c)*scale}, {scale}]')
        if clip == "warn":
            _warnings.warn(ClippedDataWarning(msg))
        elif clip == "raise":
            raise ClippedDataError(msg)

    y = midpoint + _round_with_half_towards_zero(x/scale*c)
    y = _np.clip(y, int_min, int_max).astype(_sampwidth_dtypes[sampwidth])
    return y


def _round_with_half_towards_zero(x):
    s = _np.sign(x)
    return s * _np.ceil(_np.abs(x) - 0.5)


def write(file, data, rate, scale=None, sampwidth=None, clip="warn"):
    """
    Write the numpy array `data` to a WAV file.

    The Python standard library "wave" is used to write the data to the
    file, so this function has the same limitations as that module.  In
    particular, the Python library does not support floating point data,
    so this function must convert floating point input to integers before
    writing the data to the file.  See below for the conversion rules.

    *Important notes*

    * If `data` has an *integer* data type, signed or unsigned and any bit
      depth, the values are never scaled or shifted.  The only possible
      changes to the values that can occur is if the data must be clipped
      to fit the desired output sample width.  It is an error to give a
      value for the `scale` parameter if `data` has an integer data type.
    * If `data` is a floating point type, `sampwidth` must be given.  The
      default behavior is to scale input values in the range [-1, 1] to
      the output range [min_int+1, max_int], where min_int and max_int are
      the minimum and maximum values of the output data type determined by
      `sampwidth`.  See the description of `scale` below for more options.

    Parameters
    ----------
    file : string, or file object open for writing in binary mode
        Either the name of a file or an open file pointer.
    data : numpy array, 1- or 2-dimensional, integer or floating point
        If it is 2-d, the rows are the frames (i.e. samples) and the
        columns are the channels.
    rate : int
        The sampling frequency (i.e. frame rate) of the data.
    sampwidth : int, optional
        The sample width, in bytes, of the output file.
        If `sampwidth` is not given, it is inferred (if possible) from
        the data type of `data`, as follows::

            data.dtype     sampwidth
            ----------     ---------
            uint8, int8        1
            uint16, int16      2
            uint32, int32      4

        For any other data types, or to write a 24 bit file, `sampwidth`
        must be given.
    scale : float, optional
        This controls the output range when the input is floating point.
        `scale` must not be given when the input data has integer data
        type.

        if `scale` a is numeric value, then input values in the range
        `[-scale, scale]` are mapped to the integer output range centered
        at the midpoint of the output range.  For 8 bit unsigned integer
        output (i.e. `sampwdith=1`), the midpoint is 128. For `sampwidth`
        2, 3 or 4 (corresponding to signed integer output), the midpoint
        is 0.

        If `scale` is the string `"auto"`, the data written to the file is
        scaled up or down to occupy the full range of the output data type.
        For example, with `sampwidth=2` the input
        `data=[-0.5, -1.0, 0.0, 0.25, 1.0]` would result in
        the 16 signed integers [-16384, -32767, 0, 8192, 32767] being
        written to the WAV file. By setting `scale=2`, the output values
        would be [-8192, -16384, 0, 4096, 16384].

        When the input is floating point and `scale` is not given, the
        default value `scale=1.0` is used.
    clip : str, optional
        If "warn" (the default), the function will generate a warning if
        any of the data values must be clipped when written to the format
        of the output array.  If "raise", the function will raise an
        exception if clipping occurs.  If "ignore", no warning or
        exception is generated is clipping occurs.

    Examples
    --------
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

    """
    if clip not in ["ignore", "warn", "raise"]:
        raise ValueError('clip must be one of "ignore", "warn" or "raise".')

    data = _np.asarray(data)

    if sampwidth is None:
        if not _np.issubdtype(data.dtype, _np.integer) or data.itemsize > 4:
            raise ValueError('when data.dtype is not an 8-, 16-, or 32-bit '
                             'integer type, sampwidth must be specified.')
        sampwidth = data.itemsize
    else:
        if sampwidth not in [1, 2, 3, 4]:
            raise ValueError('sampwidth must be 1, 2, 3 or 4.')

    outdtype = _sampwidth_dtypes[sampwidth]
    outmin, outmax = _sampwidth_minmax[sampwidth]

    if _np.issubdtype(data.dtype, _np.integer):
        if scale is not None:
            raise ValueError('The scale parameter must not be set when the '
                             'input is an integer array.  No shifting or '
                             'scaling is done to integer input values.')
        if (data.min() < outmin or data.max() > outmax):
            if clip == "warn":
                _warnings.warn(ClippedDataWarning())
            elif clip == "raise":
                raise ClippedDataError()
        data = data.clip(outmin, outmax).astype(outdtype)
    elif _np.issubdtype(data.dtype, _np.floating):
        data = _float_to_integer(data, sampwidth, scale=scale, clip=clip)
    else:
        raise TypeError(f'unsupported input array data type: {data.dtype}')

    # At this point, `data` has been converted to have one of the following:
    #    sampwidth   dtype
    #    ---------   -----
    #        1       uint8
    #        2       int16
    #        3       int32
    #        4       int32
    # The values in `data` are in the form in which they will be saved;
    # no more scaling will take place.

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    wavdata = _array2wav(data, sampwidth)

    w = _wave.open(file, 'wb')
    w.setnchannels(data.shape[1])
    w.setsampwidth(sampwidth)
    w.setframerate(rate)
    w.writeframes(wavdata)
    w.close()

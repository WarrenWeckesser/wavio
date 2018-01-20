
import unittest
import contextlib
import tempfile
import os
import wave
import numpy as np
import wavio


@contextlib.contextmanager
def temporary_filepath(filename):
    with tempfile.TemporaryDirectory() as tmpdir:
        fullname = os.path.join(tmpdir, filename)
        yield fullname


data1 = np.array([1, -2,
                  3, -4,
                  2**16, -2**16,
                  -2**20, 2**20,
                  2**23-1, -2**23],
                 dtype=np.int32)


class TestWavio(unittest.TestCase):

    def check_basic(self, filename, nchannels, sampwidth, framerate):
        with wave.open(filename, 'r') as f:
            self.assertEqual(f.getnchannels(), nchannels)
            self.assertEqual(f.getsampwidth(), sampwidth)
            self.assertEqual(f.getframerate(), framerate)

    def check_wavio_read(self, filename, rate, sampwidth, dtype, shape, data):
        w = wavio.read(filename)
        self.assertEqual(w.rate, rate)
        self.assertEqual(w.sampwidth, sampwidth)
        self.assertEqual(w.data.dtype, dtype)
        self.assertEqual(w.data.shape, shape)
        np.testing.assert_equal(w.data, data)

    def test1(self):
        with temporary_filepath("test1data.wav") as filename:
            wavio.write(filename, data1, 44100, sampwidth=3)

            self.check_basic(filename, nchannels=1, sampwidth=3,
                             framerate=44100)
            self.check_wavio_read(filename, rate=44100, sampwidth=3,
                                  dtype=np.int32, shape=(len(data1), 1),
                                  data=data1[:, None])

    def test2(self):
        with temporary_filepath("test2data.wav") as filename:
            data2 = data1.reshape(-1, 2)
            wavio.write(filename, data2, 44100, sampwidth=3)

            self.check_basic(filename, nchannels=2, sampwidth=3,
                             framerate=44100)
            self.check_wavio_read(filename, rate=44100, sampwidth=3,
                                  dtype=np.int32, shape=data2.shape,
                                  data=data2)

    def test3(self):
        with temporary_filepath("test3data.wav") as filename:
            data = np.zeros(32, dtype=np.int16)
            data[1::4] = 10000
            data[3::4] = -10000
            wavio.write(filename, data, 44100)

            self.check_basic(filename, nchannels=1, sampwidth=2,
                             framerate=44100)
            expected = np.zeros_like(data).reshape(-1, 1)
            expected[1::4, 0] = 32767
            expected[3::4, 0] = -32768
            self.check_wavio_read(filename, rate=44100, sampwidth=2,
                                  dtype=np.int16, shape=(32, 1),
                                  data=expected)

    def test4(self):
        with temporary_filepath("test4data.wav") as filename:
            data = np.zeros(32, dtype=np.int16)
            data[1::4] = 10000
            data[3::4] = -10000
            wavio.write(filename, data, 44100, sampwidth=1)

            self.check_basic(filename, nchannels=1, sampwidth=1,
                             framerate=44100)
            expected = 128*np.ones_like(data, dtype=np.uint8).reshape(-1, 1)
            expected[1::4, 0] = 255
            expected[3::4, 0] = 0
            self.check_wavio_read(filename, rate=44100, sampwidth=1,
                                  dtype=np.uint8, shape=(32, 1),
                                  data=expected)

    def test5(self):
        with temporary_filepath("test5data.wav") as filename:
            data = np.zeros(32, dtype=np.int16)
            data[1::4] = 10000
            data[3::4] = -10000
            wavio.write(filename, data, 44100, sampwidth=2, scale='none')

            self.check_basic(filename, nchannels=1, sampwidth=2,
                             framerate=44100)
            self.check_wavio_read(filename, rate=44100, sampwidth=2,
                                  dtype=np.int16, shape=(32, 1),
                                  data=data.reshape(-1, 1))

    def test6(self):
        with temporary_filepath("test6data.wav") as filename:
            data = np.zeros(32, dtype=np.int16)
            data[1::4] = 10000
            data[3::4] = -10000
            wavio.write(filename, data, 44100, sampwidth=1, scale='none')

            self.check_basic(filename, nchannels=1, sampwidth=1,
                             framerate=44100)
            expected = np.zeros_like(data, dtype=np.uint8).reshape(-1, 1)
            expected[1::4, 0] = 255
            self.check_wavio_read(filename, rate=44100, sampwidth=1,
                                  dtype=np.uint8, shape=(32, 1),
                                  data=expected)

    def test_clip(self):
        with temporary_filepath("testdata.wav") as filename:
            data = np.array([-100, 0, 100, 200, 300, 325])
            wavio.write(filename, data, 44100, sampwidth=1, scale='none')

            self.check_basic(filename, nchannels=1, sampwidth=1,
                             framerate=44100)
            expected = np.array([0, 0, 100, 200, 255, 255],
                                dtype=np.uint8).reshape(-1, 1)
            self.check_wavio_read(filename, rate=44100, sampwidth=1,
                                  dtype=np.uint8, shape=(len(data), 1),
                                  data=expected)

    def test_vmin_equal_vmax(self):
        with temporary_filepath("testdata.wav") as filename:
            data = np.array([-100, 0, 100, 200, 300, 325])
            wavio.write(filename, data, 44100, sampwidth=1, scale=(200, 200))

            self.check_basic(filename, nchannels=1, sampwidth=1,
                             framerate=44100)
            expected = np.array([0, 0, 0, 0, 0, 0],
                                dtype=np.uint8).reshape(-1, 1)
            self.check_wavio_read(filename, rate=44100, sampwidth=1,
                                  dtype=np.uint8, shape=(len(data), 1),
                                  data=expected)

    def test_signed8bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        with temporary_filepath("testdata.wav") as filename:
            data = np.array([-2**7, -2**3, -1, 0, 1, 2**3, 2**7-1],
                            dtype=np.int8)
            wavio.write(filename, data, 44100)

            self.check_basic(filename, nchannels=1, sampwidth=1,
                             framerate=44100)
            expected_data = (data.astype(np.int16) + 2**7).astype(np.uint8)
            self.check_wavio_read(filename, rate=44100, sampwidth=1,
                                  dtype=np.uint8, shape=(len(data), 1),
                                  data=expected_data.reshape(-1, 1))

    def test_unsigned8bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        with temporary_filepath("testdata.wav") as filename:
            data = np.array([[0, 1], [2**4-1, 2**4], [2**8-2, 2**8-1]],
                            dtype=np.uint8)
            wavio.write(filename, data, 44100)

            self.check_basic(filename, nchannels=2, sampwidth=1,
                             framerate=44100)
            self.check_wavio_read(filename, rate=44100, sampwidth=1,
                                  dtype=np.uint8, shape=data.shape,
                                  data=data)

    def test_16bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        with temporary_filepath("testdata.wav") as filename:
            data = np.array([-2**15, -2**8, -1, 0, 1, 2**8, 2**15-1],
                            dtype=np.int16)
            wavio.write(filename, data, 44100)

            self.check_basic(filename, nchannels=1, sampwidth=2,
                             framerate=44100)
            self.check_wavio_read(filename, rate=44100, sampwidth=2,
                                  dtype=np.int16, shape=(len(data), 1),
                                  data=data.reshape(-1, 1))

    def test_24bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        with temporary_filepath("testdata.wav") as filename:
            data = np.array([-2**23, -2**16, -1, 0, 1, 2**16, 2**23-1],
                            dtype=np.int32)
            wavio.write(filename, data, 44100, sampwidth=3)

            self.check_basic(filename, nchannels=1, sampwidth=3,
                             framerate=44100)
            self.check_wavio_read(filename, rate=44100, sampwidth=3,
                                  dtype=np.int32, shape=(len(data), 1),
                                  data=data.reshape(-1, 1))

    def test_32bit_full_range_round_trip(self):
        # Regression test for github issue 5.
        with temporary_filepath("testdata.wav") as filename:
            data = np.array([-2**31, -2**16, -1, 0, 1, 2**16, 2**31-1],
                            dtype=np.int32)
            wavio.write(filename, data, 44100)

            self.check_basic(filename, nchannels=1, sampwidth=4,
                             framerate=44100)
            self.check_wavio_read(filename, rate=44100, sampwidth=4,
                                  dtype=np.int32, shape=(len(data), 1),
                                  data=data.reshape(-1, 1))


if __name__ == '__main__':
    unittest.main()


import unittest
import tempfile
import os
import wave
import numpy as np
import wavio


data1 = np.array([1, -2,
                  3, -4,
                  2**16, -2**16,
                  -2**20, 2**20,
                  2**23-1, -2**23],
                 dtype=np.int32)


class TestWavio(unittest.TestCase):

    def test1(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "test1data.wav")
        wavio.write(filename, data1, 44100, sampwidth=3)
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 3)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            w = wavio.read(filename)
            self.assertEqual(w.rate, 44100)
            self.assertEqual(w.sampwidth, 3)
            self.assertEqual(w.data.dtype, np.int32)
            self.assertEqual(w.data.shape, (len(data1), 1))
            np.testing.assert_equal(w.data[:, 0], data1)
        finally:
            os.remove(filename)
            os.removedirs(path)

    def test2(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "test2data.wav")
        data2 = data1.reshape(-1, 2)
        wavio.write(filename, data2, 44100, sampwidth=3)
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 2)
            self.assertEqual(f.getsampwidth(), 3)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            w = wavio.read(filename)
            self.assertEqual(w.rate, 44100)
            self.assertEqual(w.sampwidth, 3)
            self.assertEqual(w.data.dtype, np.int32)
            self.assertEqual(w.data.shape, data2.shape)
            np.testing.assert_equal(w.data, data2)
        finally:
            os.remove(filename)
            os.removedirs(path)

    def test3(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "test3data.wav")
        data = np.zeros(32, dtype=np.int16)
        data[1::4] = 10000
        data[3::4] = -10000

        wavio.write(filename, data, 44100)
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 2)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            w = wavio.read(filename)
            self.assertEqual(w.rate, 44100)
            self.assertEqual(w.sampwidth, 2)
            self.assertEqual(w.data.dtype, np.int16)
            self.assertEqual(w.data.shape, (32, 1))
            expected = np.zeros_like(data).reshape(-1, 1)
            expected[1::4, 0] = 32767
            expected[3::4, 0] = -32768
            np.testing.assert_equal(w.data, expected)
        finally:
            os.remove(filename)
            os.removedirs(path)

    def test4(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "test4data.wav")
        data = np.zeros(32, dtype=np.int16)
        data[1::4] = 10000
        data[3::4] = -10000

        wavio.write(filename, data, 44100, sampwidth=1)
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 1)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            w = wavio.read(filename)
            self.assertEqual(w.rate, 44100)
            self.assertEqual(w.sampwidth, 1)
            self.assertEqual(w.data.dtype, np.uint8)
            self.assertEqual(w.data.shape, (32, 1))
            expected = 128*np.ones_like(data, dtype=np.uint8).reshape(-1, 1)
            expected[1::4, 0] = 255
            expected[3::4, 0] = 0
            np.testing.assert_equal(w.data, expected)
        finally:
            os.remove(filename)
            os.removedirs(path)

    def test5(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "test5data.wav")
        data = np.zeros(32, dtype=np.int16)
        data[1::4] = 10000
        data[3::4] = -10000

        wavio.write(filename, data, 44100, sampwidth=2, scale='none')
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 2)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            w = wavio.read(filename)
            self.assertEqual(w.rate, 44100)
            self.assertEqual(w.sampwidth, 2)
            self.assertEqual(w.data.dtype, np.int16)
            self.assertEqual(w.data.shape, (32, 1))
            np.testing.assert_equal(w.data, data.reshape(-1, 1))
        finally:
            os.remove(filename)
            os.removedirs(path)

    def test6(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "test6data.wav")
        data = np.zeros(32, dtype=np.int16)
        data[1::4] = 10000
        data[3::4] = -10000

        wavio.write(filename, data, 44100, sampwidth=1, scale='none')
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 1)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            w = wavio.read(filename)
            self.assertEqual(w.rate, 44100)
            self.assertEqual(w.sampwidth, 1)
            self.assertEqual(w.data.dtype, np.uint8)
            self.assertEqual(w.data.shape, (32, 1))
            expected = np.zeros_like(data, dtype=np.uint8).reshape(-1, 1)
            expected[1::4, 0] = 255
            np.testing.assert_equal(w.data, expected)
        finally:
            os.remove(filename)
            os.removedirs(path)

    def test_clip(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "testdata.wav")
        data = np.array([-100, 0, 100, 200, 300, 325])

        wavio.write(filename, data, 44100, sampwidth=1, scale='none')
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 1)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            w = wavio.read(filename)
            self.assertEqual(w.rate, 44100)
            self.assertEqual(w.sampwidth, 1)
            self.assertEqual(w.data.dtype, np.uint8)
            self.assertEqual(w.data.shape, (len(data), 1))
            expected = np.array([0, 0, 100, 200, 255, 255],
                                dtype=np.uint8).reshape(-1, 1)
            np.testing.assert_equal(w.data, expected)
        finally:
            os.remove(filename)
            os.removedirs(path)

    def test_vmin_equal_vmax(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "testdata.wav")
        data = np.array([-100, 0, 100, 200, 300, 325])

        wavio.write(filename, data, 44100, sampwidth=1, scale=(200, 200))
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 1)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            w = wavio.read(filename)
            self.assertEqual(w.rate, 44100)
            self.assertEqual(w.sampwidth, 1)
            self.assertEqual(w.data.dtype, np.uint8)
            self.assertEqual(w.data.shape, (len(data), 1))
            expected = np.array([0, 0, 0, 0, 0, 0],
                                dtype=np.uint8).reshape(-1, 1)
            np.testing.assert_equal(w.data, expected)
        finally:
            os.remove(filename)
            os.removedirs(path)


if __name__ == '__main__':
    unittest.main()

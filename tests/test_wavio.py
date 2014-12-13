
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
        wavio.writewav24(filename, 44100, data1)
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 1)
            self.assertEqual(f.getsampwidth(), 3)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            rate, sampwidth, data = wavio.readwav(filename)
            self.assertEqual(rate, 44100)
            self.assertEqual(sampwidth, 3)
            self.assertEqual(data.dtype, np.int32)
            self.assertEqual(data.shape, (len(data1), 1))
            np.testing.assert_equal(data, data1)
        finally:
            os.remove(filename)
            os.removedirs(path)

    def test2(self):
        path = tempfile.mkdtemp()
        filename = os.path.join(path, "test2data.wav")
        data2 = data1.reshape(-1, 2)
        wavio.writewav24(filename, 44100, data2)
        try:
            f = wave.open(filename, 'r')
            self.assertEqual(f.getnchannels(), 2)
            self.assertEqual(f.getsampwidth(), 3)
            self.assertEqual(f.getframerate(), 44100)
            f.close()

            rate, sampwidth, data = wavio.readwav(filename)
            self.assertEqual(rate, 44100)
            self.assertEqual(sampwidth, 3)
            self.assertEqual(data.dtype, np.int32)
            self.assertEqual(data.shape, data2.shape)
            np.testing.assert_equal(data, data2)
        finally:
            os.remove(filename)
            os.removedirs(path)


if __name__ == '__main__':
    unittest.main()

import unittest

import scipy

from feature_extraction_module import *


class TestFeatureExtractor(unittest.TestCase):
    test_data = []
    test_data_spectre = []

    def setUp(self):
        self.test_data = np.linspace(-np.pi * 100, np.pi * 100, 500)
        self.test_data_spectre = scipy.fft(self.test_data)

    def test_energy(self):
        energy = Energy()
        seq = np.random.random_integers(-573, 573, (1000, 3))
        self.assertGreater(energy.get(seq), 0)

    def test_zero_crossing_rate(self):
        zcr = ZeroCrossingRate()
        self.assertGreater(zcr.get(np.sin(self.test_data)), 200.0 / len(self.test_data))

    def test_first_order_autocorrelation(self):
        autocor = Autocorrelation()
        seq = np.arange(50)
        self.assertEqual(autocor.get(seq), 784.0)

    def test_spectal_centroid(self):
        spcentroid = SpectralCentroid()
        self.assertAlmostEqual(spcentroid.get(self.test_data_spectre), 250.0)

    def test_spectral_smoothness(self):
        spsmoothness = SpectralSmoothness()
        self.assertAlmostEqual(spsmoothness.get(self.test_data_spectre), 235.2160775)

    def test_spectral_spread(self):
        spspread = SpectralSpread()
        spcentroid = SpectralCentroid()
        a = spcentroid.get(self.test_data_spectre)
        spspread.get(self.test_data_spectre, [a])


if __name__ == '__main__':
    unittest.run()

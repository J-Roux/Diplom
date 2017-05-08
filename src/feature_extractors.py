import numpy as np
import unittest
from abc import ABCMeta, abstractmethod
import scipy

class FeatureExtractorModel:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def get(self, data, params=None):
        pass

    def check(self,data):
        if not isinstance(data, np.ndarray):
            raise TypeError("input is not array")
    
    def normalize(self, result, data):
        return result / float(len(data))

class SpectralFeature:
    __metaclass__ = ABCMeta
    
class TimeFeature:
    __metaclass__ = ABCMeta

class ZeroCrossingRate(FeatureExtractorModel, TimeFeature):
    def get(self, data, params=None):
        self.check(data)
        return self.normalize( ((data[:-1] * data[1:]) < 0).sum(), data )

class Energy(FeatureExtractorModel, TimeFeature):
    
    def get(self, data, params=None):
        self.check(data)
        return self.normalize(np.sum(np.power(data, 2)), data )

class Autocorrelation(FeatureExtractorModel, TimeFeature):

    def get(self, data, params=None):
        self.check(data)
        return self.normalize((data[:-1] * data[1:]).sum(), data)

class SpectralCentroid(FeatureExtractorModel, SpectralFeature):

    def get(self, data, params=None):
        self.check(data)
        data = np.abs(data)
        return (data * np.arange(len(data))).sum() / float(data.sum())

class SpectralSmoothness(FeatureExtractorModel, SpectralFeature):

    def get(self, data, params=None):
        self.check(data)
        data = np.abs(data)
        data = 20 *  np.log(data)  
        return (2 * data[1:-1] - data[:-2] - data[2:]).sum() / 3

class SpectralSpread(FeatureExtractorModel, SpectralFeature):

    def get(self, data, params=None):
        data = np.abs(data)
        spectral_centroid = params[0]
        return np.sqrt((np.power(np.arange(len(data)) - spectral_centroid, 2) * data ).sum() / data.sum())

class SpectralDissymmetry(FeatureExtractorModel, SpectralFeature):

    def get(self, data, params=None):
        data = np.abs(data)
        spectral_centroid = params[0]
        return np.sqrt(np.abs((np.power(np.arange(len(data)) - spectral_centroid, 3) * data ).sum() / data.sum()))

class TestFeatureExtractor(unittest.TestCase):

    test_data = np.linspace( -np.pi * 100, np.pi * 100, 500)
    test_data_spectre = scipy.fft(test_data)

    def test_energy(self):
        energy = Energy()
        seq = np.random.random_integers(-573, 573, (1000, 3))
        self.assertGreater(energy.get(seq) , 0)
        
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
        self.assertAlmostEqual( spsmoothness.get(self.test_data_spectre),  235.2160775)

    def test_spectral_spread(self):
        spspread = SpectralSpread()
        spcentroid = SpectralCentroid()
        a  = spcentroid.get(self.test_data_spectre)
        spspread.get(self.test_data_spectre, [a])

class FeatureExtractor:

    time_feature_models = {}
    spectre_feature_models = {}
    results = {}
    def set_models(self, models):
        time_feature_model_keys = filter( lambda x : isinstance(x(), TimeFeature), models) 
        time_feature_model_values = [ models[i] for i in time_feature_model_keys ]
        for k, v in zip(time_feature_model_keys, time_feature_model_values):
            self.time_feature_models[k] = v
        self.spectre_feature_models = models
        for i in time_feature_model_keys:
            del self.spectre_feature_models[i]

    def eval_models(self, extractors, data):
        for i in range(np.amax( map( lambda x : len(x), extractors.values())) + 1):
            for feature_extractor in  filter( lambda x : len(extractors[x]) == i, extractors): 
                self.results[feature_extractor] = feature_extractor().get(data,
                        map( lambda x : self.results[x] if x in self.results else None, extractors[feature_extractor]))

    def get(self, data):
        self.eval_models(self.time_feature_models, data)
        spectre_feature = scipy.fft(data)
        self.eval_models(self.spectre_feature_models, spectre_feature)
        



if __name__ == '__main__': 

    test_data = np.linspace( -np.pi * 100, np.pi * 100, 1000000)
    test_data_spectre = scipy.fft(test_data)
    
    models = { 
               Energy : [], 
               ZeroCrossingRate : [], 
               Autocorrelation : [],
               SpectralCentroid : [],
               SpectralSmoothness : [SpectralCentroid], 
               SpectralDissymmetry : [SpectralCentroid]
             }
    feature_extractor = FeatureExtractor()
    feature_extractor.set_models(models)
    feature_extractor.get(test_data)    
    
    print feature_extractor.results 
    #unittest.main()


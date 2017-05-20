from scipy import signal

from feature_extraction_module import *
from feature_processing_module import FeatureProcessing
from preprocessing_module import PreprocessingModule
from read_wav_module import WavModule
from spectral_transform_module import SpectralTransformer, SpectralTrack

models = {
    Energy: [],
    ZeroCrossingRate: [],
    Autocorrelation: [],
    SpectralCentroid: [],
    SpectralSmoothness: [],
    SpectralSpread: [SpectralCentroid],
    SpectralDissymmetry: [SpectralCentroid],
    Rolloff: [],
    LinearRegression: [],
    SFM: [],
    SCF: []
}


class MusicFeatureExtractor:
    preprocessing_module = PreprocessingModule(alpha=0.99,
                                               cut_end=0.2,
                                               cut_start=0.2,
                                               overlap=0.1,
                                               frame_size_sec=5)
    spectral_transformer = SpectralTransformer(decay=0.99,
                                               level=4,
                                               rate=16,
                                               window=signal.hamming(1024))
    feature_extractor = FeatureExtractorModule(models=models, nceps=16)
    feature_processing = FeatureProcessing(with_kurtosis=False,
                                           with_skew=False)
    read_wav_module = WavModule()

    def __int__(self,
                read_wav_module=WavModule(),
                preprocessing_module=PreprocessingModule(alpha=0.99,
                                                         cut_end=0.2,
                                                         cut_start=0.2,
                                                         overlap=0.1,
                                                         frame_size_sec=5),
                spectral_transformer=SpectralTransformer(decay=0.99,
                                                         level=4,
                                                         rate=16,
                                                         window=signal.hamming(1024)),
                feature_extractor=FeatureExtractorModule(models=models, nceps=16),
                feature_processing=FeatureProcessing(with_kurtosis=False,
                                                     with_skew=False)):

        self.read_wav_module = read_wav_module
        self.preprocessing_module = preprocessing_module
        self.spectral_transformer = spectral_transformer
        self.feature_extractor = feature_extractor
        self.feature_processing = feature_processing

    def read(self, file_name, label):
        if not file_name.endswith('.wav'):
            self.read_wav_module.create_wav(file_name)
        return self.read_wav_module.read_wav(file_name, label)

    def preprocessing(self, track):
        track = self.preprocessing_module.cutting(track)
        track = self.preprocessing_module.scale(track)
        track = self.preprocessing_module.stereo_to_mono(track)
        track = self.preprocessing_module.filter(track)
        return self.preprocessing_module.framing(track)

    def tranform(self, tracks):
        stracks = []
        for j in tracks:
            spectral = self.spectral_transformer.short_time_fourier(j)
            percussion = self.spectral_transformer.percussion_correlogramm(j)
            stracks.append(SpectralTrack(track=j,
                                         spectral_data=spectral,
                                         percussion_data=percussion))
        return stracks

    def extract(self, tracks):
        return map(lambda x: self.feature_extractor.extract_feature(x), tracks)

    def post_processing(self, tracks):
        return map(lambda x: self.feature_processing.process_feature(x), tracks)

    def get_feature(self, file_name, label):
        track = self.read(file_name, label)
        tracks = self.preprocessing(track)
        tracks = self.tranform(tracks)
        tracks = self.extract(tracks)
        tracks = self.post_processing(tracks)
        return tracks

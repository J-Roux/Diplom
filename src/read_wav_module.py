import subprocess

from scipy.io.wavfile import read

from track import Track


class WavModule:
    @staticmethod
    def __check_format(file_name, file_format):
        # type: (str, str) -> None
        if not file_name.endswith(file_format):
            raise EnvironmentError('invalid file format')

    def create_wav(self, file_name, file_format):
        WavModule.__check_format(file_name, file_format)
        bash_command = 'lame --decode ' + file_name + ' ' + file_name + '.wav'
        subprocess.call(bash_command.split())

    def read_wav(self, file_name, genre):
        WavModule.__check_format(file_name, 'wav')
        return Track(read(file_name), genre)

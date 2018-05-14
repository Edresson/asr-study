from datasets import DatasetParser

import os
import re
import librosa
import codecs

import numpy as np


class LibriSpeech(DatasetParser):
    """ LibriSpeech dataset reader and parser

    More about the dataset: http://www.openslr.org/resources/12/
    """

    def __init__(self, dataset_dir=None, name='LibriSpeech', **kwargs):

        dataset_dir = dataset_dir or 'data/LibriSpeech'

        super(LibriSpeech, self).__init__(dataset_dir, name, **kwargs)

    def _iter(self):

        subdatasets = [ 'train-clean-100',
                        'train-clean-360',
                        'train-other-500',
                        'dev-clean',
                        'dev-other',
                        'test-clean',
                        'test-other']

        for sub in subdatasets:
            
            try:
                dir_data = os.path.join(os.path.abspath(self.dataset_dir),sub+'/')
                speakers_dir = os.listdir(dir_data)
            except :
                self._logger.error('Skipping LibriSpeech Sub-dataset: %s directory does not exist'% sub)
                continue
            
            for speaker_path in speakers_dir:

                root_path = os.path.join(os.path.abspath(dir_data),
                                         speaker_path)
                for arc in os.listdir(root_path):
                    root_arc_path = os.path.join(os.path.abspath(root_path),arc)
                    labels_file = os.path.join(root_arc_path, speaker_path+'-'+arc+'.trans.txt')

                    
                    for line in codecs.open(labels_file, 'r', encoding='utf8'):

                        split = line.strip().split(' ')
                        file_id = split[0]
                        speaker_id = speaker_path

                        label = line.strip(file_id+' ').lower()
                            

                        audio_file = os.path.join(
                            root_arc_path, file_id+'.flac')

                        try:
                            duration = librosa.audio.get_duration(filename=audio_file)
                        except IOError:
                            self._logger.error('File %s not found' % audio_file)
                            continue

                        yield {'duration': duration,
                               'input': audio_file,
                               'label': label,
                               'speaker': speaker_id}
                        

            


    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f''' % (len(dl['speaker']), sum(dl['duration']))

        return report

                

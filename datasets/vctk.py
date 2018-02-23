from datasets import DatasetParser

import os
import librosa
import codecs


class VCTK(DatasetParser):
    """ VCTK-Corpus  dataset reader and parser

    More about the dataset: https://voice.mozilla.org/data
    """

    def __init__(self, dataset_dir=None, name='vctk', **kwargs):

        dataset_dir = dataset_dir or 'data/vctk'

        super(VCTK, self).__init__(dataset_dir, name, **kwargs)

    def _iter(self):
            
            try:
                parent_path = os.path.join(os.path.abspath(self.dataset_dir))
                txts_list = os.listdir(parent_path+'/txt/')
            except :
                self._logger.error('Skipping VCTK-Corpus: %s directory does not exist'% parent_path+'/txt/')
            
            for p in txts_list:
                txt_list = os.listdir(parent_path+'/txt/'+p)

                for txt in txt_list:
                    
                    with open(parent_path+'/txt/'+p+'/'+txt, 'rt') as f:
                    
                        label = f.readline()
                        file = txt.split('.')[0]
                        audio_file = parent_path+'/wav48/'+p+'/'+file+'.wav'
                        speaker_id = p[:-1]

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

from datasets import DatasetParser

import os
import librosa
import codecs


class Tatoeba(DatasetParser):
    """ Tatoeba corpus  dataset reader and parser

    More about the dataset: https://voice.mozilla.org/data
    """

    def __init__(self, dataset_dir=None, name='tatoeba', **kwargs):

        dataset_dir = dataset_dir or 'data/tatoeba'

        super(Tatoeba, self).__init__(dataset_dir, name, **kwargs)

    def _iter(self):
        
        csv_file = os.path.join(self.dataset_dir,'sentences_with_audio.csv')

        for line in codecs.open(csv_file , 'r', encoding='utf8'):
            try:



                split = line.strip().split('\t')
                #split = line.strip().split(' ')
                #split[:] = [item for item in split if item != '']

                #Ignore first line csv arquive.
                if split[0] == 'id':
                    continue
                audio_file = os.path.join(self.dataset_dir,'audio/'+split[1]+'/'+split[0]+'.mp3')
                """text = ''
                for i in split[2:]:
                    text = text+i+' '"""
                            
                
                label = split[2].lower()
                speaker_id = split[1]
                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    self._logger.error('File %s not found' % audio_file)
                    continue

                yield {'duration': duration,
                        'input': audio_file,
                        'label': label,
                        'speaker': speaker_id}
                 
            except :
                self._logger.error('Skipping Line: %s'% line)



    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f''' % (len(dl['speaker']), sum(dl['duration']))

        return report

from datasets import DatasetParser

import os
import librosa
import codecs


class CVC(DatasetParser):
    """ Common Voice corpus v1  dataset reader and parser

    More about the dataset: https://voice.mozilla.org/data
    """

    def __init__(self, dataset_dir=None, name='cvc', **kwargs):

        dataset_dir = dataset_dir or 'data/cv_corpus_v1'

        super(CVC, self).__init__(dataset_dir, name, **kwargs)

    def _iter(self):
        #common voice valid train
        train_label = os.path.join(self.dataset_dir,'cv-valid-train.csv')
        dataset_type = 'train'

        for line in codecs.open(train_label, 'r', encoding='utf8'):
            try:



                split = line.strip().split(',')

                #Ignore first line csv arquive.
                if split[0] == 'filename':
                    continue

                audio_file = os.path.join(self.dataset_dir,split[0])
                audio_name = split[0].split('/')[1]


                label = split[1].lower()

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    self._logger.error('File %s not found' % audio_file)
                    continue

                yield {'duration': duration,
                       'input': audio_file,
                       'label': label,
                       'audio_file': audio_name,
                       'dataset': dataset_type}
            except :
                self._logger.error('Skipping Line: %s'% line)


        #common voice other train
        train_label = os.path.join(self.dataset_dir,'cv-other-train.csv')
        dataset_type = 'train'

        for line in codecs.open(train_label, 'r', encoding='utf8'):

            try:


                split = line.strip().split(',')

                audio_file = os.path.join(self.dataset_dir,split[0])
                #Ignore first line csv arquive.
                if split[0] == 'filename':
                    continue
                audio_name = split[0].split('/')[1]


                label = split[1].lower()

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    self._logger.error('File %s not found' % audio_file)
                    continue

                yield {'duration': duration,
                       'input': audio_file,
                       'label': label,
                       'audio_file': audio_name,
                       'dataset': dataset_type}
            except:
                self._logger.error('Skipping Line: %s'% line)
                continue

        #common voice valid test
        test_label = os.path.join(self.dataset_dir,'cv-valid-test.csv')
        dataset_type = 'test'
        for line in codecs.open(test_label, 'r', encoding='utf8'):

            try:

                split = line.strip().split(',')

                audio_file = os.path.join(self.dataset_dir,split[0])
                #Ignore first line csv arquive.
                if split[0] == 'filename':
                    continue
                audio_name = split[0].split('/')[1]


                label = split[1].lower()

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    self._logger.error('File %s not found' % audio_file)
                    continue

                yield {'duration': duration,
                       'input': audio_file,
                       'label': label,
                       'audio_file': audio_name,
                       'dataset': dataset_type}
            except:
                self._logger.error('Skipping Line: %s'% line)
                continue


        #common voice other test

        test_label = os.path.join(self.dataset_dir,'cv-other-test.csv')
        dataset_type = 'valid'


        for line in codecs.open(test_label, 'r', encoding='utf8'):

            try:

                split = line.strip().split(',')
                #Ignore first line csv arquive.
                if split[0] == 'filename':
                    continue

                audio_file = os.path.join(self.dataset_dir,split[0])

                audio_name = split[0].split('/')[1]


                label = split[1].lower()

                try:
                    duration = librosa.audio.get_duration(filename=audio_file)
                except IOError:
                    self._logger.error('File %s not found' % audio_file)
                    continue

                yield {'duration': duration,
                       'input': audio_file,
                       'label': label,
                       'audio_file': audio_name,
                       'dataset': dataset_type}

            except :
                self._logger.error('Skipping Line: %s'% line)
                continue


    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f''' % (len(dl['audio']), sum(dl['duration']))

        return report

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datasets import DatasetParser
from datasets import CVC

from utils.generic_utils import get_from_module

class ENSD(DatasetParser):
    """ English Speech dataset reader and parser

    This dataset is a combination of four biggers datasets (voxforge, CommonVoice, LibriSpeech,VCTK-Corpus,TED-LIUM2,Tatoeba). The dataset was divided in the following
    way:
        * Train: voxforge,CommonVoice,LibriSpeech,VCTK-Corpus,TED-LIUM2,Tatoeba
        * Valid: Common Voice valid test( cv-valid-test )
        * Test: Common Voice valid test( cv-other-test )

    """

    def __init__(self, dataset_dir=None, name='ensd', **kwargs):

        dataset_dir = dataset_dir or {'cv_corpus_v1': None,
                                      'LibriSpeech': None,
                                      'voxforge_en': None,
                                      'vctk': None,
                                      'tedlium2': None,
                                      'tatoeba': None
                                      }

        super(ENSD, self).__init__(dataset_dir, name, **kwargs)

    @property
    def dataset_dir(self):
        """Filepath to the dataset directory"""
        return self._dataset_dir

    @dataset_dir.setter
    def dataset_dir(self, value):
        """Filepath to the dataset directory"""

        if value is None:
            raise ValueError("You must set the variable dataset_dir"
                             " (the location of dataset) before continue")

        if not isinstance(value, dict):
            raise ValueError("dataset_dir must be a dictionary")

        for key in ('cv_corpus_v1', 'voxforge_en','LibriSpeech','vctk','tedlium2','tatoeba'):
            if key not in value:
                raise KeyError("dataset_dir must have the key %s" % key)


        self._dataset_dir = value

    def _iter(self):

        for name, path in self.dataset_dir.items():

            if name == 'cv_corpus_v1':
                continue

            try:
                dataset_cls = get_from_module('datasets*', name, regex=True)
                dataset = dataset_cls(dataset_dir=path)
                print(dataset)
                for d in dataset._iter():
                    yield {'duration': d['duration'],
                           'input': d['input'],
                           'label': d['label'],
                           'speaker': '%s_%s' % (str(dataset), d['speaker']),
                           'dataset': 'train'}
            except ValueError as e:
                self._logger.warning('Skipping dataset %s: %s' % (name, e))

        # Test and valid set
        cvc = CVC(dataset_dir=self.dataset_dir['cv_corpus_v1'])
        for d in cvc._iter():
            yield {'duration': d['duration'],
                   'input': d['input'],
                   'label': d['label'],
                   'audio_file': d['audio_file'],
                   'dataset': d['dataset']}

    def _report(self, dl):
        report = '''General information:
           Number of utterances: %d
           Total size (in seconds) of utterances: %.f
           ''' % (len(dl['input']),sum(dl['duration']))


        return report

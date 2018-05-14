from datasets import DatasetParser

import os
import re
import librosa
import codecs

import numpy as np


from pydub import AudioSegment as audio
    



class TedLium2(DatasetParser):
    """ TED-LIUM corpus release 2 dataset reader and parser
    """

    def __init__(self, dataset_dir=None, name='TEDLIUM', **kwargs):

        dataset_dir = dataset_dir or 'data/tedlium2'

        super(TedLium2, self).__init__(dataset_dir, name, **kwargs)

    def _iter(self):

        subdatasets = [ 'train',
                        'test']

        for sub in subdatasets:
            
            try:
                parent_path = os.path.join(os.path.abspath(self.dataset_dir),sub)
                stm_list = os.listdir(parent_path+'/stm/')
            except :
                self._logger.error('Skipping LibriSpeech Sub-dataset: %s directory does not exist'% sub)
                continue
            

            starts =[]
            ends=[]
            wave_files =[]
            labels = []
            speakers_id  = []
            for stm in range(len(stm_list)):
                with open(parent_path+'/stm/'+stm_list[stm], 'rt') as f:
                    
                    records = f.readlines()
                    
                    for record in records:
                        field = record.split()
                        
                        #speaker id 
                        speakers_id.append(stm)

                        # label index
                        text = ''
                        for i in field[6:]:
                            text = text+i+' '
                            
                        if text  == 'ignore_time_segment_in_scoring ':
                            continue
                        
                        labels.append(text.lower())
                        
                        # wave file name
                        data_directory = parent_path + '/wav/%s.sph' % field[0]
                        wave = parent_path + '/sph/%s.sph.wav' % field[0]
                        wave_file = [ wave,data_directory]
                        
                        wave_files.append(wave_file)

                        # start, end info
                        start, end = float(field[3]), float(field[4])
                        starts.append(start)
                        ends.append(end)
            
            for i, (wave_file, label, start, end,speaker_id) in enumerate(zip(wave_files, labels, starts, ends,speakers_id)):

                
                # print info
                #print("TEDLIUM corpus preprocessing (%d / %d) - '%s-%.2f]" % (i, len(wave_files[0]), wave_file, start))
                
                sound = audio.from_wav(wave_file[0])
                segment_sound = sound[start*1000:end*1000]
                audio_file = wave_file[1]+'-'+str(i)+'.wav'
                segment_sound.export(audio_file, format="wav")
                
                

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

                

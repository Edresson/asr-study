echo "Downloading English datasets. This may take a while"
echo "Downloading Common Voice dataset:"
wget -c -q --show-progress -O ./cv_corpus_v1.tar.gz  https://common-voice-data-download.s3.amazonaws.com/cv_corpus_v1.tar.gz 


echo "Downloading VoxForge English dataset:"
wget -c -q --show-progress -O ./voxforge_corpus_v1.0.0.tar.gz https://s3.us-east-2.amazonaws.com/common-voice-data-download/voxforge_corpus_v1.0.0.tar.gz

echo "Downloading LibreSpeech dataset:"
wget -c -q --show-progress -O .train-clean-100.tar.gz http://www.openslr.org/resources/12/train-clean-100.tar.gz
wget -c -q --show-progress -O .train-clean-360.tar.gz http://www.openslr.org/resources/12/train-clean-360.tar.gz
wget -c -q --show-progress -O .train-other-500.tar.gz http://www.openslr.org/resources/12/train-other-500.tar.gz
wget -c -q --show-progress -O .dev-clean.tar.gz http://www.openslr.org/resources/12/dev-clean.tar.gz
wget -c -q --show-progress -O .dev-other.tar.gz http://www.openslr.org/resources/12/dev-other.tar.gz
wget -c -q --show-progress -O .test-clean.tar.gz http://www.openslr.org/resources/12/test-clean.tar.gz
wget -c -q --show-progress -O .test-other.tar.gz http://www.openslr.org/resources/12/test-other.tar.gz


echo "Downloading TED-LIUM corpus release 2 dataset:" 
wget -c -q --show-progress -O .TEDLIUM_release2.tar.gz http://www-lium.univ-lemans.fr/sites/default/files/TEDLIUM_release2.tar.gz


echo "Downloading CSTR VCTK Corpus dataset:" 
wget -c -q --show-progress -O .VCTK-Corpus.tar.gz  http://homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz


echo "Downloading Tatoeba dataset:" 
wget -c -q --show-progress -O .tatoeba_audio_eng.zip https://downloads.tatoeba.org/audio/tatoeba_audio_eng.zip





echo "Extracting Common Voice dataset..."
tar -xzf cv_corpus_v1.tar.gz



echo "Extracting VoxForge dataset..."
#exctract.py for recursive  extraction tar.gz 
python extract.py voxforge_corpus_v1.0.0
mv archive voxforge_en

echo "Extracting LibreSpeech dataset..."
tar -xzf train-clean-100.tar.gz
tar -xzf train-clean-360.tar.gz
tar -xzf train-other-500.tar.gz
tar -xzf dev-clean.tar.gz
tar -xzf dev-other.tar.gz
tar -xzf test-clean.tar.gz
tar -xzf test-other.tar.gz


echo "Extracting TED-LIUM corpus release 2 dataset..."
tar -xzf TEDLIUM_release2.tar.gz
mv TEDLIUM_release2 tedlium2
cd tedlium2
mkdir train/wav
mkdir test/wav

#.sph for .wav: Sox is required: apt install sox
find -type f -name '*.sph' | awk '{printf "sox -t sph %s -b 16 -t wav %s\n", $0, $0".wav" }' | bash
cd ..

echo "Extracting VCTK-Corpus dataset..."
tar -xzf VCTK-Corpus.tar.gz
mv VCTK-Corpus vctk

echo "Extracting Tatoeba dataset..."
unzip tatoeba_audio_eng.zip #unzip required: apt install unzip
mv tatoeba_audio_eng tatoeba


echo "Finished."

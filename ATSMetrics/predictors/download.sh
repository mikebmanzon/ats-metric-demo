### Download datasets for pretraining
mkdir -p datasets/
cd datasets

# Download BoolQ from https://github.com/google-research-datasets/boolean-questions and place it into `pretrain_data/raw/boolq`
mkdir -p boolq
wget -P boolq https://storage.cloud.google.com/boolq/train.jsonl
wget -P boolq https://storage.cloud.google.com/boolq/dev.jsonl

# Download MCTest
wget https://github.com/mcobzarenco/mctest/archive/master.zip
unzip master.zip
mv mctest-master/data/MCTest .
mv MCTest mctest
rm master.zip
rm -r mctest-master

# Download MultiRC
wget https://cogcomp.seas.upenn.edu/multirc/data/mutlirc-v2.zip
unzip mutlirc-v2.zip
mv splitv2 multirc
rm mutlirc-v2.zip

# Download RACE
wget http://www.cs.cmu.edu/~glai1/data/race/RACE.tar.gz
tar -zxvf RACE.tar.gz
mv RACE race
rm RACE.tar.gz

mkdir -p squadv2
cd squadv2
wget -P squadv2 https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -P squadv2 https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json
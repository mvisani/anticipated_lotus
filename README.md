# LOTUS anticipated

## How to run
First thing to do is to download LOTUS from Zenodo: 
```bash
wget https://zenodo.org/record/7534071/files/230106_frozen_metadata.csv.gz
mv ./230106_frozen_metadata ./data/molecules/
wget http://classyfire.wishartlab.com/system/downloads/1_0/chemont/ChemOnt_2_1.obo.zip
mv ./ChemOnt_2_1.obo.zip ./data/molecules/
# unzip the file
unzip ./data/molecules/ChemOnt_2_1.obo.zip

wget https://raw.githubusercontent.com/mwang87/NP-Classifier/master/Classifier/dict/index_v1.json
mv ./index_v1.json ./data/molecules/NPClassifier_index.json
```

We can then create and activate the environment : 
```bash 
conda env create --file environment.yml
conda activate grape
```

We can now proceed to prepare the data. To do so we need simple run: TODO
```bash
./prepare_data.sh
```
TODO


The notebooks should be run in the following order : 
* `prepare_species`
* `prepare_mol_to_chemont`
* `prepare_NPClassifier`
* `prepare_lotus`
* `prepare_graph`
* `prepare_merge_ncbi`


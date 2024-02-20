# LOTUS anticipated

## Download preprocessed data
For running the analysis you can download the data from Zenodo: TODO
```bash
TODO
``` 

### Run the analysis
You may choose which ever model you want from the [`src/models`](https://github.com/mvisani/anticipated_lotus/tree/main/src/models) folder. If you want you can also implement your own model. The available models are : 
- [DecisionTree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.fit)
- [RandomForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/python/python_api.html#xgboost.XGBClassifier)
- [LightGBM](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html)

Before running the script, you should change the variable `max_evals` in the `run_decision_tree.py` file to 100 or more.

## Prepare the data locally

First thing to do is to download LOTUS from Zenodo: 
```bash
wget https://zenodo.org/record/7534071/files/230106_frozen_metadata.csv.gz
mv ./230106_frozen_metadata.csv.gz ./data/molecules/230106_frozen_metadata.csv.gz
wget http://classyfire.wishartlab.com/system/downloads/1_0/chemont/ChemOnt_2_1.obo.zip
mv ./ChemOnt_2_1.obo.zip ./data/molecules/ChemOnt_2_1.obo.zip
# unzip the file
unzip ./data/molecules/ChemOnt_2_1.obo.zip
mv ./ChemOnt_2_1.obo ./data/molecules/ChemOnt_2_1.obo

wget https://raw.githubusercontent.com/mwang87/NP-Classifier/master/Classifier/dict/index_v1.json
mv ./index_v1.json ./data/molecules/NPClassifier_index.json
```

We can then create and activate the environment : 
```bash 
conda env create --file environment.yml
conda activate grape
```

### Prepare molecules
First run the following command to prepare lotus, the molecules and the molecule ontology:
```bash
python prepare_data/prepare_lotus.py
python prepare_data/prepare_mol_to_chemont.py
python prepare_data/prepare_NPClassifier.py
```

### Prepare species
The species edge list will take much longer to prepare. This is due to the fact that for each species, the script will fetch the data on wikidata and the create an edge list from there. 
Use the command below to prepare the species edge list. :warning: this usually takes about 7 hours to run. So be patient :). In case you need to stop the process (using Ctrl+C), the function is equipped with a cache system that will allow you to restart the process from where you stopped it. 
```bash
python prepare_data/prepare_species.py 
```

### Prepare graph
Now we need to prepare the data to be suitable for [grape](https://github.com/AnacletoLAB/grape) library. 
```bash
python prepare_data/prepare_graph.py
```

Finally we need to merge the data from NCBI and LOTUS. 
```bash
python prepare_data/prepare_merge_ncbi.py
```

Once this is done you should have different graph as such : 
```shell
.
├── full_lotus_edges.csv
├── full_lotus_nodes.csv
├── lotus
│   ├── lotus_edges.csv
│   └── lotus_nodes.csv
├── lotus_with_ncbi_clean_edges.csv
├── lotus_with_ncbi_clean_nodes.csv
├── lotus_with_ncbi_edges.csv
├── lotus_with_ncbi_nodes.csv
├── molecules
│   ├── 230106_frozen_metadata.csv.gz
│   ├── ChemOnt_2_1.obo
│   ├── NPClassifier_index.json
│   ├── chemont_edges.csv
│   ├── chemont_nodes.csv
│   ├── mol_to_chemont_edges.csv
│   ├── mol_to_chemont_nodes.csv
│   ├── mol_to_np_edges.csv
│   └── mol_to_np_nodes.csv
└── species
    ├── all_species
    │   ├── wd:Q1000854
    │   │   ├── f8e4a60d0bd3c1ce7b5195129445f73b822bdf1b041b14e1256f6d5c8bcc441e.csv.gz
    │   │   └── f8e4a60d0bd3c1ce7b5195129445f73b822bdf1b041b14e1256f6d5c8bcc441e.csv.gz.metadata
```

In our case we will use the `lotus_with_ncbi_clean` graph. But all the other graphs are also available in case you want to explore them.

### Run the analysis
This is not possible at the moment because the module `ensmallen` from [grape](https://github.com/AnacletoLAB/grape) does not support the HyperSketching yet. Once it will be available, we recommend to first run the `run_model_dummy.py` script with `max_eval=1`. This will first create the sketching of the different holdouts of training and testing. Then you can run the script with `max_eval=100` or more to find the best parameters of the model.

### Train the model
Once the best parameters are found, you can train the model using the `train_model.py` script. You should manually change the parameters and the model in the script according to the best parameters found.


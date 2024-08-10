# NAG2G: Node-Aligned Graph-to-Graph Model


Welcome to the NAG2G (Node-Aligned Graph-to-Graph) repository! NAG2G is a state-of-the-art neural network model for retrosynthesis prediction.

## Research Paper

For detailed information about the method and experimental results, please refer to [our research paper](https://arxiv.org/abs/2309.15798).

## Platform

[Uni-Retro platform](https://app.bohrium.dp.tech/retro-synthesis/workbench/): A multi-step retrosynthesis platform that integrates the NAG2G algorithm.

## Environment Setup

To begin working with NAG2G, you'll need to set up your environment. Below is a step-by-step guide to get you started:

```bash
# Install Uni-Core
git clone https://github.com/dptech-corp/Uni-Core
cd Uni-Core
pip install .
cd -

# Install Unimol plus
cd unimol_plus
pip install .
cd -

# Install additional dependencies
pip install rdchiral transformers tokenizers omegaconf rdkit
```

## Datasets and Pretrained Weights

You can obtain the dataset USPTO-50k and pretrained model weights for USPTO-50k from [the Google Drive](https://drive.google.com/drive/folders/1lZOLRGyZy18EVow7gyxtKWvs_yuwlIE3?usp=sharing):


## Model Validation

To validate the NAG2G model with the provided weights, follow the instructions below:

When using a dataset that does not include reactants, you need to modify the `valid.sh` script. Specifically, add the ` --no_reactant` command in line 95 in the code.

When using your own dataset, please modify the `data_path` in the `valid.sh` script.

```bash
# Execute the validation script with the specified checkpoint file
sh valid.sh path2weight/NAG2G_unimolplus_uspto_50k_20230513-222355/checkpoint_last.pt
```



## Data Preprocessing Instructions

If you need to regenerate the dataset, please refer to the code inside the `data_preprocess` directory.

```bash
cd data_preprocess
python lmdb_preprocess <input_csv> <output_lmdb>
```

Two sample CSV files are provided for reference:
- `sample.csv`: This sample includes given reactants.
- `sample_without_reactants.csv`: This sample does not include given reactants.



----
For any questions or issues, please open an issue on our GitHub repository.

Thank you for your interest in NAG2G!


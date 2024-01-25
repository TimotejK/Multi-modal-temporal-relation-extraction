# Multi-modal Learning for Temporal Relation Extraction in Clinical Texts

This repository contains the source code for the paper titled "Multi-modal Learning for Temporal Relation Extraction in Clinical Texts." The project focuses on temporal relation extraction in clinical texts using multi-modal learning techniques.

## Installation

To set up the environment for running the project, please execute the following commands:

```bash
pip install numpy pandas transformers rdflib tensorboard tensorboardX gens networkx matplotlib scispacy markupsafe==2.0.1 pytorch_lightning
# install scispacy dictionary
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz
# install pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
# install torch geometric
pip install --no-index pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.11.0+cu113.html
pip install torch_geometric
```

## Usage

To run the project, use the following command:

```bash
python main.py <test-scenario>
```

Replace `<test-scenario>` with one of the following arguments based on the desired scenario:

|                            | Scenario 1                 | Scenario 2               | Scenario 3                              |
|----------------------------|----------------------------|--------------------------|-----------------------------------------|
| THYME original relations   | thyme_original_unrealistic | thyme_original_realistic | thyme_original_unrealistic remove_wrong |
| THYME simplified relations | thyme_simple_unrealistic   | thyme_simple_realistic   | thyme_simple_unrealistic remove_wrong   |
| i2b2 2012                  | i2b2_unrealistic           | i2b2                     | i2b2_unrealistic remove_wrong           |
| MACCROBAT                  | maccrobat_unrealistic      | maccrobat                | maccrobat_unrealistic remove_wrong      |

## Datasets and Pretrained Models

Please note that we are unable to provide the datasets or pretrained models used in this project due to licensing restrictions. You can obtain the datasets from the following sources:

- MACCROBAT: [MACCROBAT2018](https://figshare.com/articles/dataset/MACCROBAT2018/9764942)
- i2b2 2012: [N2C2 NLP Shared Task 2012](https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/)
- THYME: [THYME Dataset](https://healthnlp.hms.harvard.edu/center/pages/data-sets.html)

Please download the datasets from the respective sources and ensure compliance with their licensing terms.
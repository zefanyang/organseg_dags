# organseg_dags
This repository contains the codebase aimed at 3D multi-organ segmentation in CT, using directed acyclic graphs and edge skip-connections.

The conference paper is "Graph-based Regional Feature Enhancing for Abdominal Multi-Organ Segmentation in CT."


# Usage
## Environment
Install the following essential packages using `pip install` command:
  - torch==1.7.1 (Or install other versions of PyTorch corresponding to your CUDA versions using commands on the [PyTorch website](https://pytorch.org/get-started/previous-versions/).)
  - numpy==1.19.2
  - MedPy==0.4.0

## Implementation
### Data
Store images, labels, and edge maps in the following folders:

`/path/to/data/image/image0002.nii.gz`

`/path/to/data/label/label0002.nii.gz`

`/path/to/data/edge/edge0002.nii.gz`

Make a look-up table using `.jason` format recording image, label and edge map directories of each patient, for example, our `./data/cross_validation.jason`.

### Dataloader
`./cacheio/Dataset.py` contains the class `Dataset`, which is a subclass of `torch.utils.data.Dataset` used for obtaining image, label, and edge map directories of a patient given the look-up table and then loading and augmenting data. Rewrite the class functions to make sure that data can be correctly loaded.

### Training
The process of cross-validation training is defined in `./train_full_scheme.py`. 

The option `--fold=0` specifies the cross-validation fold, and `--cv_json='/path/to/cross_validation.jason'` the path of a look-up table indexing your data at hand. 

Modify hyper-parameters if necessary which are by default `--batch_size=2`, `--optim='adam'`, `--lr=1e-3`, `--weight_decay=3e-4`, `--num_epoch=400`, and `--beta=1` and `--beta2=1` specifying the weights of the segmentation and edge detection objectives respectively.

### Evaluation
The evaluation process defined in `./inference.py` computes the following volumetric metrics: the Dice similarity coefficient, the 95th percentile Hausdorff distance, and the average symmetric surface distance.

Specify the validation fold `--fold` and the path of a look-up table recoding directories of validation data `--cv_json`.

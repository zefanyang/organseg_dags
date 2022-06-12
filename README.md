# organseg_dags
This repository contains the codebase for 3D multi-organ segmentation in CT, using directed acyclic graphs and edge skip-connections.
# Usage
## Environment
Install the following essential packages using `pip install` command:
  - torch==1.7.1 (or select versions corresponding to your CUDA versions on the [PyTorch website](https://pytorch.org/get-started/previous-versions/))
  - numpy==1.19.2
  - MedPy==0.4.0

## Implementation
### Data
Images, labels, and edge maps are stored in the following folders:
`/path/to/data/image/image0002.nii.gz`
`/path/to/data/label/label0002.nii.gz`
`/path/to/data/edge/edge0002.nii.gz`

  - Develop a subclass of `torch.utils.data.Dataset` for your dataset at hand
  - Develop a custom `train.py`

# Publication
The conference paper:
> "Graph-based Regional Feature Enhancing for Abdominal Multi-Organ Segmentation in CT."

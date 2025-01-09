# Uncertainty-aware Fine-tuning of Segmentation Foundation Models (SUM)


Official implementation of **Uncertainty-aware Fine-tuning of Segmentation Foundation Models** (NeurIPS 2024).

[Kangning Liu](https://kangning-liu.github.io/)<sup>1,2</sup>, [Brian Price](https://research.adobe.com/person/brian-price/)<sup>2</sup>, [Jason Kuen](https://research.adobe.com/person/jason-kuen/)<sup>2</sup>, [Yifei Fan](https://openreview.net/profile?id=~Yifei_Fan1)<sup>2</sup>, [Zijun Wei](https://scholar.google.com/citations?user=8l3bFYYAAAAJ&hl=en)<sup>2</sup>, [Luis Figueroa](https://luisf.me/)<sup>2</sup>, [Krzysztof J. Geras](https://cs.nyu.edu/~kgeras/)<sup>1</sup>, [Carlos Fernandez-Granda](https://math.nyu.edu/~cfgranda/)<sup>1</sup>

<sup>1</sup> New York University  
<sup>2</sup> Adobe 

[NeurIPS 2024 Poster](https://neurips.cc/virtual/2024/poster/93500)

[Project Website](https://kangning-liu.github.io/SUM_website/)

## Table of Contents

- [Status Update](#status-update)
  - [Current Progress](#current-progress)
  - [Next Steps](#next-steps)
  - [Known Issues](#known-issues)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Notebook](#notebook)
- [Contact](#contact)



## Status Update

### Current Progress


- [NEW] SUM (HQ-SAM arch.) Provide the training code and inference code of SUM implemented with HQ-SAM architecture [sum_on_hq-sam](sum_on_hq-sam) 

- Main experiments
  - Provided the model building code [build_sam.py](build_sam.py)
  - Provided the key components of uncertainty-aware fine-tuning for the main experiment: 
      - Uncertainty-aware loss [losses.py](utils%2Flosses.py)
      - Uncertainty-aware prompt sampling [interactive_sampling.py](utils%2Finteractive_sampling.py)


### Next Steps
- Main experiments
  - Provide demo Jupyter notebooks
  - Add support for the evaluation dataloader
  - Release model weights trained on the public dataset
  - Provide the full training code

### Known Issues
- Some scripts may require additional dependencies not listed in the prerequisites.
- Documentation is still in progress and may lack detailed instructions for some scripts.

## Prerequisites

The code requires `python>=3.8`, as well as `pytorch>=1.7` and `torchvision>=0.8`. Please follow the instructions [here](https://pytorch.org/get-started/locally/) to install both PyTorch and TorchVision dependencies. Installing both PyTorch and TorchVision with CUDA support is strongly recommended.



## Dataset
*TODO*


## Notebook 

*TODO*




## Contact
For any questions or issues, please contact:
- Kangning Liu - [kangning.liu@nyu.edu](mailto:kangning.liu@nyu.edu)








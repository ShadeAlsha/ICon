# I-Con: A Unifying Framework for Representation Learning
###  ICLR 2025


[![Website](https://img.shields.io/badge/ICon-%F0%9F%8C%90Website-purple?style=flat)](https://aka.ms/i-con) [![Website](https://img.shields.io/badge/ICon-Paper-blue?style=flat)]([https://aka.ms/icon-paper](https://openreview.net/pdf?id=WfaQrKCr4X))  


[Shaden Alshammari](http://shadealsha.github.io),
[John R. Hershey](https://research.google/people/john-hershey/),
[Axel Feldmann](https://feldmann.nyc/),
[William T. Freeman](https://billf.mit.edu/about/bio),
[Mark Hamilton](https://mhamilton.net/)

![ICon Overview Graphic](https://mhamilton.net/images/periodic_table.svg)

**TL;DR**: We introduce a single equation that unifies >20 machine learning methods into a periodic table. We use this framework to make a state-of-the-art unsupervised image classifier.

## Contents
<!--ts-->
   * [Install](#install)
   * [Define Models](#define-model)
   * [Train Models](#train-model)
   * [Evaluate Models](#evaluate-models)
   * [Citation](#citation)
   * [Contact](#contact)
<!--te-->

## Install

To use ICon locally clone the repository:

```shell script
git clone https://github.com/ShadeAlsha/ICon.git
cd ICon
pip install -e .
```


## Defining Models



## Evaluate Models

```shell
cd ICon
python evaluate.py
```

After evaluation, see the results in tensorboard's hparams tab. 

```shell
cd ../logs/evaluate
tensorboard --logdir .
```

Then visit [https://localhost:6006](https://localhost:6006) and click on hparams to browse results.


## Train a Model

```shell
cd ICon
python train.py
```

## Citation

```
@inproceedings{alshammariunifying,
  title={A Unifying Framework for Representation Learning},
  author={Alshammari, Shaden Naif and Hershey, John R and Feldmann, Axel and Freeman, William T and Hamilton, Mark},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

## Contact

For feedback, questions, or press inquiries please contact [Shaden Alshammari](mailto:shaden@mit.edu)

# BDR-main

This repository is an official PyTorch implementation of "Balanced Destruction-Reconstruction Dynamics for Memory-replay Class Incremental Learning"

### Getting Started

This framework is implemented under Pytorch. It's better to implement an independent virtual environment on machine like this:
```
conda create --name pybdr python=3.9
conda activate pybdr
conda install pytorch=1.2.1
conda install torchvision -c pytorch
```

### Download the Datasets
#### CIFAR-100
It will be downloaded automatically by `torchvision` when running the experiments.

#### ImageNet-Subset
You may download the dataset using the following links like [this](https://github.com/yaoyao-liu/class-incremental-learning) or directly from:
- [Download from Google Drive](https://drive.google.com/file/d/1n5Xg7Iye_wkzVKc0MTBao5adhYSUlMCL/view?usp=sharing)

#### ImageNet
See the terms of ImageNet [here](https://image-net.org/download.php).


## Running Experiments
Before training, you need to set the dataset path `_BASE_DATA_PATH`  in `./dataset/dataset_config.py`.

Here is an example of how to use the code of BDR for training.
```
CUDA_VISIBLE_DEVICES=0 python main_incremental.py --exp-name nc_first_50_ntask_6 \
	--datasets cifar100_icarl --num-tasks 6 --nc-first-task 50 --network resnet18_cifar --seed 1993 \
	--nepochs 160 --batch-size 128 --lr 0.1 --momentum 0.9 --weight-decay 5e-4 --decay-mile-stone 80 120 \
	--clipping -1 --results-path results --save-models \
	--approach lucir_cwd_BDR --lamb 5.0 --num-exemplars-per-class 20 --exemplar-selection herding \
	--aux-coef 0.5 --reject-threshold 1 --dist 0.5 \
	--cwd --BDR --m1 0.8 --m2 0.8\
```

## Citation

If you find this repository useful to your research, please consider citing:
~~~
@article{zhou2021uncertainty,
  title={Uncertainty-aware Incremental Learning for Multi-organ Segmentation},
  author={Zhou, Yuhang and Zhang, Xiaoman and Feng, Shixiang and Zhang, Ya and others},
  journal={arXiv preprint arXiv:2103.05227},
  year={2021}
}
~~~

## Acknowledgement

This repository is built based on [FACIL](https://github.com/mmasana/FACIL) and [CwD](https://github.com/Yujun-Shi/CwD). We thank the authors for releasing their codes.


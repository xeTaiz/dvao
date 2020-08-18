# Deep Volumetric Ambient Occlusion
This is the official PyTorch implementation of [DVAO](https://xetaiz.github.io/dvao).
If you find our work useful, please cite our paper:
```bibtex
@article{engel2020dvao,
  title={Deep Volumetric Ambient Occlusion},
  author={Engel, Dominik and Ropinski, Timo},
  journal={IEEE transactions on visualization and computer graphics},
  volume={TODO},
  number={TODO},
  pages={TODO},
  year={2020},
  publisher={IEEE}
}
```

## Setup
### Docker
While you can setup the environment locally, we suggest using the Docker container (`Dockerfile`).

### Local
You will need to install PyTorch >= 1.4 and NVIDIA's [Apex](https://github.com/nvidia/apex) for mixed precision. Since apex installation is often quite tricky locally, we suggest using the Docker container.

The commands below should setup your conda environment
```
conda create -n dvao python=3.6
conda activate dvao
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
git clone https://github.com/NVIDIA/apex.git
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
conda install -c conda-forge gdcm pydicom
pip install pytorch_lightning dicom_numpy scikit-image comet_ml pillow pytorch-msssim
pip install git+https://github.com/aliutkus/torchinterp1d/tarball/master#egg=torchinterp1d
pip install git+https://github.com/aliutkus/torchsearchsorted/tarball/master#egg=torchsearchsorted
pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer@master#egg=ranger
```
Note that Python<=3.6 is required for `gdcm`. If you have prepared the dataset, you don't need gdcm anymore. Also using PyTorch >= 1.6 includes automatic mixed precision within PyTorch, so Apex is no longer needed.
It works without Apex on PyTorch=1.6 and PyTorch-Lightning=0.8.5.

## Usage
### Inference using pretrained model
```
python infer.py /path/to/checkpoint.ckpt /path/to/item
```
Check the code to see how to implement it in your script.

### Train the model yourself
```
python train_ao.py /path/to/ds
```
The default parameters should lead to the model we identify as best in our paper. See `python train_ao.py --help` for possible parameters you might want to change.

To reproduce the run from the paper use `--seed 1871677067`.

### Training Data (CQ500)
The training data for DVAO uses the [CQ500 dataset](http://headctstudy.qure.ai/dataset) for CT volume data. The accompanying transfer function is randomly generated (see `tf_utils.py`) and the ground truth AO volume is computed using `raycast_cuda.py`.

After downloading the dataset, you can use the `cuda_runner.py` script to generate training data by first generating a random transfer function (TF), then computing the ground truth ambient occlusion volume using that TF and lastly saving it in the format that `QureDataset` can use (see `train_ao.py`).

### Relevant Scripts and Files
```
train_ao.py            Training script for DVAO (has argparse parameters)
raycast_cuda.py        Computes ground truth AO
raycast_volume.cu      CUDA code for ground truth AO

Dockerfile             Dockerfile to setup our environment in a container
peak_finder.py         Used to extract histogram peaks for random TF generation
tf_utils.py            Functions used for our TF generation
torch_interp.py        PyTorch CUDA-accelerated 1d interpolation from https://github.com/aliutkus/torchinterp1d
ranger.py              Ranger optimizer (RAdam + Lookahead) from https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer
ssim3d_torch.py        3D SSIM, adopted from https://github.com/VainF/pytorch-msssim
```

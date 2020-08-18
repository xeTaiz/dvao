# Base image must at least have pytorch and CUDA installed.
ARG BASE_IMAGE=pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN echo "Installing Apex on top of ${BASE_IMAGE}"
RUN conda create -n dvao --clone base
RUN echo "source activate dvao" > ~/.bashrc
ENV PATH /opt/conda/envs/dvao/bin:$PATH
# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN pip uninstall -y apex || :
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN apt-get update
RUN apt-get install -y libturbojpeg libgdcm2.6
RUN conda run -n dvao pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
RUN conda run -n dvao conda install -c conda-forge gdcm pydicom
RUN conda run -n dvao pip install pytorch_lightning torchsummary dicom_numpy scikit-image pillow pytorch-msssim
RUN conda run -n dvao pip install git+https://github.com/aliutkus/torchinterp1d/tarball/master#egg=torchinterp1d
RUN conda run -n dvao pip install git+https://github.com/aliutkus/torchsearchsorted/tarball/master#egg=torchsearchsorted
RUN conda run -n dvao pip install git+https://github.com/lessw2020/Ranger-Deep-Learning-Optimizer@master#egg=ranger

WORKDIR /workspace

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV CONDA_DEFAULT_ENV dvao
SHELL ["/bin/bash", "-c"]
COPY ./*.py /workspace/

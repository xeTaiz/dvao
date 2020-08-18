# Base image must at least have pytorch and CUDA installed.
ARG BASE_IMAGE=pytorch/pytorch:1.3-cuda10.1-cudnn7-devel
FROM $BASE_IMAGE
ARG BASE_IMAGE
RUN echo "Installing Apex on top of ${BASE_IMAGE}"
RUN conda create -n deep-vol --clone base
RUN echo "source activate deep-vol" > ~/.bashrc
ENV PATH /opt/conda/envs/deep-vol/bin:$PATH
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
RUN conda run -n deep-vol pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
RUN conda run -n deep-vol conda install -c conda-forge gdcm pydicom
RUN conda run -n deep-vol pip install torchsummary test_tube dicom_numpy scikit-image comet_ml wandb pillow
COPY ./pytorch-lightning /workspace/pytorch-lightning
RUN conda run -n deep-vol pip install --upgrade -e file:///workspace/pytorch-lightning
WORKDIR /workspace

COPY ./raycast_volume.so /workspace/raycast_volume.so

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

ENV CONDA_DEFAULT_ENV deep-vol
SHELL ["/bin/bash", "-c"]
COPY ./*.py /workspace/
# ENTRYPOINT python /workspace/train_ao_nn.py --use_16bit --cos_anneal --min_epochs 5 --max_epochs 20 --train_path /workspace/train --valid_path /workspace/valid
# RUN /bin/bash

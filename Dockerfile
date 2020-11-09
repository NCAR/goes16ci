FROM nvidia/cuda:10.1-cudnn7-runtime-centos8

LABEL maintainer="David John Gagne <dgagne@ucar.edu>"
ARG NB_USER="goeser"

USER root

RUN useradd -ms /bin/bash $NB_USER

RUN yum install -yq wget \
                    sudo \
                    vim \
                    git \
                    bzip2 \
                    ca-certificates 


ENV CONDA_DIR=/home/$NB_USER/conda \
    SHELL=/bin/bash \
    NB_USER=$NB_USER \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8

ENV PATH=$CONDA_DIR/bin:$PATH \
    HOME=/home/$NB_USER

USER $NB_USER
WORKDIR $HOME

ENV MINICONDA_VERSION=4.8.3

RUN cd $HOME && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    /bin/bash Miniconda3-py37_4.8.3-Linux-x86_64.sh -f -b -p $CONDA_DIR && \
    rm Miniconda3-py37_${MINICONDA_VERSION}-Linux-x86_64.sh && \
    conda install --quiet --yes conda && \
    conda install --quiet --yes pip && \
    conda update --all --quiet --yes && \
    conda clean --all -f -y && \
    rm -rf /home/$NB_USER/.cache/yarn

RUN cd $HOME && \
    git clone https://github.com/djgagne/goes16ci.git && \
    cd goes16ci && \
    conda env create -f environment.yml && \
    source activate goes && \
    pip install . && \
    python download_data.py

CMD /bin/bash

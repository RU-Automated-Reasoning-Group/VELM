FROM --platform=linux/x86_64 ubuntu:22.04

# install linux packages and python
RUN apt update
RUN apt -y install m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex gnuplot-x11 libglpk-dev gcc g++ libopenmpi-dev python3.10-dev python3-pip libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libcairo2-dev libjpeg-dev libgif-dev pkg-config libgirepository1.0-dev libdbus-1-3 libdbus-1-dev
RUN echo "alias python3='python3.10'" >> ~/.bashrc

# copy our code
COPY VELM /VELM

# # install python packages
WORKDIR /VELM/
RUN python3 -m pip install -r requirements.txt

# WORKDIR /VELM/dependency
# RUN python3 -m pip install pyoperon-0.3.6-cp310-cp310-linux_x86_64.whl

WORKDIR /VELM/dependency/DSO/dso
RUN python3 -m pip install -e .


# # build POLAR and flowstar
WORKDIR /VELM/POLAR_Tool/flowstar/flowstar-toolbox
RUN make clean && make -j 4

WORKDIR /VELM/POLAR_Tool/POLAR
RUN make clean && make -j 4

WORKDIR /VELM
RUN bash compile.sh


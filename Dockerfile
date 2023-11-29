FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime as cpp-builder

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# https://stackoverflow.com/questions/55313610/importerror-libgl-so-1-cannot-open-shared-object-file-no-such-file-or-directo

RUN pip install --upgrade pip setuptools wheel
RUN pip install opencv-python==4.5.5.64
RUN pip install pynrrd tqdm future pydicom scipy matplotlib scikit-image pandas termcolor tensorboard

RUN /opt/conda/bin/conda install --debug --override-channels -c main -c conda-forge cython numpy -y

# RUN mkdir /lib
WORKDIR /lib/
RUN git clone https://github.com/mwwahi/SANet.git SANet

RUN chmod 777 -R /lib/SANet

WORKDIR /lib/SANet
WORKDIR /lib/SANet/build/box
RUN python setup.py install

WORKDIR /lib/SANet

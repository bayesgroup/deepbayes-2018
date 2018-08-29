FROM andrewosh/binder-base
MAINTAINER Ano Nymous <ano@nymo.us>
USER root

RUN echo "deb http://archive.ubuntu.com/ubuntu trusty-backports main restricted universe multiverse" >> /etc/apt/sources.list
RUN apt-get -qq update

RUN apt-get install -y gcc-4.9 g++-4.9 libstdc++6 wget unzip
RUN apt-get install -y libsdl2-dev libboost-all-dev graphviz
RUN apt-get install -y cmake zlib1g-dev libjpeg-dev 
RUN apt-get install -y xvfb libav-tools xorg-dev python-opengl python3-opengl
RUN apt-get -y install swig3.0
RUN ln -s /usr/bin/swig3.0 /usr/bin/swig


USER main
RUN pip install --upgrade pip==9.0.3
RUN pip install --upgrade --ignore-installed setuptools  #fix https://github.com/tensorflow/tensorflow/issues/622
RUN pip install --upgrade sklearn tqdm nltk editdistance joblib graphviz

# install all gym stuff except mujoco - it fails at "import importlib.util" (no module named util)
RUN pip install --upgrade gym
RUN pip install --upgrade gym[atari]
RUN pip install --upgrade gym[box2d]
RUN pip install --upgrade torch torchvision 

RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade pip==9.0.3

RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade --ignore-installed setuptools
# python3: fix `GLIBCXX_3.4.20' not found - conda's libgcc blocked system's gcc-4.9 and libstdc++6
RUN bash -c "conda update -y conda && source activate python3 && conda uninstall -y libgcc && source deactivate"
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade matplotlib numpy scipy pandas graphviz

RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade sklearn tqdm nltk editdistance joblib
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade --ignore-installed setuptools  #fix https://github.com/tensorflow/tensorflow/issues/622

# install all gym stuff except mujoco - it fails at "mjmodel.h: no such file or directory"
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade gym
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade gym[atari]
RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade gym[box2d]

RUN /home/main/anaconda/envs/python3/bin/pip install --upgrade torch torchvision

FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
# Ubuntu 22.04 base image, personally prefer 20.04 but whatever
USER root
ENV HOME_DIR /home/rl_course
WORKDIR $HOME_DIR
RUN mkdir -p $HOME_DIR
ENV MOUNT_DIR /mnt
RUN mkdir -p $MOUNT_DIR

# System level dependencies
RUN apt-get update 
RUN apt-get install -y swig cmake ffmpeg freeglut3-dev xvfb git wget unzip tldr-py vim
RUN apt-get install -y python3-dev python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python
# RUN ln -s /usr/bin/pip3 /usr/bin/pip
RUN pip install --upgrade pip

# RL course dependencies
## using pytorch docker
# RUN pip install torch==2.3.1 
RUN pip install "autorom[accept-rom-license]"
RUN pip install rl-zoo3==2.3.0
# to install the latest commit of a repo
ADD "https://api.github.com/repos/yusenz/gym-maze/commits?per_page=1" latest_commit
RUN git clone https://github.com/yusenz/gym-maze.git $HOME_DIR/gym-maze
RUN cd $HOME_DIR/gym-maze && python setup.py install
# RUN git clone https://github.com/DLR-RM/rl-baselines3-zoo $HOME_DIR/rl-baselines3-zoo
# RUN cd $HOME_DIR/rl-baselines3-zoo && pip install -r requirements.txt
# RUN pip install -e .[plots,tests]

# Jupyter
RUN pip install jupyter
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]

# some sanity python dependencies, put last so that only installed if not exist
RUN pip install numpy scipy matplotlib scikit-learn pandas pygame
RUN cd $HOME_DIR
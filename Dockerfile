FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 AS nvidia

ENV TZ="Asia/Beijing"

ARG DEBIAN_FRONTEND="noninteractive"

RUN mkdir -p /workspace
WORKDIR /workspace

RUN apt update \
    && apt-get install dialog apt-utils -y \
    && apt install tzdata -y \
    && apt-get install -y build-essential checkinstall \
    && apt-get install libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev -y \
    && apt install libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libffi-dev gnupg pciutils wget swig curl git vim make locales -y \
    && apt clean \
    && rm -rf /var/cache/apt/* \
    && sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen \
    && locale-gen

RUN cd /workspace/ \
    && wget https://www.python.org/ftp/python/3.8.13/Python-3.8.13.tgz \
    && tar xzf Python-3.8.13.tgz \
    && cd Python-3.8.13 \
    && ./configure --enable-optimizations \
    && make altinstall \
    && ln -s /usr/local/bin/python3.8 /usr/bin/python3 \
    && ln -s /usr/local/bin/python3.8 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip3 \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip \
    && cd /workspace/ \
    && rm -rf ./Python-3.8.13*

RUN cd /workspace/ \
    && git clone -b main https://github.com/opendilab/DI-engine.git \
    && cd /workspace/DI-engine \
    && pip install --no-cache-dir .[common_env,test] \
    && pip install --no-cache-dir Autorom \
    && AutoROM -y

RUN cd /workspace/ \
    && git clone -b main https://github.com/opendilab/DI-hpc.git \
    && cd /workspace/DI-hpc \
    && python setup.py install

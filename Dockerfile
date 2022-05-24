FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 AS di-hpc-develop

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
    && ln -s /usr/local/bin/python3.8 /usr/bin/python3.8.13 \
    && ln -s /usr/local/bin/python3.8 /usr/bin/python3.8 \
    && ln -s /usr/local/bin/python3.8 /usr/bin/python3 \
    && ln -s /usr/local/bin/python3.8 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip3 \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip \
    && cd /workspace/ \
    && rm -rf ./Python-3.8.13*

RUN cd /workspace/ \
    && wget https://download.pytorch.org/whl/cu113/torch-1.11.0%2Bcu113-cp38-cp38-linux_x86_64.whl -O torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl \
    && pip install --no-cache-dir /workspace/torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl \
    && rm /workspace/torch-1.10.0+cu113-cp38-cp38-linux_x86_64.whl

ADD setup.py /workspace/setup.py
ADD hpc_rll /workspace/hpc_rll
ADD include /workspace/include
ADD src /workspace/src
ADD tests /workspace/tests

RUN python /workspace/setup.py install

FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04 AS di-hpc-runtime

COPY --from=di-hpc-develop /usr/local/bin/ /usr/local/bin/
COPY --from=di-hpc-develop /usr/local/include/ /usr/local/include/
COPY --from=di-hpc-develop /usr/local/lib/ /usr/local/lib/
COPY --from=di-hpc-develop /usr/local/share/ /usr/local/share/

RUN ln -s /usr/local/bin/python3.8 /usr/bin/python3 \
    && ln -s /usr/local/bin/python3.8 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip3 \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip

FROM ubuntu:20.04 AS di-hpc-nightly

COPY --from=di-hpc-develop /usr/local/bin/ /usr/local/bin/
COPY --from=di-hpc-develop /usr/local/include/ /usr/local/include/
COPY --from=di-hpc-develop /usr/local/lib/ /usr/local/lib/
COPY --from=di-hpc-develop /usr/local/share/ /usr/local/share/

RUN ln -s /usr/local/bin/python3.8 /usr/bin/python3 \
    && ln -s /usr/local/bin/python3.8 /usr/bin/python \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip3 \
    && ln -s /usr/local/bin/pip3.8 /usr/bin/pip

## DI-HPC: Decision Intelligence - High Performance Computation
**DI-HPC** is an acceleration operator component for general algorithm modules in reinforcement learning algorithms, such as GAE, n-step TD and LSTM, etc. The operators support forward and backward propagation, and can be used in training, data collection, and test modules.

## Requirements
#### Setting 1
* CUDA 9.2
* PyTorch 1.5 (recommend)
* python 3.6 or python 3.7 or python3.8
* Linux Platform

#### Setting 2
* CUDA 9.0
* gcc 5.4.0
* PyTorch 1.1.0
* python 3.6 or python 3.7
* Linux Platform

*Note: We recommend that DI-HPC and DI-Engine share the same environment, and it should be fine with PyTorch from 1.1.0 to 1.10.0.*

## Quick Start
#### Install from whl
The easiest way to get DI-HPC is to use pip, and you can get `.whl` from
* [di_hpc_rll-0.0.2-cp36-cp36m-linux_x86_64.whl](http://opendilab.org/download/DI-hpc/di_hpc_rll-0.0.2-cp36-cp36m-linux_x86_64.whl)
* [di_hpc_rll-0.0.2-cp37-cp37m-linux_x86_64.whl](http://opendilab.org/download/DI-hpc/di_hpc_rll-0.0.2-cp37-cp37m-linux_x86_64.whl)
* [di_hpc_rll-0.0.2-cp38-cp38-linux_x86_64.whl](http://opendilab.org/download/DI-hpc/di_hpc_rll-0.0.2-cp38-cp38-linux_x86_64.whl)

and then call
```
$ pip install <YOUR_WHL>
```

#### Install from source code
Alternatively you can install latest DI-HPC from git master branch:
```
$ python3 setup.py install
```

#### Run on Linux
You will get benchmark result by following commands:
```
$ python3 tests/test_gae.py
```
## TODO
- [] Trition Kernel for Reinfocement Learning

## Feedback and Contribution

- [File an issue](https://github.com/opendilab/DI-hpc/issues/new/choose) on Github
- Discuss on DI-engine's (also for DI-hpc) [discord server](https://discord.gg/dkZS2JF56X)
- Contact our email (opendilab@pjlab.org.cn)

We appreciate all the feedbacks and contributions to improve DI-engine, both algorithms and system designs. And `CONTRIBUTING.md` offers some necessary information. 


## License
DI-hpc released under the Apache 2.0 license.

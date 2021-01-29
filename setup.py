import setuptools
from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NAME = 'hpc_rll'
VERSION = '0.0.1'
AUTHOR = 'wangyingrui'
EMAIL = 'wangyingrui@sensetime.com'
DESC = 'GPU-Accelerated library for reinforcement learning, from HPC SenseTime'
PLATFORMS = 'linux-x86_64'
PACKAGES = ['hpc_rll', 'hpc_rll.origin', 'hpc_rll.rl_utils', 'hpc_rll.torch_utils', 'hpc_rll.torch_utils.network']

setup(
    name = NAME,
    version = VERSION,
    author = AUTHOR,
    author_email = EMAIL,
    description = DESC,
    platforms = PLATFORMS,
    packages = PACKAGES,
 
    ext_modules=[
        CUDAExtension('hpc_rl_utils', sources=[
            'src/rl_utils/entry.cu',
            'src/rl_utils/dist_nstep_td.cu',
            'src/rl_utils/gae.cu',
            'src/rl_utils/ppo.cu',
            'src/rl_utils/q_nstep_td.cu',
            'src/rl_utils/q_nstep_td_rescale.cu',
            'src/rl_utils/td_lambda.cu',
            'src/rl_utils/upgo.cu',
            'src/rl_utils/vtrace.cu',
            ], include_dirs=['include']),
        CUDAExtension('hpc_torch_utils_network', sources=[
            'src/torch_utils/network/entry.cu',
            'src/torch_utils/network/lstm.cu',
            'src/torch_utils/network/scatter_connection.cu'
            ], include_dirs=['include']),
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

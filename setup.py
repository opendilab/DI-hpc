import setuptools
from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NAME = 'di_hpc_rll'
VERSION = '0.0.2'
DESC = 'GPU-Accelerated library for reinforcement learning'
PLATFORMS = 'linux-x86_64'
PACKAGES = ['hpc_rll', 'hpc_rll.origin', 'hpc_rll.rl_utils', 'hpc_rll.torch_utils', 'hpc_rll.torch_utils.network']

setup(
    name = NAME,
    version = VERSION,
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
            'src/rl_utils/iqn_nstep_td_error.cu',
            'src/rl_utils/qrdqn_nstep_td_error.cu',
            ], include_dirs=['include']),
        CUDAExtension('hpc_torch_utils_network', sources=[
            'src/torch_utils/network/entry.cu',
            'src/torch_utils/network/lstm.cu',
            'src/torch_utils/network/scatter_connection.cu'
            ], include_dirs=['include']),
        CUDAExtension('hpc_models', sources=[
            'src/models/entry.cu',
            'src/models/actor_critic.cu',
            ], include_dirs=['include']),

        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)

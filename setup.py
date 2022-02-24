import setuptools
from setuptools import setup
import os
import glob
import torch
import warnings
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

NAME = 'di_hpc_rll'
VERSION = '0.0.2'
DESC = 'GPU-Accelerated library for reinforcement learning'
PLATFORMS = 'linux-x86_64'
PACKAGES = ['hpc_rll', 'hpc_rll.origin', 'hpc_rll.rl_utils', 'hpc_rll.torch_utils', 'hpc_rll.torch_utils.network']
include_dirs = [os.path.join(os.getcwd(), 'include')]
print('include_dirs', include_dirs)

ext_modules = []
ext_modules.append(
        CUDAExtension('hpc_rl_utils', sources=[
            'src/rl_utils/entry.cpp',
            'src/rl_utils/dist_nstep_td.cu',
            'src/rl_utils/gae.cu',
            'src/rl_utils/padding.cu',
            'src/rl_utils/ppo.cu',
            'src/rl_utils/q_nstep_td.cu',
            'src/rl_utils/q_nstep_td_rescale.cu',
            'src/rl_utils/td_lambda.cu',
            'src/rl_utils/upgo.cu',
            'src/rl_utils/vtrace.cu',
            'src/rl_utils/iqn_nstep_td_error.cu',
            'src/rl_utils/qrdqn_nstep_td_error.cu',
            'src/models/actor_critic.cu',
            ], include_dirs=include_dirs)
        )
ext_modules.append(
        CUDAExtension('hpc_torch_utils_network', sources=[
            'src/torch_utils/network/entry.cpp',
            'src/torch_utils/network/lstm.cu',
            'src/torch_utils/network/scatter_connection.cu'
            ], include_dirs=include_dirs),
        )

if int("".join(list(filter(str.isdigit, torch.__version__)))) >= 120:
    ext_modules.append(
            CUDAExtension('hpc_models', sources=[
                'src/models/entry.cpp',
                'src/models/actor_critic.cu',
                ], include_dirs=include_dirs),
            )
else:
    warnings.warn("Torch version is less than 1.2. BoolTensor is not yet well implemented. Thus we skip the compiliation of hpc_models.")

setup(
    name = NAME,
    version = VERSION,
    description = DESC,
    platforms = PLATFORMS,
    packages = PACKAGES,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)

#!/bin/bash

SITE=c:/programdata/anaconda3/envs/nervex/lib/site-packages
sed -i.bak -e 's/CONSTEXPR_EXCEPT_WIN_CUDA/const/g' "${SITE}/torch/include/torch/csrc/jit/api/module.h"
sed -i.bak -e 's/return \*(this->value)/return \*((type\*)this->value)/g' "${SITE}/torch/include/pybind11/cast.h"
sed -i.bak '/static constexpr Symbol Kind/d' "${SITE}/torch/include/torch/csrc/jit/ir/ir.h"
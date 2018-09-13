备忘-IPython小技巧
===

Index
---
<!-- TOC -->

- [IPython](#ipython)
  - [自动重新加载模块](#自动重新加载模块)
- [Anaconda](#anaconda)
  - [虚拟环境相关](#虚拟环境相关)
- [PyCharm](#pycharm)
  - [Python Console](#python-console)
- [常用库安装](#常用库安装)
  - [PyTorch 安装](#pytorch-安装)

<!-- /TOC -->

## IPython

### 自动重新加载模块
```
%load_ext autoreload
%autoreload 2
```
- 这个有时候也不太好用
- 需要反复测试的，建议使用 Jupyter Notebook


## Anaconda

### 虚拟环境相关
- 创建虚拟环境
  ```
  conda create -n env_name anaconda python=3
  ```
- 复制虚拟环境
  ```
  conda create --name dst_name --clone src_name
  ```
- 删除虚拟环境
  ```
  conda remove --name nev_name --all
  ```


## PyCharm

### Python Console
- 默认
  ```
  import sys; print('Python %s on %s' % (sys.version, sys.platform))
  sys.path.extend([WORKING_DIR_AND_PYTHON_PATHS])
  ```
- 修改为，避免每次重新导入
  ```
  %load_ext autoreload
  %autoreload 2

  import sys; print('Python %s on %s' % (sys.version, sys.platform))
  sys.path.extend([WORKING_DIR_AND_PYTHON_PATHS])

  import numpy as np
  import tensorflow as tf
  ```

## 常用库安装

### PyTorch 安装
- Windows Python3.6 CPU
  - conda 提供的源很慢，不建议
  - 官网提供的 pip 安装方法，在安装 `torchvision` 时会报错
  - 正确安装方法（2018年8月17日15:09:40）
    ```
    pip3 install http://download.pytorch.org/whl/cpu/torch-0.4.1-cp36-cp36m-win_amd64.whl 
    pip3 install --no-deps torchvision
    ```
    > 相比官方，只添加了 `--no-deps` 参数
  - 推荐先下载在安装
    ```
    pip3 install torch-0.4.1-cp36-cp36m-win_amd64.whl
    ```

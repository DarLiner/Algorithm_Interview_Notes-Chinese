备忘-IPython小技巧
===

Index
---
<!-- TOC -->

- [IPython](#ipython)
  - [自动重新加载模块](#自动重新加载模块)
- [Anaconda](#anaconda)
  - [虚拟环境相关](#虚拟环境相关)

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
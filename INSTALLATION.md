## 🛠️ Installation

- 如何安装Nvidia CUDA和驱动（Ubuntu24.04 LTS）:
  ```bash
  REF: https://blog.csdn.net/weixin_52326559/article/details/126359130
  
  卸载系统自带驱动
  sudo apt remove nvidia*
  sudo apt purge nvidia*
  nvidia-smi

  //sudo apt install nvidia-cuda-toolkit

  降低GCC版本
  //sudo apt install gcc-9 g++-9
  //sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 100 # gcc-9替换成你要用的版本

  安装CUDA：12.1
  wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda_12.1.0_530.30.02_linux.run

  sudo sh cuda_12.1.0_530.30.02_linux.run
  不要选择显卡驱动的安装，也就是去掉第一个框框内的×，点击install，等待。

  从nvidia.com / nvidia.cn下载对应版本驱动并安装
  sudo sh NVIDIA-Linux-x86_64-550.78.run


  配置：加入CUDA环境变量配置信息
  sudo vim ~/.bashrc

  export PATH=$PATH:/usr/local/cuda-12.1/bin
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-12.1/lib64
  export LIBRARY_PATH=$LIBRARY_PATH:/usr/local/cuda-12.1/lib64

  安装cuDNN：
  wget https://developer.download.nvidia.com/compute/cudnn/9.1.1/local_installers/cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb
  sudo dpkg -i cudnn-local-repo-ubuntu2204-9.1.1_1.0-1_amd64.deb
  sudo cp /var/cudnn-local-repo-ubuntu2204-9.1.1/cudnn-*-keyring.gpg /usr/share/keyrings/
  sudo apt-get update
  sudo apt-get -y install cudnn

  To install for CUDA 11, perform the above configuration but install the CUDA 11 specific package:
  sudo apt-get -y install cudnn-cuda-11

  To install for CUDA 12, perform the above configuration but install the CUDA 12 specific package:
  sudo apt-get -y install cudnn-cuda-12
  
  检测：
  
  ```
- Clone this repository:

  ```bash
  git clone https://github.com/OpenGVLab/InternVL.git
  ```

- Create a conda virtual environment and activate it:

  ```bash
  conda create -n internvl python=3.9 -y
  conda activate internvl
  ```

- Install `PyTorch>=2.0` and `torchvision>=0.15.2` with `CUDA>=11.6`:

  For examples, to install `torch==2.0.1` with `CUDA==11.8`:

  ```bash
  # 需要注意pytory-cuda的版本和系统cuda版本是否一致
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
  conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=12.1 -c pytorch -c nvidia

  or
  
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

- Install `flash-attn==2.3.6`:

  ```bash
  MAX_JOBS=4 pip install flash-attn==2.3.6 --no-build-isolation
  pip install flash-attn==2.3.6 --no-build-isolation
  ```

  Alternatively you can compile from source:

  ```bash
  git clone https://github.com/Dao-AILab/flash-attention.git
  cd flash-attention
  git checkout v2.3.6

  cd ./csrc/
  git clone https://github.com/NVIDIA/cutlass.git
  
  python setup.py install
  ```

- Install `timm==0.9.12` and `mmcv-full==1.6.2`:

  ```bash
  pip install timm==0.9.12
  pip install -U openmim
  mim install mmcv-full==1.6.2  # (optional, for mmsegmentation)
  ```

- Install `transformers==4.37.2`:

  ```bash
  pip install transformers==4.37.2
  ```

- Install `apex` (optional):

  ```bash
  git clone https://github.com/NVIDIA/apex.git
  git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82  # https://github.com/NVIDIA/apex/issues/1735
  pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
  ```

  If you meet `ModuleNotFoundError: No module named 'fused_layer_norm_cuda'`, please note that this is because apex's CUDA extensions are not being installed successfully. You can try to uninstall apex and the code will default to the PyTorch version of RMSNorm; Or, if you want to use apex, try adding a few lines to `setup.py`, like this, and then recompiling.

  <img src=https://github.com/OpenGVLab/InternVL/assets/23737120/c04a989c-8024-49fa-b62c-2da623e63729 width=50%>

- Install other requirements:

  ```bash
  pip install opencv-python termcolor yacs pyyaml scipy
  pip install deepspeed==0.13.5
  pip install pycocoevalcap tqdm
  ```

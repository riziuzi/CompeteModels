# CompeteModel : Models used in the project [MindScape India (Compete)](https://github.com/riziuzi/compete)
This repository contains all the models currently under development and does not handle the hosting of these models. (refer to the [Compete](https://github.com/riziuzi/compete) repository)
## Setup

preferred setup [video](https://www.youtube.com/watch?v=VE5OiQSfPLg) (Note: for running tensorflow-addons used in the models, tensorflow<=2.14.0 is required)

```
Tenserflow GPU(2.14) installation on Windows 11 through WSL2 ( VS Code installation and Jupiter LAB installation included)
1.GPU Drivers update
2.Create Windows Subsystem for Linux (WSL)
 2.1 wsl --install
 2.2 Setup user and login
 2.3 Update the linux system
  2.3.1 sudo apt-get update
  2.3.2 sudo apt-get upgrade
  2.3.3 sudo reboot
3.Install Anaconda(For managing environments)
 3.1  https://www.anaconda.com/download Linux Python 3.11 64-Bit (x86) Installer (1015.6 MB)
 3.2 Copy file to the linux system
 3.3 Install Anaconda 
  3.3.1 bash Anaconda-latest-Linux-x86_64.sh
  3.3.2 conda config --set auto_activate_base False 
 3.4 Create environments
  3.4.1 conda create -n myenv python=3.11
  3.4.2 conda activate myenv
4. Install CUDA
 4.1  https://developer.nvidia.com/cuda-too... (11.8)
 4.2 wget https://developer.download.nvidia.com...
 4.3 sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
 4.4 wget https://developer.download.nvidia.com...
 4.5 sudo dpkg -i cuda-repo-wsl-ubuntu-11-8-local_11.8.0-1_amd64.deb
 4.6 sudo cp /var/cuda-repo-wsl-ubuntu-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
 4.7 sudo apt-get update
 4.8 sudo apt-get -y install cuda
5. Install cuDDN
 5.1  https://developer.nvidia.com/rdp/cudn... (11.x, Local Installer for Ubuntu22.04 x86_64 (Deb) )
 5.2 Copy file to the linux system
 5.3 sudo dpkg -i cudnn-local-repo-$distro-8.x.x.x_1.0-1_amd64.deb
 5.4 sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
 5.5 sudo apt-get update
 5.6 sudo apt-get install libcudnn8=8.x.x.x-1+cudaX.Y
 5.7 sudo apt-get install libcudnn8-dev=8.x.x.x-1+cudaX.Y
 5.8 sudo apt-get install libcudnn8-samples=8.x.x.x-1+cudaX.Y
 5.9 sudo reboot 
6. pip install --upgrade pip
7. python3 -m pip install tensorflow[and-cuda]
8. pip install --ignore-installed --upgrade tensorflow==2.14
9. python3 -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
10. conda install -c conda-forge jupyterlab
11. code .
12. VS Code WSL2 and Python plugin
```


## Model Used (incomplete description)
Encoder -> LSTM
Decoder -> LSTM

## References (incomplete description)


Note: all the incomplete description will be completed in future, if you want to contribute, please do the PR
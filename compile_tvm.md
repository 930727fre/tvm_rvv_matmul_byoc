# TVM compilation

# Compile yourself

WE ARE USING TVM v0.20.0 RELEASE!!!!!!!

[https://github.com/apache/tvm/blob/v0.20.0/docs/install/from_source.rst](https://github.com/apache/tvm/blob/v0.20.0/docs/install/from_source.rst)

## Must read

Don’t use Conda!!

Dont’t use WSL!

## CMake

```bash
#!/bin/bash

# Update package list
sudo apt update

# Install necessary dependencies
sudo apt install -y wget software-properties-common lsb-release gnupg

# Download and add Kitware's GPG key
wget -O - https://apt.kitware.com/keys/kitware-archive-latest.asc | sudo gpg --dearmor -o /usr/share/keyrings/kitware-archive-keyring.gpg

# Determine Ubuntu version
UBUNTU_CODENAME=$(lsb_release -sc)

# Add Kitware repository manually
echo "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] https://apt.kitware.com/ubuntu/ $UBUNTU_CODENAME main" | sudo tee /etc/apt/sources.list.d/kitware.list > /dev/null

# Update package list again
sudo apt update

# Install CMake
sudo apt install -y cmake

```

## LLVM 15

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 15
llvm-config-15 --version # is it normal that there's no llvm-config, but a lot of llvm-* commands?

# if llvm-config is missing, do 
llvm-config --version
sudo ln -s $(which llvm-config-15) /usr/local/bin/llvm-config

```

or 

```python
sudo apt update
sudo apt install clang-15 llvm-15 llvm-15-dev llvm-15-tools
```

# Etc

```php
sudo apt install git

#tvm requires python>=3.8, 
# in my case, i already have python3->3.10, but i would like to have `python` as well
sudo apt update
sudo apt install python-is-python3

sudo apt install g++ # install if needed
sudo apt install zlib1g-dev

sudo apt update
sudo apt install python3-pip python3-setuptools python3-wheel
python3 -m pip install --user --upgrade setuptools wheel cython numpy psutil typing_extensions
```

# Tvm

WE HAVE SPECIFY `0.20.0`!!!!!

```php
git clone  -b v0.20.0 --recursive https://github.com/apache/tvm tvm
cd tvm
rm -rf build && mkdir build && cd build
# Specify the build configuration via CMake options
cp ../cmake/config.cmake .

# controls default compilation flags (Candidates: Release, Debug, RelWithDebInfo)
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake

# LLVM is a must dependency for compiler end
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake

# GPU SDKs, turn on if needed
echo "set(USE_CUDA   OFF)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake

# cuBLAS, cuDNN, cutlass support, turn on if needed
echo "set(USE_CUBLAS OFF)" >> config.cmake
echo "set(USE_CUDNN  OFF)" >> config.cmake
echo "set(USE_CUTLASS OFF)" >> config.cmake
cmake .. && cmake --build . --parallel $(nproc)

```

## Note

Don’t forget to store `export` to bashrc

```bash
echo 'export TVM_HOME=/home/fre930727/tvm' >> ~/.bashrc
echo 'export PYTHONPATH=$TVM_HOME/python:$PYTHONPATH' >> ~/.bashrc
source ~/.bashrc

```

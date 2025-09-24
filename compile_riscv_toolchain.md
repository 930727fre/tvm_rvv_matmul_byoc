--prefix 位置為編譯好的工具要放哪裡

```php
cd
git clone https://github.com/riscv/riscv-gnu-toolchain riscv
sudo apt-get install autoconf automake autotools-dev curl python3 python3-pip python3-tomli libmpc-dev libmpfr-dev libgmp-dev gawk build-essential bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev ninja-build git cmake libglib2.0-dev libslirp-dev -y

cd
cd riscv
mkdir build_riscv && cd build_risc
../configure --prefix=/opt/riscv --enable-multilib
# don't use all the cores, it will crash
sudo make -j5 # 安裝 Newlib 版本，適用於 target 是裸機或嵌入式裝置
sudo make linux -j5 # 安裝 linux  版本，適用於 target 有 linux 類型 OS (若有產生.so 檔的需求需要安裝這版本)
echo 'export PATH=/opt/riscv/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

```
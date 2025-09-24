# Envrionment setup

# Setup

## Setup x86 vm for compiling 3 models (encoder.onnx, decoder.onnx, decoder_with_past.onnx)

The reason we use cross-compile is because `onnx` package is not available on risc-v system at Sep, 2025.

The following is our vm hardware spec:

```php
Ubuntu 22.04.5 LTS
6 vCore
20gb ram
100gb disk
```

Steps to setup:

1. Enable VirtualBox copy/paste (optional)
2. Disable sleep 5 min in settings (optional)
3. Install dependencies and TVM
    
    [compile_tvm.md](compile_tvm.md)
    
4. Install risc-v toolchain
    
    [compile_riscv_toolchain.md](compile_riscv_toolchain.md)
    
5. Setup bananapi runtime and codegen
    
    [set_up_bananapi_rutime_and_codegen.md](set_up_bananapi_rutime_and_codegen.md)
    
6. Download whisper-tiny models from hugging face
    
    https://huggingface.co/onnx-community/whisper-tiny/tree/main
    
7. Compile 3 models and rsync them.
    
    ```bash
    cd ~/whisper-tiny/onnx
    python3 compile_encoder.py
    python3 compile_decoder.py
    python3 compile_decoder_with_past.py
    rsync -avz -e ssh ./*.so fre930727@your_ip:~/whisper-tiny/onnx/
    ```
    

# Setup Banana pi f3 for inference

Note: Bianbu 2.2 is derived from Ubuntu 24.04

```bash
        #####           fre930727@spacemit-k1-x-deb1-board
       #######          ----------------------------------
       ##O#O##          OS: Bianbu 2.2 riscv64
       #######          Host: spacemit k1-x deb1 board
     ###########        Kernel: 6.6.63
    #############       Uptime: 3 days, 5 hours, 56 mins
   ###############      Packages: 2265 (dpkg)
   ################     Shell: zsh 5.9
  #################     Terminal: /dev/pts/1
#####################   CPU: Spacemit X60 (8) @ 1.600GHz
#####################   Memory: 307MiB / 3807MiB
  #################

```

1. Install dependencies and TVM
    
    [compile_tvm.md](compile_tvm.md)
    
2. Setup bananapi runtime and codegen
    
    [set_up_bananapi_rutime_and_codegen.md](set_up_bananapi_rutime_and_codegen.md)
    
3. Compile libmatmul.cpp
    
    For native risc-v compiler
    
    ```python
    cd /home/fre930727/tvm/src/runtime/contrib/bananapi
    
    g++ -std=c++11 -shared -fPIC -O3\\
        -march=rv64gcv -mabi=lp64d \\
        -I ~/tvm/3rdparty/dlpack/include \\
        -o libmatmul.so libmatmul.cpp
    
    ```
    
    For x86 compiler
    
    ```php
    cd /home/fre930727/tvm/src/runtime/contrib/bananapi
    
    g++ -std=c++11 -shared -fPIC -O3\\
        -I ~/tvm/3rdparty/dlpack/include \\
        -o libmatmul.so libmatmul.cpp
    
    ```
    Note: you can also try libmatmul_golden.cpp. This is a textbook-level implementation of matrix multiplication from linear algebra. Just for testing out the difference with our rvv+algorithmic implementation. The compilation usage is same as the above libmatmul.cpp’s g++ command.
4. Download whisper-tiny models from hugging face, this step is necessary, because tokenizer and vocab.json is required
    
    https://huggingface.co/onnx-community/whisper-tiny/tree/main
    
    ```bash
    cd
    git clone ... (lfs or something)
    
    ```
    
5. On x86, cross-compile 3 models and rsync/scp to the board.
6. Run inference. 
    
    ```bash
    cd
    cd whisper-tiny
    mkdir profile_data
    cd ..
    python3 inference.py # or python3 inference_profile.py
    
    ```
    

# uname -a

## x86 vm:

```php
Linux tvm 6.8.0-65-generic #68~22.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Jul 15 18:06:34 UTC 2 x86_64 x86_64 x86_64 GNU/Linux

```

## Banana pi f3

```php
Linux spacemit-k1-x-deb1-board 6.6.63 #2.2.6.2 SMP PREEMPT Fri Jul 25 11:55:54 UTC 2025 riscv64 riscv64 riscv64 GNU/Linux

```

# pip list

## x86 vm:

```php
Package                  Version
------------------------ ----------------
accelerate               1.6.0
apturl                   0.5.2
bcrypt                   3.2.0
blinker                  1.4
Brlapi                   0.8.3
certifi                  2020.6.20
chardet                  4.0.0
click                    8.0.3
colorama                 0.4.4
coloredlogs              15.0.1
command-not-found        0.3
cryptography             3.4.8
cupshelpers              1.0
Cython                   3.0.12
dbus-python              1.2.18
defer                    1.0.6
distro                   1.7.0
distro-info              1.1+ubuntu0.2
duplicity                0.8.21
exceptiongroup           1.3.0
fasteners                0.14.1
filelock                 3.17.0
flatbuffers              25.2.10
fsspec                   2025.3.0
future                   0.18.2
httplib2                 0.20.2
huggingface-hub          0.29.2
humanfriendly            10.0
idna                     3.3
importlib-metadata       4.6.4
iniconfig                2.1.0
jeepney                  0.7.1
Jinja2                   3.1.6
keyring                  23.5.0
language-selector        0.1
launchpadlib             1.10.16
lazr.restfulclient       0.14.4
lazr.uri                 1.0.6
lockfile                 0.12.2
louis                    3.20.0
macaroonbakery           1.3.1
Mako                     1.1.3
MarkupSafe               2.0.1
monotonic                1.6
more-itertools           8.10.0
mpmath                   1.3.0
netifaces                0.11.0
networkx                 3.4.2
numpy                    2.2.3
nvidia-cublas-cu12       12.4.5.8
nvidia-cuda-cupti-cu12   12.4.127
nvidia-cuda-nvrtc-cu12   12.4.127
nvidia-cuda-runtime-cu12 12.4.127
nvidia-cudnn-cu12        9.1.0.70
nvidia-cufft-cu12        11.2.1.3
nvidia-curand-cu12       10.3.5.147
nvidia-cusolver-cu12     11.6.1.9
nvidia-cusparse-cu12     12.3.1.170
nvidia-cusparselt-cu12   0.6.2
nvidia-nccl-cu12         2.21.5
nvidia-nvjitlink-cu12    12.4.127
nvidia-nvtx-cu12         12.4.127
oauthlib                 3.2.0
olefile                  0.46
onnx                     1.17.0
onnxruntime              1.21.0
onnxslim                 0.1.48
optimum                  1.25.0.dev0
packaging                24.2
paramiko                 2.9.3
pexpect                  4.8.0
Pillow                   9.0.1
pip                      22.0.2
pluggy                   1.6.0
protobuf                 6.30.0
psutil                   7.0.0
ptyprocess               0.7.0
pycairo                  1.20.1
pycups                   2.0.1
Pygments                 2.11.2
PyGObject                3.42.1
PyJWT                    2.3.0
pymacaroons              0.13.0
PyNaCl                   1.5.0
pyparsing                2.4.7
pyRFC3339                1.1
pytest                   8.4.1
python-apt               2.4.0+ubuntu4
python-dateutil          2.8.1
python-debian            0.1.43+ubuntu1.1
pytz                     2022.1
pyxdg                    0.27
PyYAML                   5.4.1
regex                    2024.11.6
reportlab                3.6.8
requests                 2.25.1
safetensors              0.5.3
SecretStorage            3.3.1
sentencepiece            0.2.0
setuptools               59.6.0
six                      1.16.0
sympy                    1.13.1
systemd-python           234
tokenizers               0.21.0
tomli                    2.2.1
torch                    2.6.0
torchaudio               2.6.0
tqdm                     4.67.1
transformers             4.49.0
triton                   3.2.0
typing_extensions        4.12.2
ubuntu-drivers-common    0.0.0
ubuntu-pro-client        8001
ufw                      0.36.1
unattended-upgrades      0.1
urllib3                  1.26.5
usb-creator              0.3.7
wadllib                  1.3.6
wheel                    0.37.1
xdg                      5
xkit                     0.0.0
zipp                     1.0.0

```

## Banana pi:

```php
Package               Version
--------------------- -----------------
autocommand           2.2.2
bcrypt                3.2.2
beautifulsoup4        4.12.3
blinker               1.7.0
Brlapi                0.8.5
Brotli                1.1.0
bsptester             25.15.3
build                 1.3.0
certifi               2023.11.17
cffi                  2.0.0
chardet               5.2.0
click                 8.1.6
cloudpickle           3.1.1
colorama              0.4.6
cryptography          41.0.7
cssselect             1.2.0
cupshelpers           1.0
Cython                3.1.3
dbus-python           1.3.2
defer                 1.0.6
distlib               0.3.8
distro                1.9.0
distro-info           1.7+build1
duplicity             2.1.4
et-xmlfile            1.1.0
fasteners             0.18
filelock              3.13.1
fsspec                2025.9.0
geoip2                2.9.0
html5lib              1.1
httplib2              0.20.4
huggingface-hub       0.34.4
idna                  3.6
inflect               7.0.0
jaraco.context        4.3.0
jaraco.functools      4.0.0
jaraco.text           3.11.1
language-selector     0.1
launchpadlib          1.11.0
lavatester            24.13.7
lazr.restfulclient    0.14.6
lazr.uri              1.0.6
louis                 3.29.0
lxml                  5.2.1
markdown-it-py        3.0.0
maxminddb             2.5.2
mdurl                 0.1.2
meson                 1.3.2
ml_dtypes             0.5.3
monotonic             1.6
more-itertools        10.2.0
mutagen               1.46.0
mysqlclient           1.4.6
netifaces             0.11.0
ninja                 1.13.0
numpy                 1.26.4
oauthlib              3.2.2
olefile               0.46
openpyxl              3.1.2
ostester              25.14.2
packaging             24.2
PAM                   0.4.2
paramiko              2.12.0
pexpect               4.9.0
pillow                10.2.0
pip                   25.2
platformdirs          4.2.0
protobuf              6.32.0
psutil                7.0.0
ptyprocess            0.7.0
pycairo               1.25.1
pycparser             2.23
pycryptodomex         3.20.0
pycups                2.0.1
pycurl                7.45.3
pydantic              1.10.14
Pygments              2.17.2
PyGObject             3.48.2
PyJWT                 2.7.0
pylibacl              0.7.0
PyNaCl                1.5.0
pyparsing             3.1.1
pyproject_hooks       1.2.0
python-apt            2.7.7+ubuntu1.bb2
python-dateutil       2.8.2
python-debian         0.1.49+ubuntu2
pyxattr               0.8.1
pyxdg                 0.28
PyYAML                6.0.1
regex                 2025.9.1
requests              2.31.0
rich                  13.7.1
safetensors           0.6.2
scipy                 1.16.2
setuptools            80.9.0
six                   1.16.0
soundfile             0.13.1
soupsieve             2.5
spacemit-ort          1.2.2
ssh-import-id         5.11
systemd-python        235
terminaltables        3.1.10
titan-suit            25.4.5
titanauto             25.14.5
tokenizers            0.22.0
tqdm                  4.67.1
transformers          4.56.1
tvm                   0.20.0
typeguard             4.1.5
typing_extensions     4.10.0
ubuntu-drivers-common 0.0.0
ufw                   0.36.2
unattended-upgrades   0.1
urllib3               2.0.7
virtualenv            20.25.0+ds
wadllib               1.3.6
webencodings          0.5.1
websockets            10.4
wheel                 0.45.1
xdg                   5
xkit                  0.0.0
yt-dlp                2024.4.9
zipp                  1.0.0

```
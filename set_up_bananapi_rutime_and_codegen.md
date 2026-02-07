# List of files modified

You need to check every file listed below, use `ctrl-f` to find all “bananapi” and “BANANAPi” to replace it with your intended name.

```php
config.cmake :              tvm/build/config.cmake
CMakeLists.txt :            tvm/CMakeLists.txt
LibInfo.cmake :             tvm/cmake/modules/LibInfo.cmake
bananapi.cmake :            tvm/cmake/modules/contrib/bananapi.cmake
libinfo.cc :                tvm/src/support/libinfo.cc
bananapi_codegen.cc :       tvm/src/relax/backend/contrib/bananapi/bananapi_codegen.cc
bananapi_runtime.cc :       tvm/src/runtime/contrib/bananapi/bananapi_runtime.cc

```

To know more, you must checkout zin’s code

[source_code.tar.gz](source_code.tar.gz)

# Note

## bananapi.cmake

Note: `list(APPEND RUNTIME_BANANAPI_SRCS src/runtime/contrib/bananapi/libmatmul.cpp)` is commented, because this is static-compilation of BYOC approach, which requires you to write libmatmul.h (which is use for declaring `matmul()` in bananapi_runtime.cc). In our case, we cross-compile 3 whisper-tiny models for risc-v board, and TVM runtime compilation on x86 can’t use risc-v toolchain, so we use `dlopen()` approach, which does not require libmatmul.cpp to be compiled in x86 TVM compilation and can be later compiled by ourself on banana pi using its native `g++`.

Note: changes in TVM's c++ code, requires you to `cmake --build . --parallel $(nproc)`

/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file src/runtime/contrib/bananapi/bananapi_runtime.cc
 * \brief JSON runtime implementation for bananapi.
 */

#include <dmlc/parameter.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../file_utils.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

// user add
#include<stdio.h>
#include<dlfcn.h>
#include<stdlib.h>
#include<iostream>
// #include "matmul.h"

namespace tvm {
namespace runtime {
namespace contrib {

using namespace tvm::runtime::json;

class bananapi_Runtime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The bananapi runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit bananapi_Runtime(const std::string& symbol_name, const std::string& graph_json,
                           const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names) {}

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const final { return "bananapi"; }

  /*!
   * \brief Initialize runtime. Create bananapi layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    SetupConstants(consts);
  }

  ~bananapi_Runtime() override {
    VLOG(1) << "Destroying bananapi runtime";
    VLOG(1) << "Destroyed bananapi runtime";
  }

  /*! \brief Run inference using built engine. */
  void Run() override {

    arg_arr.clear();
    type_arr.clear();
    buffer_arr.clear();   

    // for(size_t i=0; i<nodes_.size(); i++){
      
    //   // shape
    //   // the following implementation is add operator's runtime, i need to implement matrix multiplication, which need to get the shape of m,n,o
    //   auto A_shape = nodes_[i].GetOpShape()[0][0];
    //   auto B_shape = nodes_[i].GetOpShape()[0][1];

    //   int64_t n = A_shape[A_shape.size() - 2];
    //   int64_t m = A_shape[A_shape.size() - 1];
    //   int64_t o = B_shape[B_shape.size() - 1];
    //   std::cout<<"N: "<<n<<", M: "<<m<<", O: "<<o<<std::endl;
    //   arg_arr.push_back(n);
    //   arg_arr.push_back(m);
    //   arg_arr.push_back(o);


    //   // dtype
    //   auto tmp = nodes_[i].GetOpDataType();
    //   auto node_type = tmp[0].code;
    //   auto node_type_byte = tmp[0].bits/8;
    //   type_arr.push_back({node_type, node_type_byte});


    //   auto buffer = (float*)malloc(node_arg * node_type_byte);
    //   buffer_arr.push_back(buffer);
    // }
    // for instance, we have matmul of [6, 1500, 384] multiply by [384, 384], they are equivalent with [x, n, m] multiply by [m, o]
    // std::cout<<"Run() starts here"<<std::endl;

    A_shape = nodes_[0].GetOpShape()[0]; // vector<int64_t>
    //A_shape = data_entry_[0]->shape; // vector<int64_t>
    B_shape = nodes_[1].GetOpShape()[0]; // vector<int64_t>
    //B_shape = data_entry_[1]->shape; // vector<int64_t>
    // std::cout<<A_shape<<std::endl;
    // std::cout<<B_shape<<std::endl;
    
    int64_t x = A_shape[A_shape.size() - 3];
    int64_t n = A_shape[A_shape.size() - 2];
    int64_t m = A_shape[A_shape.size() - 1];
    int64_t o = B_shape[B_shape.size() - 1];
    //std::cout<<"X: "<<x<<" N: "<<n<<", M: "<<m<<", O: "<<o<<std::endl;
    arg_arr.push_back(x);
    arg_arr.push_back(n);
    arg_arr.push_back(m);
    arg_arr.push_back(o);
    // for (size_t i = 0; i < input_nodes_.size(); ++i) {
    //   auto nid = input_nodes_[i];
    //   if (nodes_[nid].GetOpType() == "input") {
    //     memcpy(buffer_arr[nid], data_entry_[nid]->data, arg_arr[nid] * type_arr[nid].second);// data_entry's 0 and 1 is input A and B
    //   }
    // }

    // for (size_t i = 0; i < outputs_.size(); ++i) {
    //   auto tmp = outputs_[i];
    //   auto nid = tmp.id_;
    //   memcpy(buffer_arr[nid], data_entry_[nid]->data, arg_arr[nid] * type_arr[nid].second);
    // }

    // for(size_t i=0; i<nodes_.size(); i++){
    //   auto input = nodes_[i].GetInputs();
    //   std::cout << "node: " << i << std::endl;
    //   std::cout << "\top_type: " << nodes_[i].GetOpType() << std::endl;
    //   std::cout << "\tname:    " << nodes_[i].GetOpName() << std::endl;
    //   for(size_t j=0; j<input.size(); j++){
    //     std::cout << "\tinput " << j << " : " << input[j].id_ << std::endl;
    //   }
    // }

    // for(int i=0; i<data_entry_[0]->ndim; i++)
    //   std::cout << "shape " << i << " : " << *(data_entry_[0]->shape+i) << std::endl;
    
    for(size_t nid=0; nid<nodes_.size(); nid++){
      if(nodes_[nid].GetOpType() == "kernel"){
        if(nodes_[nid].GetOpName() == "bananapi.matmul")
          bananapi_matmul(nid);
        // 後續增加其他 OP
        else;
      }
    }
    // if we directly write data to data_entry_'s [2], then buffer_arr is not necessary

    // for (size_t i = 0; i < outputs_.size(); ++i) {  
    //   auto tmp = outputs_[i];
    //   auto nid = tmp.id_; // that constant is 2
    //   memcpy(data_entry_[nid]->data, buffer_arr[nid], arg_arr[nid] * type_arr[nid].second); // put back answer to [2]
    // }
  }

 private:
 
  using MatmulFn =
      void (*)(std::vector<const DLTensor*>&, std::vector<int64_t>&, std::vector<int64_t>&);
  // 一定要宣告成 class 成員
  void* so_handle_{nullptr};
  MatmulFn matmul_fp_{nullptr};
  
  void EnsureMatmulLoaded() {
    if (matmul_fp_) return;

    const char* env_path = std::getenv("BANANAPI_MATMUL_SO");
    std::vector<const char*> candidates;
    if (env_path && *env_path) candidates.push_back(env_path);
    candidates.push_back("libmatmul.so");
    candidates.push_back("./libmatmul.so");
    candidates.push_back("/home/fre930727/tvm/src/runtime/contrib/bananapi/libmatmul.so");


    for (const char* p : candidates) {
      so_handle_ = dlopen(p, RTLD_NOW | RTLD_LOCAL);
      if (!so_handle_) continue;
      void* sym = dlsym(so_handle_, "matmul");
      if (!sym) {
        dlclose(so_handle_);
        so_handle_ = nullptr;
        continue;
      }
      matmul_fp_ = reinterpret_cast<MatmulFn>(sym);
      break;
    }

    ICHECK(matmul_fp_ != nullptr)
        << "Failed to load symbol 'matmul' from shared library. "
        << "Set BANANAPI_MATMUL_SO to the absolute path of your RVV matmul .so. "
        << "dlerror: " << dlerror();
  }

  // ---------------- 改寫這個：用 dlsym 叫進來 ----------------
  void bananapi_matmul(size_t idx) {
    (void)idx;  // 目前用不到，但保留介面
    EnsureMatmulLoaded();
    // 直接用外部 .so 的 matmul 實作：就吃 data_entry_ / A_shape / B_shape
    matmul_fp_(data_entry_, A_shape, B_shape);
  }
  //void bananapi_matmul(size_t idx){
    // open shared library

    // void *handle = dlopen("/home/fre930727/tvm/libmatmul.so", RTLD_LAZY);
    // if (!handle) {
    //     printf("open lib error\n");
    //     return;
    // }

    // // define function pointer type
    // typedef void (*matmul_t)(std::vector<const DLTensor*>& , std::vector<int64_t>& , std::vector<int64_t>& );

    // // load symbol
    // matmul_t matmul = (matmul_t)dlsym(handle, "matmul");

    // if (!matmul) {
    //     printf("use function error\n");
    //     dlclose(handle);
    //     return;
    // }
      
      //if(data_entry_[0]->dtype.code == 0)
      //  typedef int TYPE;
      //else if(data_entry_[0]->dtype.code == 2)
      //  typedef float TYPE;
      //else;
      
      //typedef float TYPE;

      /*
      int size = (data_entry_[0]->dtype.bits/8);
      TYPE* out = (TYPE*)malloc(arg * size);*/

      // inputs 
      // std::vector<TYPE*> input_arr;
      // auto inputs = nodes_[idx].GetInputs();
      // for (size_t i = 0; i < inputs.size(); ++i) {
      //   auto nid = inputs[i].id_;
      //   input_arr.push_back((TYPE*)(buffer_arr[nid]));
      // }
      // auto out = (TYPE*)(buffer_arr[idx]);

      /*
      for (size_t i = 0; i < input_nodes_.size(); ++i) {
        auto nid = input_nodes_[i];
        if (nodes_[nid].GetOpType() == "input") {
          TYPE* tmp = (TYPE*)malloc(arg * size);
          memcpy(tmp, data_entry_[nid]->data, arg * size);
          input_arr.push_back(tmp);
        }
      }*/

      // add(input_arr[0], input_arr[1], out, arg_arr[idx]);
      //matmul(data_entry_, A_shape, B_shape);

      // outputs 
      /*
      for (size_t i = 0; i < outputs_.size(); ++i) {
        uint32_t eid = EntryID(outputs_[i]);
        memcpy(data_entry_[eid]->data, out, arg * size);
      }*/

      //free(out);
      //dlclose(handle);
    //}

    std::vector<int64_t> arg_arr;
    std::vector<std::pair<uint8_t, uint8_t>> type_arr;
    std::vector<void*> buffer_arr;
    std::vector<int64_t> A_shape;
    std::vector<int64_t> B_shape;
};

runtime::Module bananapiRuntimeCreate(const String& symbol_name, const String& graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<bananapi_Runtime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.bananapi_runtime_create").set_body_typed(bananapiRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_bananapi")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<bananapi_Runtime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
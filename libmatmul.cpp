#include <iostream>
#include <vector>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlpack/dlpack.h>
#include <riscv_vector.h>

// === Blocking tile size ===
#ifndef MC
#define MC 64 //在 A 矩陣上分塊的行數
#endif
#ifndef NC
#define NC 128 //在 B 矩陣上分塊的列數
#endif
#ifndef KC
#define KC 128 //在內積方向上的塊大小
#endif

// -------------------- 打包 B -------------------- //可能有bug(M,O顛倒)
static inline void pack_B_kc_nc_f32_colmajor(
    const float* B, int o,          // B has o columns
    int pc, int kc, int jc, int nc,
    float* Bp
) {
    for (int j = 0; j < nc; ++j) {
        const float* Bj = B + (size_t)(jc + j); 
        for (int kk = 0; kk < kc; ++kk) {
            Bp[j * kc + kk] = Bj[(size_t)(pc + kk) * (size_t)o];
        }
    }
}

// -------------------- 微核心 (RVV e32m1) --------------------
static inline void microkernel_accum_f32A_f32B_to_f32C_e32m1(
    const float* Ablk, int lda,
    const float* Bp,
    float* Cblk, int ldc,
    int mc, int kc, int jTail
) {
    for (int i = 0; i < mc; ++i) {
        int col = 0;
        while (col < jTail) {
            size_t vl = __riscv_vsetvl_e32m1((size_t)(jTail - col));
            float* Cptr = Cblk + (size_t)i * (size_t)ldc + (size_t)col;
            vfloat32m1_t vacc = __riscv_vle32_v_f32m1(Cptr, vl);

            const float* Ai = Ablk + (size_t)i * (size_t)lda;
            const float* Bcol = Bp + (size_t)col * (size_t)kc;

            for (int kk = 0; kk < kc; ++kk) {
                float a_scalar = Ai[kk];
                const float* Bvec_ptr = Bcol + (size_t)kk;
                vfloat32m1_t bv = __riscv_vlse32_v_f32m1(
                    Bvec_ptr, (ptrdiff_t)((size_t)kc * sizeof(float)), vl);
                vacc = __riscv_vfmacc_vf_f32m1(vacc, a_scalar, bv, vl);
            }

            __riscv_vse32_v_f32m1(Cptr, vacc, vl);
            col += (int)vl;
        }
    }
}

// -------------------- Blocked matmul using RVV kernel -------------------- //可能有bug(M,O顛倒)
void matmul_f32_blocked_rvv_e32(
    const float* A, const float* B, float* C,
    int n, int m, int o
) {
		static float Bpack[KC * NC] alignas(64);

    for (int jc = 0; jc < o; jc += NC) {
        int jTail = (jc + NC <= o) ? NC : (o - jc);
        for (int ic = 0; ic < n; ic += MC) {
            int mc = (ic + MC <= n) ? MC : (n - ic);

            for (int i = 0; i < mc; ++i) {
                float* Crow = C + (size_t)(ic + i) * (size_t)o + (size_t)jc;
                memset(Crow, 0, (size_t)jTail * sizeof(float));
            }

            for (int pc = 0; pc < m; pc += KC) {
                int kc = (pc + KC <= m) ? KC : (m - pc);
                pack_B_kc_nc_f32_colmajor(B, o, pc, kc, jc, jTail, Bpack);

                const float* Ablk = A + (size_t)ic * (size_t)m + (size_t)pc; 
                float* Cblk = C + (size_t)ic * (size_t)o + (size_t)jc;      
                microkernel_accum_f32A_f32B_to_f32C_e32m1(
                    Ablk, m, Bpack, Cblk, o, mc, kc, jTail
                );
            }
        }
    }
}

// -------------------- Batch x Batch --------------------
void matmul_bxb(std::vector<const DLTensor*>& data_entry_,
                int n, int m, int o, int batch) {

    const float* A = static_cast<const float*>(data_entry_[0]->data);
    const float* B = static_cast<const float*>(data_entry_[1]->data);
    float*       C = static_cast<float*>(data_entry_[2]->data);

    for (int bidx = 0; bidx < batch; ++bidx) {
        const float* Ab = A + (size_t)bidx * (size_t)n * (size_t)m;
        const float* Bb = B + (size_t)bidx * (size_t)m * (size_t)o;
        float*       Cb = C + (size_t)bidx * (size_t)n * (size_t)o;
        matmul_f32_blocked_rvv_e32(Ab, Bb, Cb, n, m, o);
    }
}

// -------------------- Batch x Single --------------------
void matmul_bxs(std::vector<const DLTensor*>& data_entry_,
                int n, int m, int o, int batch) {

    const float* A = static_cast<const float*>(data_entry_[0]->data);
    const float* B = static_cast<const float*>(data_entry_[1]->data); // single B
    float*       C = static_cast<float*>(data_entry_[2]->data);

    for (int bidx = 0; bidx < batch; ++bidx) {
        const float* Ab = A + (size_t)bidx * (size_t)n * (size_t)m;
        float*       Cb = C + (size_t)bidx * (size_t)n * (size_t)o;
        matmul_f32_blocked_rvv_e32(Ab, B, Cb, n, m, o);
    }
}

// -------------------- Wrapper --------------------
extern "C"
void matmul(std::vector<const DLTensor*>& data_entry_,
            std::vector<int64_t>& shapeA,
            std::vector<int64_t>& shapeB) {
    int n, m, o, x;
    //std::cout<<"matmul() starts"<<std::endl;
    if (shapeB.size() == 3) {
        x = (int)shapeA[0];
        n = (int)shapeA[1];
        m = (int)shapeA[2];
        o = (int)shapeB[2];
        matmul_bxb(data_entry_, n, m, o, x);
    } else {
        x = (int)shapeA[0];
        n = (int)shapeA[1];
        m = (int)shapeA[2];
        o = (int)shapeB[1];
        matmul_bxs(data_entry_, n, m, o, x);
    }
    //std::cout<<"matmul() ends"<<std::endl;
}
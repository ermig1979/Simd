/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/
#include "Simd/SimdMemory.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdNeon.h"

namespace Simd
{
#if defined(SIMD_NEON_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Neon
    {
        SIMD_INLINE uint8x16_t LoadTail(const uint8_t* ptr, size_t tail)
        {
            uint8_t buf[A] = { 0 };
            for (size_t i = 0; i < tail; ++i)
                buf[i] = ptr[i];
            return Load<false>(buf);
        }

        SIMD_INLINE int8x16_t LoadTail(const int8_t* ptr, size_t tail)
        {
            int8_t buf[A] = { 0 };
            for (size_t i = 0; i < tail; ++i)
                buf[i] = ptr[i];
            return Load<false>(buf);
        }

        SIMD_INLINE int ExtractInt32Sum(int32x4_t value)
        {
            int32_t buf[4];
            vst1q_s32(buf, value);
            return buf[0] + buf[1] + buf[2] + buf[3];
        }

        static SIMD_INLINE void Save4Sums(const int32x4_t& sum0, const int32x4_t& sum1, const int32x4_t& sum2, const int32x4_t& sum3, int32_t* dst)
        {
            dst[0] = ExtractInt32Sum(sum0);
            dst[1] = ExtractInt32Sum(sum1);
            dst[2] = ExtractInt32Sum(sum2);
            dst[3] = ExtractInt32Sum(sum3);
        }

        template<bool overflow> static void SynetInnerProduct8i1x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            int32x4_t d00 = vdupq_n_s32(0);
            uint8x16_t s0;
            int8x16_t w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = Load<false>(S0 + k);
                w0 = Load<false>(W0 + k);
                Madd4<overflow>(d00, s0, w0);
            }
            if (KA < K)
            {
                size_t tail = K - KA;
                s0 = LoadTail(S0 + KA, tail);
                w0 = LoadTail(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
            }
            D[0] = ExtractInt32Sum(d00);
        }

        template<bool overflow> static void SynetInnerProduct8i1x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            int32x4_t d00 = vdupq_n_s32(0);
            int32x4_t d01 = vdupq_n_s32(0);
            int32x4_t d02 = vdupq_n_s32(0);
            int32x4_t d03 = vdupq_n_s32(0);
            uint8x16_t s0;
            int8x16_t w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = Load<false>(S0 + k);
                w0 = Load<false>(W0 + k);
                Madd4<overflow>(d00, s0, w0);
                w0 = Load<false>(W1 + k);
                Madd4<overflow>(d01, s0, w0);
                w0 = Load<false>(W2 + k);
                Madd4<overflow>(d02, s0, w0);
                w0 = Load<false>(W3 + k);
                Madd4<overflow>(d03, s0, w0);
            }
            if (KA < K)
            {
                size_t tail = K - KA;
                s0 = LoadTail(S0 + KA, tail);
                w0 = LoadTail(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                w0 = LoadTail(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                w0 = LoadTail(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                w0 = LoadTail(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
            }
            Save4Sums(d00, d01, d02, d03, D);
        }

        template<bool overflow> static void SynetInnerProduct8i2x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            int32x4_t d00 = vdupq_n_s32(0);
            int32x4_t d10 = vdupq_n_s32(0);
            uint8x16_t s0, s1;
            int8x16_t w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = Load<false>(S0 + k);
                s1 = Load<false>(S1 + k);
                w0 = Load<false>(W0 + k);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            if (KA < K)
            {
                size_t tail = K - KA;
                s0 = LoadTail(S0 + KA, tail);
                s1 = LoadTail(S1 + KA, tail);
                w0 = LoadTail(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            D[0 * ldd] = ExtractInt32Sum(d00);
            D[1 * ldd] = ExtractInt32Sum(d10);
        }

        template<bool overflow> static void SynetInnerProduct8i2x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            size_t KA = AlignLo(K, A);
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            int32x4_t d00 = vdupq_n_s32(0);
            int32x4_t d01 = vdupq_n_s32(0);
            int32x4_t d02 = vdupq_n_s32(0);
            int32x4_t d03 = vdupq_n_s32(0);
            int32x4_t d10 = vdupq_n_s32(0);
            int32x4_t d11 = vdupq_n_s32(0);
            int32x4_t d12 = vdupq_n_s32(0);
            int32x4_t d13 = vdupq_n_s32(0);
            uint8x16_t s0, s1;
            int8x16_t w0;
            for (size_t k = 0; k < KA; k += A)
            {
                s0 = Load<false>(S0 + k);
                s1 = Load<false>(S1 + k);
                w0 = Load<false>(W0 + k);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = Load<false>(W1 + k);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = Load<false>(W2 + k);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = Load<false>(W3 + k);
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            if (KA < K)
            {
                size_t tail = K - KA;
                s0 = LoadTail(S0 + KA, tail);
                s1 = LoadTail(S1 + KA, tail);
                w0 = LoadTail(W0 + KA, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = LoadTail(W1 + KA, tail);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = LoadTail(W2 + KA, tail);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = LoadTail(W3 + KA, tail);
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            Save4Sums(d00, d01, d02, d03, D + 0 * ldd);
            Save4Sums(d10, d11, d12, d13, D + 1 * ldd);
        }

        template<bool overflow> void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst)
        {
            size_t M2 = AlignLoAny(M, 2);
            size_t N4 = AlignLoAny(N, 4);
            size_t i = 0;
            for (; i < M2; i += 2)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i2x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i2x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K * 2;
                dst += N * 2;
            }
            for (; i < M; i += 1)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    SynetInnerProduct8i1x4<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                for (; j < N; j += 1)
                    SynetInnerProduct8i1x1<overflow>(K, src, K, weight + j * K, K, dst + j, N);
                src += K;
                dst += N;
            }
        }

        void SynetInnerProduct8i(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::Precise(compatibility))
                SynetInnerProduct8i<false>(M, N, K, src, weight, dst);
            else
                SynetInnerProduct8i<true>(M, N, K, src, weight, dst);
        }
    }
#endif
}

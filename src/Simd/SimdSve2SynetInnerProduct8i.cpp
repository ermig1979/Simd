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
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSve2.h"

namespace Simd
{
#if defined(SIMD_SVE2_ENABLE) && defined(SIMD_SYNET_ENABLE)
    namespace Sve2
    {
        SIMD_INLINE svuint8_t LoadU8(const uint8_t* ptr, const svbool_t& mask)
        {
            return svsel_u8(mask, svld1_u8(mask, ptr), svdup_n_u8(0));
        }

        SIMD_INLINE svint8_t LoadI8(const int8_t* ptr, const svbool_t& mask)
        {
            return svsel_s8(mask, svld1_s8(mask, ptr), svdup_n_s8(0));
        }

        SIMD_INLINE int32_t ExtractInt32Sum(const svint32_t& sum)
        {
            return (int32_t)svaddv_s32(svptrue_b32(), sum);
        }

        static SIMD_INLINE void Save4Sums(const svint32_t& sum0, const svint32_t& sum1, const svint32_t& sum2, const svint32_t& sum3, int32_t* dst)
        {
            dst[0] = ExtractInt32Sum(sum0);
            dst[1] = ExtractInt32Sum(sum1);
            dst[2] = ExtractInt32Sum(sum2);
            dst[3] = ExtractInt32Sum(sum3);
        }

        template<bool overflow> static void SynetInnerProduct8i1x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            const size_t A = svcntb();
            const size_t KA = AlignLo(K, A);
            const svbool_t body = svptrue_b8();
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            svint32_t d00 = svdup_n_s32(0);
            svuint8_t s0;
            svint8_t w0;
            size_t k = 0;
            for (; k < KA; k += A)
            {
                s0 = LoadU8(S0 + k, body);
                w0 = LoadI8(W0 + k, body);
                Madd4<overflow>(d00, s0, w0);
            }
            if (k < K)
            {
                const svbool_t tail = svwhilelt_b8(k, K);
                s0 = LoadU8(S0 + k, tail);
                w0 = LoadI8(W0 + k, tail);
                Madd4<overflow>(d00, s0, w0);
            }
            D[0] = ExtractInt32Sum(d00);
        }

        template<bool overflow> static void SynetInnerProduct8i1x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            const size_t A = svcntb();
            const size_t KA = AlignLo(K, A);
            const svbool_t body = svptrue_b8();
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            svint32_t d00 = svdup_n_s32(0);
            svint32_t d01 = svdup_n_s32(0);
            svint32_t d02 = svdup_n_s32(0);
            svint32_t d03 = svdup_n_s32(0);
            svuint8_t s0;
            svint8_t w0;
            size_t k = 0;
            for (; k < KA; k += A)
            {
                s0 = LoadU8(S0 + k, body);
                w0 = LoadI8(W0 + k, body);
                Madd4<overflow>(d00, s0, w0);
                w0 = LoadI8(W1 + k, body);
                Madd4<overflow>(d01, s0, w0);
                w0 = LoadI8(W2 + k, body);
                Madd4<overflow>(d02, s0, w0);
                w0 = LoadI8(W3 + k, body);
                Madd4<overflow>(d03, s0, w0);
            }
            if (k < K)
            {
                const svbool_t tail = svwhilelt_b8(k, K);
                s0 = LoadU8(S0 + k, tail);
                w0 = LoadI8(W0 + k, tail);
                Madd4<overflow>(d00, s0, w0);
                w0 = LoadI8(W1 + k, tail);
                Madd4<overflow>(d01, s0, w0);
                w0 = LoadI8(W2 + k, tail);
                Madd4<overflow>(d02, s0, w0);
                w0 = LoadI8(W3 + k, tail);
                Madd4<overflow>(d03, s0, w0);
            }
            Save4Sums(d00, d01, d02, d03, D);
        }

        template<bool overflow> static void SynetInnerProduct8i2x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            const size_t A = svcntb();
            const size_t KA = AlignLo(K, A);
            const svbool_t body = svptrue_b8();
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            svint32_t d00 = svdup_n_s32(0);
            svint32_t d10 = svdup_n_s32(0);
            svuint8_t s0, s1;
            svint8_t w0;
            size_t k = 0;
            for (; k < KA; k += A)
            {
                s0 = LoadU8(S0 + k, body);
                s1 = LoadU8(S1 + k, body);
                w0 = LoadI8(W0 + k, body);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            if (k < K)
            {
                const svbool_t tail = svwhilelt_b8(k, K);
                s0 = LoadU8(S0 + k, tail);
                s1 = LoadU8(S1 + k, tail);
                w0 = LoadI8(W0 + k, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
            }
            D[0 * ldd] = ExtractInt32Sum(d00);
            D[1 * ldd] = ExtractInt32Sum(d10);
        }

        template<bool overflow> static void SynetInnerProduct8i2x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            const size_t A = svcntb();
            const size_t KA = AlignLo(K, A);
            const svbool_t body = svptrue_b8();
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            svint32_t d00 = svdup_n_s32(0);
            svint32_t d01 = svdup_n_s32(0);
            svint32_t d02 = svdup_n_s32(0);
            svint32_t d03 = svdup_n_s32(0);
            svint32_t d10 = svdup_n_s32(0);
            svint32_t d11 = svdup_n_s32(0);
            svint32_t d12 = svdup_n_s32(0);
            svint32_t d13 = svdup_n_s32(0);
            svuint8_t s0, s1;
            svint8_t w0;
            size_t k = 0;
            for (; k < KA; k += A)
            {
                s0 = LoadU8(S0 + k, body);
                s1 = LoadU8(S1 + k, body);
                w0 = LoadI8(W0 + k, body);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = LoadI8(W1 + k, body);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = LoadI8(W2 + k, body);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = LoadI8(W3 + k, body);
                Madd4<overflow>(d03, s0, w0);
                Madd4<overflow>(d13, s1, w0);
            }
            if (k < K)
            {
                const svbool_t tail = svwhilelt_b8(k, K);
                s0 = LoadU8(S0 + k, tail);
                s1 = LoadU8(S1 + k, tail);
                w0 = LoadI8(W0 + k, tail);
                Madd4<overflow>(d00, s0, w0);
                Madd4<overflow>(d10, s1, w0);
                w0 = LoadI8(W1 + k, tail);
                Madd4<overflow>(d01, s0, w0);
                Madd4<overflow>(d11, s1, w0);
                w0 = LoadI8(W2 + k, tail);
                Madd4<overflow>(d02, s0, w0);
                Madd4<overflow>(d12, s1, w0);
                w0 = LoadI8(W3 + k, tail);
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

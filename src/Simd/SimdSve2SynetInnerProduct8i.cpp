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

        template<bool overflow> struct Madd;

        template<> struct Madd<false>
        {
            struct Sum
            {
                svint32_t lo0, lo1, hi0, hi1;
            };

            static SIMD_INLINE Sum Zero()
            {
                Sum sum = { svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0), svdup_n_s32(0) };
                return sum;
            }

            static SIMD_INLINE void Add(Sum& sum, const uint8_t* src, const int8_t* weight, const svbool_t& mask)
            {
                svuint8_t _src = LoadU8(src, mask);
                svint8_t _weight = LoadI8(weight, mask);
                svint16_t srcLo = svreinterpret_s16_u16(svmovlb_u16(_src));
                svint16_t srcHi = svreinterpret_s16_u16(svmovlt_u16(_src));
                svint16_t weightLo = svmovlb_s16(_weight);
                svint16_t weightHi = svmovlt_s16(_weight);
                sum.lo0 = svmlalb_s32(sum.lo0, srcLo, weightLo);
                sum.lo1 = svmlalt_s32(sum.lo1, srcLo, weightLo);
                sum.hi0 = svmlalb_s32(sum.hi0, srcHi, weightHi);
                sum.hi1 = svmlalt_s32(sum.hi1, srcHi, weightHi);
            }

            static SIMD_INLINE int32_t Extract(const Sum& sum)
            {
                const svbool_t body = svptrue_b32();
                svint32_t sums = svadd_s32_x(body, svadd_s32_x(body, sum.lo0, sum.lo1), svadd_s32_x(body, sum.hi0, sum.hi1));
                return (int32_t)svaddv_s32(body, sums);
            }
        };

        template<> struct Madd<true>
        {
            typedef int32_t Sum;

            static SIMD_INLINE Sum Zero()
            {
                return 0;
            }

            static SIMD_INLINE int32_t PairSum(const svint16_t& products)
            {
                svint16_t even = svuzp1_s16(products, products);
                svint16_t odd = svuzp2_s16(products, products);
                svint16_t pairs = svqadd_s16(even, odd);
                return (int32_t)svaddv_s16(svwhilelt_b16((size_t)0, svcnth() / 2), pairs);
            }

            static SIMD_INLINE void Add(Sum& sum, const uint8_t* src, const int8_t* weight, const svbool_t& mask)
            {
                const svbool_t body = svptrue_b16();
                svuint8_t _src = LoadU8(src, mask);
                svint8_t _weight = LoadI8(weight, mask);
                svint16_t lo = svmul_s16_x(body, svreinterpret_s16_u16(svmovlb_u16(_src)), svmovlb_s16(_weight));
                svint16_t hi = svmul_s16_x(body, svreinterpret_s16_u16(svmovlt_u16(_src)), svmovlt_s16(_weight));
                sum += PairSum(lo) + PairSum(hi);
            }

            static SIMD_INLINE int32_t Extract(const Sum& sum)
            {
                return sum;
            }
        };

        template<bool overflow> static SIMD_INLINE void Add(size_t K, const uint8_t* src, const int8_t* weight, typename Madd<overflow>::Sum& sum)
        {
            const size_t A = svcntb();
            const size_t KA = AlignLo(K, A);
            const svbool_t body = svptrue_b8();
            size_t k = 0;
            for (; k < KA; k += A)
                Madd<overflow>::Add(sum, src + k, weight + k, body);
            if (k < K)
                Madd<overflow>::Add(sum, src + k, weight + k, svwhilelt_b8(k, K));
        }

        template<bool overflow> static void SynetInnerProduct8i1x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            typename Madd<overflow>::Sum d00 = Madd<overflow>::Zero();
            Add<overflow>(K, S0, W0, d00);
            D[0] = Madd<overflow>::Extract(d00);
        }

        template<bool overflow> static void SynetInnerProduct8i1x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            const uint8_t* S0 = S + 0 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            typename Madd<overflow>::Sum d00 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d01 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d02 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d03 = Madd<overflow>::Zero();
            Add<overflow>(K, S0, W0, d00);
            Add<overflow>(K, S0, W1, d01);
            Add<overflow>(K, S0, W2, d02);
            Add<overflow>(K, S0, W3, d03);
            D[0] = Madd<overflow>::Extract(d00);
            D[1] = Madd<overflow>::Extract(d01);
            D[2] = Madd<overflow>::Extract(d02);
            D[3] = Madd<overflow>::Extract(d03);
        }

        template<bool overflow> static void SynetInnerProduct8i2x1(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            typename Madd<overflow>::Sum d00 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d10 = Madd<overflow>::Zero();
            Add<overflow>(K, S0, W0, d00);
            Add<overflow>(K, S1, W0, d10);
            D[0 * ldd] = Madd<overflow>::Extract(d00);
            D[1 * ldd] = Madd<overflow>::Extract(d10);
        }

        template<bool overflow> static void SynetInnerProduct8i2x4(size_t K, const uint8_t* S, size_t lds, const int8_t* W, size_t ldw, int32_t* D, size_t ldd)
        {
            const uint8_t* S0 = S + 0 * lds;
            const uint8_t* S1 = S + 1 * lds;
            const int8_t* W0 = W + 0 * ldw;
            const int8_t* W1 = W + 1 * ldw;
            const int8_t* W2 = W + 2 * ldw;
            const int8_t* W3 = W + 3 * ldw;
            typename Madd<overflow>::Sum d00 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d01 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d02 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d03 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d10 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d11 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d12 = Madd<overflow>::Zero();
            typename Madd<overflow>::Sum d13 = Madd<overflow>::Zero();
            Add<overflow>(K, S0, W0, d00);
            Add<overflow>(K, S0, W1, d01);
            Add<overflow>(K, S0, W2, d02);
            Add<overflow>(K, S0, W3, d03);
            Add<overflow>(K, S1, W0, d10);
            Add<overflow>(K, S1, W1, d11);
            Add<overflow>(K, S1, W2, d12);
            Add<overflow>(K, S1, W3, d13);
            D[0 * ldd + 0] = Madd<overflow>::Extract(d00);
            D[0 * ldd + 1] = Madd<overflow>::Extract(d01);
            D[0 * ldd + 2] = Madd<overflow>::Extract(d02);
            D[0 * ldd + 3] = Madd<overflow>::Extract(d03);
            D[1 * ldd + 0] = Madd<overflow>::Extract(d10);
            D[1 * ldd + 1] = Madd<overflow>::Extract(d11);
            D[1 * ldd + 2] = Madd<overflow>::Extract(d12);
            D[1 * ldd + 3] = Madd<overflow>::Extract(d13);
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

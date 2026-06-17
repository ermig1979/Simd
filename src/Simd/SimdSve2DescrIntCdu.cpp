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
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        static void UnpackNorm(size_t count, const uint8_t* const* src, float* dst, size_t stride)
        {
            for (size_t i = 0; i < count; ++i)
            {
                const float* ps = (const float*)src[i];
                float* pd = dst + i * 4;
                pd[0] = ps[0];
                pd[1] = ps[1];
                pd[2] = ps[2];
                pd[3] = ps[3];
            }
        }

        Base::DescrInt::UnpackNormPtr GetUnpackNorm(bool transpose)
        {
            return UnpackNorm;
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> SIMD_INLINE uint64_t LoadBitsBlock(const uint8_t* src)
        {
            uint64_t value = 0;
            for (size_t i = 0; i < bits; ++i)
                value |= uint64_t(src[i]) << (8 * i);
            return value;
        }

        template<int bits> void UnpackData(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            const uint64_t mask = (uint64_t(1) << bits) - 1;
            for (size_t i = 0; i < count; ++i)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = dst + i * size;
                for (size_t j = 0; j < size; j += 8, ps += bits, pd += 8)
                {
                    uint64_t value = LoadBitsBlock<bits>(ps);
                    pd[0] = uint8_t((value >> (0 * bits)) & mask);
                    pd[1] = uint8_t((value >> (1 * bits)) & mask);
                    pd[2] = uint8_t((value >> (2 * bits)) & mask);
                    pd[3] = uint8_t((value >> (3 * bits)) & mask);
                    pd[4] = uint8_t((value >> (4 * bits)) & mask);
                    pd[5] = uint8_t((value >> (5 * bits)) & mask);
                    pd[6] = uint8_t((value >> (6 * bits)) & mask);
                    pd[7] = uint8_t((value >> (7 * bits)) & mask);
                }
            }
        }

        template<> void UnpackData<8>(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            const svuint8_t zero = svdup_n_u8(0);
            for (size_t i = 0; i < count; ++i)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = dst + i * size;
                for (size_t j = 0; j < size; j += svcntb())
                {
                    svbool_t mask = svwhilelt_b8(j, size);
                    svuint8_t value = svsel_u8(mask, svld1_u8(mask, ps + j), zero);
                    svst1_u8(mask, pd + j, value);
                }
            }
        }

        Base::DescrInt::UnpackDataPtr GetUnpackData(size_t depth)
        {
            switch (depth)
            {
            case 4: return UnpackData<4>;
            case 5: return UnpackData<5>;
            case 6: return UnpackData<6>;
            case 7: return UnpackData<7>;
            case 8: return UnpackData<8>;
            default: return NULL;
            }
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE uint32_t Correlation8u(const uint8_t* a, const uint8_t* b, size_t size)
        {
            const svuint8_t zero = svdup_n_u8(0);
            svuint32_t sums = svdup_n_u32(0);
            for (size_t i = 0; i < size; i += svcntb())
            {
                svbool_t mask = svwhilelt_b8(i, size);
                svuint8_t _a = svsel_u8(mask, svld1_u8(mask, a + i), zero);
                svuint8_t _b = svsel_u8(mask, svld1_u8(mask, b + i), zero);
                sums = svdot_u32(sums, _a, _b);
            }
            return (uint32_t)svaddv_u32(svptrue_b32(), sums);
        }

        SIMD_INLINE float DecodeCosineDistance(const float* a, const float* b, uint32_t abSum)
        {
            float ab = float(abSum) * a[0] * b[0] + a[2] * b[1] + b[2] * a[1];
            return Simd::RestrictRange(1.0f - ab / (a[3] * b[3]), 0.0f, 2.0f);
        }

        template<int M> SIMD_INLINE void CorrelationsMx1(size_t K, const uint8_t* ad, const uint8_t* bd, uint32_t* ab)
        {
            const svuint8_t zero = svdup_n_u8(0);
            svuint32_t sums0 = svdup_n_u32(0), sums1 = sums0, sums2 = sums0, sums3 = sums0;
            for (size_t i = 0; i < K; i += svcntb())
            {
                svbool_t mask = svwhilelt_b8(i, K);
                svuint8_t b = svsel_u8(mask, svld1_u8(mask, bd + i), zero);
                if (M > 0)
                {
                    svuint8_t a = svsel_u8(mask, svld1_u8(mask, ad + 0 * K + i), zero);
                    sums0 = svdot_u32(sums0, a, b);
                }
                if (M > 1)
                {
                    svuint8_t a = svsel_u8(mask, svld1_u8(mask, ad + 1 * K + i), zero);
                    sums1 = svdot_u32(sums1, a, b);
                }
                if (M > 2)
                {
                    svuint8_t a = svsel_u8(mask, svld1_u8(mask, ad + 2 * K + i), zero);
                    sums2 = svdot_u32(sums2, a, b);
                }
                if (M > 3)
                {
                    svuint8_t a = svsel_u8(mask, svld1_u8(mask, ad + 3 * K + i), zero);
                    sums3 = svdot_u32(sums3, a, b);
                }
            }
            if (M > 0) ab[0] = (uint32_t)svaddv_u32(svptrue_b32(), sums0);
            if (M > 1) ab[1] = (uint32_t)svaddv_u32(svptrue_b32(), sums1);
            if (M > 2) ab[2] = (uint32_t)svaddv_u32(svptrue_b32(), sums2);
            if (M > 3) ab[3] = (uint32_t)svaddv_u32(svptrue_b32(), sums3);
        }

        template<int M> SIMD_INLINE void MacroCorrelationMx1(size_t K, const uint8_t* ad, const float* an, const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            uint32_t ab[4];
            CorrelationsMx1<M>(K, ad, bd, ab);
            if (M > 0) distances[0 * stride] = DecodeCosineDistance(an + 0 * 4, bn, ab[0]);
            if (M > 1) distances[1 * stride] = DecodeCosineDistance(an + 1 * 4, bn, ab[1]);
            if (M > 2) distances[2 * stride] = DecodeCosineDistance(an + 2 * 4, bn, ab[2]);
            if (M > 3) distances[3 * stride] = DecodeCosineDistance(an + 3 * 4, bn, ab[3]);
        }

        void MacroCorrelation(size_t M, size_t N, size_t K, const uint8_t* ad, const float* an, const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            size_t M4 = AlignLoAny(M, 4);
            for (size_t j = 0; j < N; ++j)
            {
                size_t i = 0;
                const uint8_t* b = bd + j * K;
                const float* nb = bn + j * 4;
                for (; i < M4; i += 4)
                    MacroCorrelationMx1<4>(K, ad + i * K, an + i * 4, b, nb, distances + i * stride + j, stride);
                if (i < M)
                {
                    switch (M - i)
                    {
                    case 1: MacroCorrelationMx1<1>(K, ad + i * K, an + i * 4, b, nb, distances + i * stride + j, stride); break;
                    case 2: MacroCorrelationMx1<2>(K, ad + i * K, an + i * 4, b, nb, distances + i * stride + j, stride); break;
                    case 3: MacroCorrelationMx1<3>(K, ad + i * K, an + i * 4, b, nb, distances + i * stride + j, stride); break;
                    }
                }
            }
        }

        Base::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth)
        {
            return MacroCorrelation;
        }
    }
#endif// SIMD_SVE2_ENABLE
}

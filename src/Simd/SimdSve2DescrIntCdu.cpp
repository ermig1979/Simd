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
        SIMD_INLINE void LoadHeader(const uint8_t* src, float* dst)
        {
            memcpy(dst, src, 16);
        }

        static void UnpackNormA(size_t count, const uint8_t* const* src, float* dst, size_t)
        {
            for (size_t i = 0; i < count; ++i, dst += 4)
                LoadHeader(src[i], dst);
        }

        static void UnpackNormB(size_t count, const uint8_t* const* src, float* dst, size_t stride)
        {
            for (size_t i = 0; i < count; ++i)
            {
                float header[4];
                LoadHeader(src[i], header);
                dst[0 * stride + i] = header[0];
                dst[1 * stride + i] = header[1];
                dst[2 * stride + i] = header[2];
                dst[3 * stride + i] = header[3];
            }
        }

        Base::DescrInt::UnpackNormPtr GetUnpackNorm(bool transpose)
        {
            return transpose ? UnpackNormB : UnpackNormA;
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits> void UnpackData(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t)
        {
            const size_t valuesPerVec = svcntb();
            const size_t packedPerVec = valuesPerVec * bits / 8;
            const svuint8_t tbl = UnpackTbl<bits>();
            const svuint16_t shr = UnpackShr<bits>();
            const svbool_t all = svptrue_b8();
            for (size_t i = 0; i < count; ++i)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = dst + i * size;
                size_t j = 0, packed = 0;
                for (; j + valuesPerVec <= size; j += valuesPerVec, packed += packedPerVec)
                    svst1_u8(all, pd + j, UnpackTo8<bits>(ps + packed, packedPerVec, tbl, shr));
                if (j < size)
                {
                    size_t packedTail = (size - j) * bits / 8;
                    svst1_u8(svwhilelt_b8(j, size), pd + j, UnpackTo8<bits>(ps + packed, packedTail, tbl, shr));
                }
            }
        }

        template<> void UnpackData<4>(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t)
        {
            const size_t width = svcntb();
            const size_t byteSize = size / 2;
            const svbool_t all = svptrue_b8();
            for (size_t i = 0; i < count; ++i)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = dst + i * size;
                size_t j = 0;
                for (; j + width <= byteSize; j += width, ps += width, pd += 2 * width)
                {
                    svuint8_t value = svld1_u8(all, ps);
                    svuint8_t lo = svand_n_u8_x(all, value, 0x0F);
                    svuint8_t hi = svlsr_n_u8_x(all, value, 4);
                    svst1_u8(all, pd, svzip1_u8(lo, hi));
                    svst1_u8(all, pd + width, svzip2_u8(lo, hi));
                }
                for (; j < byteSize; ++j, ++ps, pd += 2)
                {
                    pd[0] = uint8_t(ps[0] & 0x0F);
                    pd[1] = uint8_t(ps[0] >> 4);
                }
            }
        }

        template<> void UnpackData<8>(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t)
        {
            const svbool_t all = svptrue_b8();
            const size_t width = svcntb();
            for (size_t i = 0; i < count; ++i)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = dst + i * size;
                size_t j = 0;
                for (; j + width <= size; j += width)
                    svst1_u8(all, pd + j, svld1_u8(all, ps + j));
                if (j < size)
                {
                    svbool_t mask = svwhilelt_b8(j, size);
                    svst1_u8(mask, pd + j, svld1_u8(mask, ps + j));
                }
            }
        }

        Base::DescrInt::UnpackDataPtr GetUnpackData(size_t depth, bool)
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

        template<int M> SIMD_INLINE void CorrelationMx4(size_t K, const uint8_t* ad, const float* an,
            const uint8_t* bd, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            const size_t width = svcntb();
            const size_t main = K & ~(width - 1);
            const svbool_t all = svptrue_b8();
            svuint32_t ab00 = svdup_n_u32(0), ab01 = ab00, ab02 = ab00, ab03 = ab00;
            svuint32_t ab10 = ab00, ab11 = ab00, ab12 = ab00, ab13 = ab00;
            svuint32_t ab20 = ab00, ab21 = ab00, ab22 = ab00, ab23 = ab00;
            svuint32_t ab30 = ab00, ab31 = ab00, ab32 = ab00, ab33 = ab00;
            const uint8_t* ad0 = ad;
            const uint8_t* ad1 = ad + K;
            const uint8_t* ad2 = ad + 2 * K;
            const uint8_t* ad3 = ad + 3 * K;
            const uint8_t* bd0 = bd;
            const uint8_t* bd1 = bd + K;
            const uint8_t* bd2 = bd + 2 * K;
            const uint8_t* bd3 = bd + 3 * K;
            size_t k = 0;
            for (; k < main; k += width)
            {
                svuint8_t b0 = svld1_u8(all, bd0 + k);
                svuint8_t b1 = svld1_u8(all, bd1 + k);
                svuint8_t b2 = svld1_u8(all, bd2 + k);
                svuint8_t b3 = svld1_u8(all, bd3 + k);
                if (M > 0)
                {
                    svuint8_t a0 = svld1_u8(all, ad0 + k);
                    ab00 = svdot_u32(ab00, a0, b0);
                    ab01 = svdot_u32(ab01, a0, b1);
                    ab02 = svdot_u32(ab02, a0, b2);
                    ab03 = svdot_u32(ab03, a0, b3);
                }
                if (M > 1)
                {
                    svuint8_t a1 = svld1_u8(all, ad1 + k);
                    ab10 = svdot_u32(ab10, a1, b0);
                    ab11 = svdot_u32(ab11, a1, b1);
                    ab12 = svdot_u32(ab12, a1, b2);
                    ab13 = svdot_u32(ab13, a1, b3);
                }
                if (M > 2)
                {
                    svuint8_t a2 = svld1_u8(all, ad2 + k);
                    ab20 = svdot_u32(ab20, a2, b0);
                    ab21 = svdot_u32(ab21, a2, b1);
                    ab22 = svdot_u32(ab22, a2, b2);
                    ab23 = svdot_u32(ab23, a2, b3);
                }
                if (M > 3)
                {
                    svuint8_t a3 = svld1_u8(all, ad3 + k);
                    ab30 = svdot_u32(ab30, a3, b0);
                    ab31 = svdot_u32(ab31, a3, b1);
                    ab32 = svdot_u32(ab32, a3, b2);
                    ab33 = svdot_u32(ab33, a3, b3);
                }
            }
            if (k < K)
            {
                svbool_t mask = svwhilelt_b8(k, K);
                svuint8_t b0 = svld1_u8(mask, bd0 + k);
                svuint8_t b1 = svld1_u8(mask, bd1 + k);
                svuint8_t b2 = svld1_u8(mask, bd2 + k);
                svuint8_t b3 = svld1_u8(mask, bd3 + k);
                if (M > 0)
                {
                    svuint8_t a0 = svld1_u8(mask, ad0 + k);
                    ab00 = svdot_u32(ab00, a0, b0);
                    ab01 = svdot_u32(ab01, a0, b1);
                    ab02 = svdot_u32(ab02, a0, b2);
                    ab03 = svdot_u32(ab03, a0, b3);
                }
                if (M > 1)
                {
                    svuint8_t a1 = svld1_u8(mask, ad1 + k);
                    ab10 = svdot_u32(ab10, a1, b0);
                    ab11 = svdot_u32(ab11, a1, b1);
                    ab12 = svdot_u32(ab12, a1, b2);
                    ab13 = svdot_u32(ab13, a1, b3);
                }
                if (M > 2)
                {
                    svuint8_t a2 = svld1_u8(mask, ad2 + k);
                    ab20 = svdot_u32(ab20, a2, b0);
                    ab21 = svdot_u32(ab21, a2, b1);
                    ab22 = svdot_u32(ab22, a2, b2);
                    ab23 = svdot_u32(ab23, a2, b3);
                }
                if (M > 3)
                {
                    svuint8_t a3 = svld1_u8(mask, ad3 + k);
                    ab30 = svdot_u32(ab30, a3, b0);
                    ab31 = svdot_u32(ab31, a3, b1);
                    ab32 = svdot_u32(ab32, a3, b2);
                    ab33 = svdot_u32(ab33, a3, b3);
                }
            }
            if (M > 0) DecodeCosineDistances1x4(an + 0 * 4, bn, bnStride, ab00, ab01, ab02, ab03, distances + 0 * stride);
            if (M > 1) DecodeCosineDistances1x4(an + 1 * 4, bn, bnStride, ab10, ab11, ab12, ab13, distances + 1 * stride);
            if (M > 2) DecodeCosineDistances1x4(an + 2 * 4, bn, bnStride, ab20, ab21, ab22, ab23, distances + 2 * stride);
            if (M > 3) DecodeCosineDistances1x4(an + 3 * 4, bn, bnStride, ab30, ab31, ab32, ab33, distances + 3 * stride);
        }

        template<int M> SIMD_INLINE void CorrelationMx1(size_t K, const uint8_t* ad, const float* an,
            const uint8_t* bd, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            const size_t width = svcntb();
            const size_t main = K & ~(width - 1);
            const svbool_t all = svptrue_b8();
            svuint32_t sums0 = svdup_n_u32(0), sums1 = sums0, sums2 = sums0, sums3 = sums0;
            size_t k = 0;
            for (; k < main; k += width)
            {
                svuint8_t b = svld1_u8(all, bd + k);
                if (M > 0) sums0 = svdot_u32(sums0, svld1_u8(all, ad + 0 * K + k), b);
                if (M > 1) sums1 = svdot_u32(sums1, svld1_u8(all, ad + 1 * K + k), b);
                if (M > 2) sums2 = svdot_u32(sums2, svld1_u8(all, ad + 2 * K + k), b);
                if (M > 3) sums3 = svdot_u32(sums3, svld1_u8(all, ad + 3 * K + k), b);
            }
            if (k < K)
            {
                svbool_t mask = svwhilelt_b8(k, K);
                svuint8_t b = svld1_u8(mask, bd + k);
                if (M > 0) sums0 = svdot_u32(sums0, svld1_u8(mask, ad + 0 * K + k), b);
                if (M > 1) sums1 = svdot_u32(sums1, svld1_u8(mask, ad + 1 * K + k), b);
                if (M > 2) sums2 = svdot_u32(sums2, svld1_u8(mask, ad + 2 * K + k), b);
                if (M > 3) sums3 = svdot_u32(sums3, svld1_u8(mask, ad + 3 * K + k), b);
            }
            if (M > 0) DecodeCosineDistance(an + 0 * 4, bn, bnStride, (uint32_t)svaddv_u32(svptrue_b32(), sums0), distances + 0 * stride);
            if (M > 1) DecodeCosineDistance(an + 1 * 4, bn, bnStride, (uint32_t)svaddv_u32(svptrue_b32(), sums1), distances + 1 * stride);
            if (M > 2) DecodeCosineDistance(an + 2 * 4, bn, bnStride, (uint32_t)svaddv_u32(svptrue_b32(), sums2), distances + 2 * stride);
            if (M > 3) DecodeCosineDistance(an + 3 * 4, bn, bnStride, (uint32_t)svaddv_u32(svptrue_b32(), sums3), distances + 3 * stride);
        }

        void MacroCorrelation(size_t M, size_t N, size_t K, const uint8_t* ad, const float* an,
            const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            size_t M4 = AlignLoAny(M, 4), N4 = AlignLo(N, 4), i = 0;
            for (; i < M4; i += 4)
            {
                size_t j = 0;
                for (; j < N4; j += 4)
                    CorrelationMx4<4>(K, ad + i * K, an + i * 4, bd + j * K, bn + j, N, distances + i * stride + j, stride);
                for (; j < N; ++j)
                    CorrelationMx1<4>(K, ad + i * K, an + i * 4, bd + j * K, bn + j, N, distances + i * stride + j, stride);
            }
            if (i < M)
            {
                size_t m = M - i;
                size_t j = 0;
                for (; j < N4; j += 4)
                {
                    if (m == 1)
                        CorrelationMx4<1>(K, ad + i * K, an + i * 4, bd + j * K, bn + j, N, distances + i * stride + j, stride);
                    else if (m == 2)
                        CorrelationMx4<2>(K, ad + i * K, an + i * 4, bd + j * K, bn + j, N, distances + i * stride + j, stride);
                    else
                        CorrelationMx4<3>(K, ad + i * K, an + i * 4, bd + j * K, bn + j, N, distances + i * stride + j, stride);
                }
                for (; j < N; ++j)
                {
                    if (m == 1)
                        CorrelationMx1<1>(K, ad + i * K, an + i * 4, bd + j * K, bn + j, N, distances + i * stride + j, stride);
                    else if (m == 2)
                        CorrelationMx1<2>(K, ad + i * K, an + i * 4, bd + j * K, bn + j, N, distances + i * stride + j, stride);
                    else
                        CorrelationMx1<3>(K, ad + i * K, an + i * 4, bd + j * K, bn + j, N, distances + i * stride + j, stride);
                }
            }
        }

        Base::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t)
        {
            return MacroCorrelation;
        }
    }
#endif
}

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
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdGather.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdMax.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        void SynetSoftmax16b21(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            Avx2::Exp exp;
            size_t outerF = Simd::AlignLo(outer, F);
            size_t o = 0;
            for (; o < outerF; o += F)
            {
                __m256i s01 = _mm256_loadu_si256((__m256i*)src);
                __m256 ss0 = BFloat16ToFloat32Even(s01);
                __m256 ss1 = BFloat16ToFloat32Odd(s01);
                __m256 max = _mm256_max_ps(ss0, ss1);
                __m256 exp0 = exp.Exponent(_mm256_sub_ps(ss0, max));
                __m256 exp1 = exp.Exponent(_mm256_sub_ps(ss1, max));
                __m256 sum = _mm256_add_ps(exp0, exp1);
                __m256 d0 = _mm256_div_ps(exp0, sum);
                __m256 d1 = _mm256_div_ps(exp1, sum);
                _mm256_storeu_si256((__m256i*)dst, Float32ToBFloat16Interlived(d0, d1));
                src += DF;
                dst += DF;
            }
            for (; o < outer; ++o)
            {
                float src0 = Base::BFloat16ToFloat32(src[0]);
                float src1 = Base::BFloat16ToFloat32(src[1]);
                float max = Simd::Max(src0, src1);
                float exp0 = ::exp(src0 - max);
                float exp1 = ::exp(src1 - max);
                float sum = exp0 + exp1;
                dst[0] = Base::Float32ToBFloat16(exp0 / sum);
                dst[1] = Base::Float32ToBFloat16(exp1 / sum);
                src += 2;
                dst += 2;
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetSoftmax16b31Load(const uint16_t* src, __m256 buf[3])
        {
            static const __m256i SFL00 = SIMD_MM256_SETR_EPI8(
                -1, -1, 0x0, 0x1, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, -1, -1,
                -1, -1, 0x8, 0x9, -1, -1, 0xE, 0xF, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m256i SFL01 = SIMD_MM256_SETR_EPI8(
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x2, 0x3,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x4, 0x5, -1, -1, 0xA, 0xB);
            static const __m256i SFL10 = SIMD_MM256_SETR_EPI8(
                -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xE, 0xF, -1, -1, -1, -1,
                -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m256i SFL11 = SIMD_MM256_SETR_EPI8(
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x4, 0x5,
                -1, -1, -1, -1, -1, -1, 0x0, 0x1, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD);
            static const __m256i SFL20 = SIMD_MM256_SETR_EPI8(
                -1, -1, 0x4, 0x5, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m256i SFL21 = SIMD_MM256_SETR_EPI8(
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x1, -1, -1, 0x6, 0x7,
                -1, -1, -1, -1, -1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xE, 0xF);
            __m256i s01 = _mm256_loadu_si256((__m256i*)(src + 0));
            __m256i s12 = _mm256_loadu_si256((__m256i*)(src + 8));
            buf[0] = _mm256_castsi256_ps(_mm256_or_si256(_mm256_shuffle_epi8(s01, SFL00), _mm256_shuffle_epi8(s12, SFL01)));
            buf[1] = _mm256_castsi256_ps(_mm256_or_si256(_mm256_shuffle_epi8(s01, SFL10), _mm256_shuffle_epi8(s12, SFL11)));
            buf[2] = _mm256_castsi256_ps(_mm256_or_si256(_mm256_shuffle_epi8(s01, SFL20), _mm256_shuffle_epi8(s12, SFL21)));
        }

        SIMD_INLINE void SynetSoftmax16b31(const Avx2::Exp& exp, __m256 buf[3])
        {
            __m256 max = _mm256_max_ps(buf[0], _mm256_max_ps(buf[1], buf[2]));
            buf[0] = exp.Exponent(_mm256_sub_ps(buf[0], max));
            buf[1] = exp.Exponent(_mm256_sub_ps(buf[1], max));
            buf[2] = exp.Exponent(_mm256_sub_ps(buf[2], max));
            __m256 sum = _mm256_add_ps(buf[0], _mm256_add_ps(buf[1], buf[2]));
            buf[0] = _mm256_div_ps(buf[0], sum);
            buf[1] = _mm256_div_ps(buf[1], sum);
            buf[2] = _mm256_div_ps(buf[2], sum);
        }

        SIMD_INLINE void SynetSoftmax16b31Load(const uint16_t* src, size_t size, __m256 dst[3])
        {
            SIMD_ALIGNED(32) uint16_t buf[A];
            for (size_t i = 0; i < size; i += 1)
            {
                buf[0 * 8 + i] = src[i * 3 + 0];
                buf[1 * 8 + i] = src[i * 3 + 1];
                buf[2 * 8 + i] = src[i * 3 + 2];
            }
            __m128i b01 = _mm_loadu_si128((__m128i*)buf);
            dst[0] = BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)buf + 0)));
            dst[1] = BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)buf + 1)));
            dst[2] = BFloat16ToFloat32(_mm256_cvtepu16_epi32(_mm_loadu_si128((__m128i*)buf + 2)));
        }

        SIMD_INLINE void SynetSoftmax16b31Save(const __m256 src[3], uint16_t* dst)
        {
            __m256i s01 = Float32ToBFloat16Interlived(src[0], src[1]);
            __m256i s2 = Float32ToBFloat16(src[2]);

            static const __m256i SFL020 = SIMD_MM256_SETR_EPI8(
                0x0, 0x1, 0x2, 0x3, -1, -1, 0x4, 0x5, 0x6, 0x7, -1, -1, 0x8, 0x9, 0xA, 0xB,
                0x6, 0x7, -1, -1, 0x8, 0x9, 0xA, 0xB, -1, -1, 0xC, 0xD, 0xE, 0xF, -1, -1);
            static const __m256i SFL021 = SIMD_MM256_SETR_EPI8(
                -1, -1, -1, -1, 0x0, 0x1, -1, -1, -1, -1, 0x4, 0x5, -1, -1, -1, -1,
                -1, -1, 0x4, 0x5, -1, -1, -1, -1, 0x8, 0x9, -1, -1, -1, -1, 0xC, 0xD);
            __m256i d02 = _mm256_or_si256(_mm256_shuffle_epi8(s01, SFL020), _mm256_shuffle_epi8(s2, SFL021));

            static const __m256i SFL10 = SIMD_MM256_SETR_EPI8(
                -1, -1, 0xC, 0xD, 0xE, 0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x1, 0x2, 0x3, -1, -1, 0x4, 0x5);
            static const __m256i SFL11 = SIMD_MM256_SETR_EPI8(
                0x8, 0x9, -1, -1, -1, -1, 0xC, 0xD, -1, -1, -1, -1, -1, -1, -1, -1,
                -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x0, 0x1, -1, -1);
            __m256i d1 = _mm256_or_si256(_mm256_shuffle_epi8(s01, SFL10), _mm256_shuffle_epi8(s2, SFL11));

            _mm_storeu_si128((__m128i*)dst + 0, _mm256_extractf128_si256(d02, 0));
            _mm_storeu_si128((__m128i*)dst + 1, _mm_or_si128(_mm256_extractf128_si256(d1, 0), _mm256_extractf128_si256(d1, 1)));
            _mm_storeu_si128((__m128i*)dst + 2, _mm256_extractf128_si256(d02, 1));
        }

        SIMD_INLINE void SynetSoftmax16b31Save(const __m256 src[3], size_t size, uint16_t* dst)
        {
            SIMD_ALIGNED(16) uint16_t buf[A];
            _mm256_storeu_si256((__m256i*)buf + 0, Float32ToBFloat16(src[0], src[1]));
            _mm256_storeu_si256((__m256i*)buf + 1, Float32ToBFloat16(src[2], src[2]));
            for (size_t i = 0; i < size; i += 1)
            {
                dst[i * 3 + 0] = buf[0 * 8 + i];
                dst[i * 3 + 1] = buf[1 * 8 + i];
                dst[i * 3 + 2] = buf[2 * 8 + i];
            }
        }

        void SynetSoftmax16b31(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            Avx2::Exp exp;
            __m256 buf[3];
            size_t aligned = Simd::AlignLo(outer, F);
            for (size_t o = 0; o < aligned; o += F)
            {
                SynetSoftmax16b31Load(src,  buf);
                SynetSoftmax16b31(exp, buf);
                SynetSoftmax16b31Save(buf, dst);
                src += 3 * F;
                dst += 3 * F;
            }
            if (aligned < outer)
            {
                size_t tail = outer - aligned;
                SynetSoftmax16b31Load(src, tail, buf);
                SynetSoftmax16b31(exp, buf);
                SynetSoftmax16b31Save(buf, tail, dst);
            }
        }

        //--------------------------------------------------------------------------------------------------

        SIMD_INLINE void LoadTansp8x8(const uint16_t* src, size_t count, float* dst, __m256& max)
        {
            __m256 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = BFloat16ToFloat32(_mm_loadu_si128((__m128i*)(src + 0 * count)));
            a1 = BFloat16ToFloat32(_mm_loadu_si128((__m128i*)(src + 1 * count)));
            a2 = BFloat16ToFloat32(_mm_loadu_si128((__m128i*)(src + 2 * count)));
            a3 = BFloat16ToFloat32(_mm_loadu_si128((__m128i*)(src + 3 * count)));
            a4 = BFloat16ToFloat32(_mm_loadu_si128((__m128i*)(src + 4 * count)));
            a5 = BFloat16ToFloat32(_mm_loadu_si128((__m128i*)(src + 5 * count)));
            a6 = BFloat16ToFloat32(_mm_loadu_si128((__m128i*)(src + 6 * count)));
            a7 = BFloat16ToFloat32(_mm_loadu_si128((__m128i*)(src + 7 * count)));

            b0 = _mm256_unpacklo_ps(a0, a2);
            b1 = _mm256_unpacklo_ps(a1, a3);
            b2 = _mm256_unpackhi_ps(a0, a2);
            b3 = _mm256_unpackhi_ps(a1, a3);
            b4 = _mm256_unpacklo_ps(a4, a6);
            b5 = _mm256_unpacklo_ps(a5, a7);
            b6 = _mm256_unpackhi_ps(a4, a6);
            b7 = _mm256_unpackhi_ps(a5, a7);

            a0 = _mm256_unpacklo_ps(b0, b1);
            a1 = _mm256_unpackhi_ps(b0, b1);
            a2 = _mm256_unpacklo_ps(b2, b3);
            a3 = _mm256_unpackhi_ps(b2, b3);
            a4 = _mm256_unpacklo_ps(b4, b5);
            a5 = _mm256_unpackhi_ps(b4, b5);
            a6 = _mm256_unpacklo_ps(b6, b7);
            a7 = _mm256_unpackhi_ps(b6, b7);

            b0 = _mm256_permute2f128_ps(a0, a4, 0x20);
            b1 = _mm256_permute2f128_ps(a1, a5, 0x20);
            b2 = _mm256_permute2f128_ps(a2, a6, 0x20);
            b3 = _mm256_permute2f128_ps(a3, a7, 0x20);
            b4 = _mm256_permute2f128_ps(a0, a4, 0x31);
            b5 = _mm256_permute2f128_ps(a1, a5, 0x31);
            b6 = _mm256_permute2f128_ps(a2, a6, 0x31);
            b7 = _mm256_permute2f128_ps(a3, a7, 0x31);

            max = _mm256_max_ps(max, b0);
            max = _mm256_max_ps(max, b1);
            max = _mm256_max_ps(max, b2);
            max = _mm256_max_ps(max, b3);
            max = _mm256_max_ps(max, b4);
            max = _mm256_max_ps(max, b5);
            max = _mm256_max_ps(max, b6);
            max = _mm256_max_ps(max, b7);

            _mm256_storeu_ps(dst + 0 * F, b0);
            _mm256_storeu_ps(dst + 1 * F, b1);
            _mm256_storeu_ps(dst + 2 * F, b2);
            _mm256_storeu_ps(dst + 3 * F, b3);
            _mm256_storeu_ps(dst + 4 * F, b4);
            _mm256_storeu_ps(dst + 5 * F, b5);
            _mm256_storeu_ps(dst + 6 * F, b6);
            _mm256_storeu_ps(dst + 7 * F, b7);
        }

        SIMD_INLINE void StoreTansp8x8(const float* src, __m256 k, uint16_t* dst, size_t count)
        {
            __m256 a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7;

            a0 = _mm256_mul_ps(_mm256_loadu_ps(src + 0 * F), k);
            a1 = _mm256_mul_ps(_mm256_loadu_ps(src + 1 * F), k);
            a2 = _mm256_mul_ps(_mm256_loadu_ps(src + 2 * F), k);
            a3 = _mm256_mul_ps(_mm256_loadu_ps(src + 3 * F), k);
            a4 = _mm256_mul_ps(_mm256_loadu_ps(src + 4 * F), k);
            a5 = _mm256_mul_ps(_mm256_loadu_ps(src + 5 * F), k);
            a6 = _mm256_mul_ps(_mm256_loadu_ps(src + 6 * F), k);
            a7 = _mm256_mul_ps(_mm256_loadu_ps(src + 7 * F), k);

            b0 = _mm256_unpacklo_ps(a0, a2);
            b1 = _mm256_unpacklo_ps(a1, a3);
            b2 = _mm256_unpackhi_ps(a0, a2);
            b3 = _mm256_unpackhi_ps(a1, a3);
            b4 = _mm256_unpacklo_ps(a4, a6);
            b5 = _mm256_unpacklo_ps(a5, a7);
            b6 = _mm256_unpackhi_ps(a4, a6);
            b7 = _mm256_unpackhi_ps(a5, a7);

            a0 = _mm256_unpacklo_ps(b0, b1);
            a1 = _mm256_unpackhi_ps(b0, b1);
            a2 = _mm256_unpacklo_ps(b2, b3);
            a3 = _mm256_unpackhi_ps(b2, b3);
            a4 = _mm256_unpacklo_ps(b4, b5);
            a5 = _mm256_unpackhi_ps(b4, b5);
            a6 = _mm256_unpacklo_ps(b6, b7);
            a7 = _mm256_unpackhi_ps(b6, b7);

            b0 = _mm256_permute2f128_ps(a0, a4, 0x20);
            b1 = _mm256_permute2f128_ps(a1, a5, 0x20);
            b2 = _mm256_permute2f128_ps(a2, a6, 0x20);
            b3 = _mm256_permute2f128_ps(a3, a7, 0x20);
            b4 = _mm256_permute2f128_ps(a0, a4, 0x31);
            b5 = _mm256_permute2f128_ps(a1, a5, 0x31);
            b6 = _mm256_permute2f128_ps(a2, a6, 0x31);
            b7 = _mm256_permute2f128_ps(a3, a7, 0x31);

            _mm_storeu_si128((__m128i*)(dst + 0 * count), PackFloat32ToBFloat16(b0));
            _mm_storeu_si128((__m128i*)(dst + 1 * count), PackFloat32ToBFloat16(b1));
            _mm_storeu_si128((__m128i*)(dst + 2 * count), PackFloat32ToBFloat16(b2));
            _mm_storeu_si128((__m128i*)(dst + 3 * count), PackFloat32ToBFloat16(b3));
            _mm_storeu_si128((__m128i*)(dst + 4 * count), PackFloat32ToBFloat16(b4));
            _mm_storeu_si128((__m128i*)(dst + 5 * count), PackFloat32ToBFloat16(b5));
            _mm_storeu_si128((__m128i*)(dst + 6 * count), PackFloat32ToBFloat16(b6));
            _mm_storeu_si128((__m128i*)(dst + 7 * count), PackFloat32ToBFloat16(b7));
        }

        void SynetSoftmax16bX1(const uint16_t* src, size_t outer, size_t count, uint16_t* dst)
        {
            size_t o = 0, c = 0, outerF = AlignLo(outer, F), countF = AlignLo(count, F);
            Array32f buf(AlignHi(count, F) * F);
            Exp exp;
            for (; o < outerF; o += F)
            {
                __m256 _max = _mm256_set1_ps(-FLT_MAX);
                for (c = 0; c < countF; c += F)
                    LoadTansp8x8(src + c, count, buf.data + c * F, _max);
                if (c < count)
                {
                    c = count - F;
                    LoadTansp8x8(src + c, count, buf.data + c * F, _max);
                }
                __m256 _sum = _mm256_setzero_ps();
                for (size_t c = 0; c < count; ++c)
                {
                    __m256 _exp = exp.Exponent(_mm256_sub_ps(_mm256_loadu_ps(buf.data + c * F), _max));
                    _sum = _mm256_add_ps(_sum, _exp);
                    _mm256_storeu_ps(buf.data + c * F, _exp);
                }
                __m256 _k = _mm256_div_ps(_mm256_set1_ps(1.0f), _sum);
                for (c = 0; c < countF; c += F)
                    StoreTansp8x8(buf.data + c * F, _k, dst + c, count);
                if (c < count)
                {
                    c = count - F;
                    StoreTansp8x8(buf.data + c * F, _k, dst + c, count);
                }
                src += count * F;
                dst += count * F;
            }
            for (; o < outer; ++o)
            {
                for (size_t c = 0; c < count; ++c)
                    buf[c] = Base::BFloat16ToFloat32(src[c]);

                float max = buf[0];
                for (size_t c = 1; c < count; ++c)
                    max = Simd::Max(max, buf[c]);
                float sum = 0;
                for (size_t c = 0; c < count; ++c)
                {
                    buf[c] = ::exp(buf[c] - max);
                    sum += buf[c];
                }
                float k = 1.0f / sum;
                for (size_t c = 0; c < count; ++c)
                    dst[c] = Base::Float32ToBFloat16(buf[c] * k);
                src += count;
                dst += count;
            }
        }

        void SynetSoftmax16b(const uint16_t* src, size_t outer, size_t count, size_t inner, uint16_t* dst)
        {
            if (inner == 1)
            {
                if (count == 2)
                    SynetSoftmax16b21(src, outer, dst);
                else if (count == 3)
                    SynetSoftmax16b31(src, outer, dst);
                else if(count < 8)
                    Sse41::SynetSoftmax16bX1(src, outer, count, dst);
                else
                    SynetSoftmax16bX1(src, outer, count, dst);
            }
            else
            {
                Avx2::Exp exp;
                size_t innerF = Simd::AlignLo(inner, F), innerDF = Simd::AlignLo(inner, DF);
                Array32f _buf(inner * (count + 2));
                float* max = _buf.data, * sum = _buf.data + inner, * buf = sum + inner, * b;
                for (size_t o = 0; o < outer; ++o)
                {
                    BFloat16ToFloat32(src, count * inner, buf);
                    memcpy(max, buf, inner * sizeof(float));
                    b = buf + inner;
                    for (size_t c = 1; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerF; i += F)
                            _mm256_storeu_ps(max + i, _mm256_max_ps(_mm256_loadu_ps(b + i), _mm256_loadu_ps(max + i)));
                        for (; i < inner; ++i)
                            max[i] = Simd::Max(max[i], b[i]);
                        b += inner;
                    }

                    b = buf;
                    memset(sum, 0, inner * sizeof(float));
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerF; i += F)
                        {
                            __m256 _d = exp.Exponent(_mm256_sub_ps(_mm256_loadu_ps(b + i), _mm256_loadu_ps(max + i)));
                            _mm256_storeu_ps(b + i, _d);
                            _mm256_storeu_ps(sum + i, _mm256_add_ps(_d, _mm256_loadu_ps(sum + i)));
                        }
                        for (; i < inner; ++i)
                        {
                            b[i] = ::exp(b[i] - max[i]);
                            sum[i] += b[i];
                        }
                        b += inner;
                    }

                    b = buf;
                    for (size_t c = 0; c < count; ++c)
                    {
                        size_t i = 0;
                        for (; i < innerDF; i += DF)
                            _mm256_storeu_si256((__m256i*)(dst + i), Float32ToBFloat16(
                                _mm256_div_ps(_mm256_loadu_ps(b + i + 0), _mm256_loadu_ps(sum + i + 0)), 
                                _mm256_div_ps(_mm256_loadu_ps(b + i + F), _mm256_loadu_ps(sum + i + F))));
                        for (; i < innerF; i += F)
                            StoreHalf<false, 0>((__m128i*)(dst + i), PackU32ToI16(Float32ToBFloat16(
                                _mm256_div_ps(_mm256_loadu_ps(b + i), _mm256_loadu_ps(sum + i))), K_ZERO));
                        for (; i < inner; ++i)
                            dst[i] = Base::Float32ToBFloat16(b[i] / sum[i]);
                        b += inner;
                        dst += inner;
                    }
                    src += count * inner;
                }
            }
        }
    }
#endif
}

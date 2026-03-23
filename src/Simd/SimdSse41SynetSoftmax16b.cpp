/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdGather.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdMax.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        void SynetSoftmax16b21(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            Exp exp;
            size_t aligned = Simd::AlignLo(outer, F);
            size_t o = 0;
            for (; o < aligned; o += F)
            {
                __m128i s01 = _mm_loadu_si128((__m128i*)src);
                __m128 ss0 = BFloat16ToFloat32Even(s01);
                __m128 ss1 = BFloat16ToFloat32Odd(s01);
                __m128 max = _mm_max_ps(ss0, ss1);
                __m128 exp0 = exp.Exponent(_mm_sub_ps(ss0, max));
                __m128 exp1 = exp.Exponent(_mm_sub_ps(ss1, max));
                __m128 sum = _mm_add_ps(exp0, exp1);
                __m128 d0 = _mm_div_ps(exp0, sum);
                __m128 d1 = _mm_div_ps(exp1, sum);
                _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16Interlived(d0, d1));
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

        SIMD_INLINE void SynetSoftmax16b31Load(const uint16_t* src, __m128 buf[3])
        {
            static const __m128i SFL00 = SIMD_MM_SETR_EPI8(-1, -1, 0x0, 0x1, -1, -1, 0x6, 0x7, -1, -1, 0xC, 0xD, -1, -1, -1, -1);
            static const __m128i SFL01 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xA, 0xB);
            static const __m128i SFL10 = SIMD_MM_SETR_EPI8(-1, -1, 0x2, 0x3, -1, -1, 0x8, 0x9, -1, -1, 0xE, 0xF, -1, -1, -1, -1);
            static const __m128i SFL11 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0xC, 0xD);
            static const __m128i SFL20 = SIMD_MM_SETR_EPI8(-1, -1, 0x4, 0x5, -1, -1, 0xA, 0xB, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m128i SFL21 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0x8, 0x9, -1, -1, 0xE, 0xF);
            __m128i s01 = _mm_loadu_si128((__m128i*)(src + 0));
            __m128i s12 = _mm_loadu_si128((__m128i*)(src + 4));
            buf[0] = _mm_castsi128_ps(_mm_or_si128(_mm_shuffle_epi8(s01, SFL00), _mm_shuffle_epi8(s12, SFL01)));
            buf[1] = _mm_castsi128_ps(_mm_or_si128(_mm_shuffle_epi8(s01, SFL10), _mm_shuffle_epi8(s12, SFL11)));
            buf[2] = _mm_castsi128_ps(_mm_or_si128(_mm_shuffle_epi8(s01, SFL20), _mm_shuffle_epi8(s12, SFL21)));
        }

        SIMD_INLINE void SynetSoftmax16b31Load(const uint16_t* src, size_t size, __m128 dst[3])
        {
            SIMD_ALIGNED(16) uint16_t buf[A];
            for (size_t i = 0; i < size; i += 1)
            {
                buf[0 * 4 + i] = src[i * 3 + 0];
                buf[1 * 4 + i] = src[i * 3 + 1];
                buf[2 * 4 + i] = src[i * 3 + 2];
            }
            __m128i b01 = _mm_loadu_si128((__m128i*)buf);
            dst[0] = BFloat16ToFloat32<0>(b01);
            dst[1] = BFloat16ToFloat32<1>(b01);
            __m128i b2 = _mm_loadu_si128((__m128i*)buf + 1);
            dst[2] = BFloat16ToFloat32<0>(b2);
        }

        SIMD_INLINE void SynetSoftmax16b31(const Exp& exp, __m128 buf[3])
        {
            __m128 max = _mm_max_ps(buf[0], _mm_max_ps(buf[1], buf[2]));
            buf[0] = exp.Exponent(_mm_sub_ps(buf[0], max));
            buf[1] = exp.Exponent(_mm_sub_ps(buf[1], max));
            buf[2] = exp.Exponent(_mm_sub_ps(buf[2], max));
            __m128 sum = _mm_add_ps(buf[0], _mm_add_ps(buf[1], buf[2]));
            buf[0] = _mm_div_ps(buf[0], sum);
            buf[1] = _mm_div_ps(buf[1], sum);
            buf[2] = _mm_div_ps(buf[2], sum);
        }

        SIMD_INLINE void SynetSoftmax16b31Save(const __m128 src[3], uint16_t* dst)
        {
            __m128i s01 = Float32ToBFloat16(src[0], src[1]);
            __m128i s12 = Float32ToBFloat16(src[1], src[2]);
            static const __m128i SFL00 = SIMD_MM_SETR_EPI8(0x0, 0x1, 0x8, 0x9, -1, -1, 0x2, 0x3, 0xA, 0xB, -1, -1, 0x4, 0x5, 0xC, 0xD);
            static const __m128i SFL01 = SIMD_MM_SETR_EPI8(-1, -1, -1, -1, 0x8, 0x9, -1, -1, -1, -1, 0xA, 0xB, -1, -1, -1, -1);
            static const __m128i SFL10 = SIMD_MM_SETR_EPI8(-1, -1, 0x6, 0x7, 0xE, 0xF, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
            static const __m128i SFL11 = SIMD_MM_SETR_EPI8(0xC, 0xD, -1, -1, -1, -1, 0xE, 0xF, -1, -1, -1, -1, -1, -1, -1, -1);
            _mm_storeu_si128((__m128i*)(dst + 0), _mm_or_si128(_mm_shuffle_epi8(s01, SFL00), _mm_shuffle_epi8(s12, SFL01)));
            _mm_storel_epi64((__m128i*)(dst + 8), _mm_or_si128(_mm_shuffle_epi8(s01, SFL10), _mm_shuffle_epi8(s12, SFL11)));
        }

        SIMD_INLINE void SynetSoftmax16b31Save(const __m128 src[3], size_t size, uint16_t *dst)
        {
            SIMD_ALIGNED(16) uint16_t buf[A];
            _mm_storeu_si128((__m128i*)buf + 0, Float32ToBFloat16(src[0], src[1]));
            _mm_storeu_si128((__m128i*)buf + 1, Float32ToBFloat16(src[2], _mm_setzero_ps()));
            for (size_t i = 0; i < size; i += 1)
            {
                dst[i * 3 + 0] = buf[0 * 4 + i];
                dst[i * 3 + 1] = buf[1 * 4 + i];
                dst[i * 3 + 2] = buf[2 * 4 + i];
            }
        }

        void SynetSoftmax16b31(const uint16_t* src, size_t outer, uint16_t* dst)
        {
            Exp exp;
            __m128 buf[3];
            size_t aligned = Simd::AlignLo(outer, F);
            for (size_t o = 0; o < aligned; o += F)
            {
                SynetSoftmax16b31Load(src, buf);
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

        SIMD_INLINE void LoadTansp4x4(const uint16_t* src, size_t count, float* dst, __m128& max)
        {
            __m128 a0 = BFloat16ToFloat32<0>(_mm_loadl_epi64((__m128i*)(src + 0 * count)));
            __m128 a1 = BFloat16ToFloat32<0>(_mm_loadl_epi64((__m128i*)(src + 1 * count)));
            __m128 a2 = BFloat16ToFloat32<0>(_mm_loadl_epi64((__m128i*)(src + 2 * count)));
            __m128 a3 = BFloat16ToFloat32<0>(_mm_loadl_epi64((__m128i*)(src + 3 * count)));
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpacklo_ps(a1, a3);
            __m128 b2 = _mm_unpackhi_ps(a0, a2);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            a0 = _mm_unpacklo_ps(b0, b1);
            max = _mm_max_ps(max, a0);
            _mm_storeu_ps(dst + 0 * F, a0);
            a1 = _mm_unpackhi_ps(b0, b1);
            max = _mm_max_ps(max, a1);
            _mm_storeu_ps(dst + 1 * F, a1);
            a2 = _mm_unpacklo_ps(b2, b3);
            max = _mm_max_ps(max, a2);
            _mm_storeu_ps(dst + 2 * F, a2);
            a3 = _mm_unpackhi_ps(b2, b3);
            max = _mm_max_ps(max, a3);
            _mm_storeu_ps(dst + 3 * F, a3);
        }

        SIMD_INLINE void StoreTansp4x4(const float* src, __m128 k, uint16_t* dst, size_t count)
        {
            __m128 a0 = _mm_mul_ps(_mm_loadu_ps(src + 0 * F), k);
            __m128 a1 = _mm_mul_ps(_mm_loadu_ps(src + 1 * F), k);
            __m128 a2 = _mm_mul_ps(_mm_loadu_ps(src + 2 * F), k);
            __m128 a3 = _mm_mul_ps(_mm_loadu_ps(src + 3 * F), k);
            __m128 b0 = _mm_unpacklo_ps(a0, a2);
            __m128 b1 = _mm_unpacklo_ps(a1, a3);
            __m128 b2 = _mm_unpackhi_ps(a0, a2);
            __m128 b3 = _mm_unpackhi_ps(a1, a3);
            __m128i d01 = Float32ToBFloat16(_mm_unpacklo_ps(b0, b1), _mm_unpackhi_ps(b0, b1));
            StoreHalf<0>((__m128i*)(dst + 0 * count), d01);
            StoreHalf<1>((__m128i*)(dst + 1 * count), d01);
            __m128i d23 = Float32ToBFloat16(_mm_unpacklo_ps(b2, b3), _mm_unpackhi_ps(b2, b3));
            StoreHalf<0>((__m128i*)(dst + 2 * count), d23);
            StoreHalf<1>((__m128i*)(dst + 3 * count), d23);
        }

        void SynetSoftmax16bX1(const uint16_t* src, size_t outer, size_t count, uint16_t* dst)
        {
            size_t o = 0, c = 0, outerF = AlignLo(outer, F), countF = AlignLo(count, F);
            Array32f buf(AlignHi(count, F) * F);
            Exp exp;
            for (; o < outerF; o += F)
            {
                __m128 _max = _mm_set1_ps(-FLT_MAX);
                for (c = 0; c < countF; c += F)
                    LoadTansp4x4(src + c, count, buf.data + c * F, _max);
                if (c < count)
                {
                    c = count - F;
                    LoadTansp4x4(src + c, count, buf.data + c * F, _max);
                }
                __m128 _sum = _mm_setzero_ps();
                for (size_t c = 0; c < count; ++c)
                {
                    __m128 _exp = exp.Exponent(_mm_sub_ps(_mm_loadu_ps(buf.data + c * F), _max));
                    _sum = _mm_add_ps(_sum, _exp);
                    _mm_storeu_ps(buf.data + c * F, _exp);
                }
                __m128 _k = _mm_div_ps(_mm_set1_ps(1.0f), _sum);
                for (c = 0; c < countF; c += F)
                    StoreTansp4x4(buf.data + c * F, _k, dst + c, count);
                if (c < count)
                {
                    c = count - F;
                    StoreTansp4x4(buf.data + c * F, _k, dst + c, count);
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
                else
                    SynetSoftmax16bX1(src, outer, count, dst);
            }
            else
            {
                Exp exp;
                size_t innerF = Simd::AlignLo(inner, F);
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
                            _mm_storeu_ps(max + i, _mm_max_ps(_mm_loadu_ps(b + i), _mm_loadu_ps(max + i)));
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
                            __m128 _d = exp.Exponent(_mm_sub_ps(_mm_loadu_ps(b + i), _mm_loadu_ps(max + i)));
                            _mm_storeu_ps(b + i, _d);
                            _mm_storeu_ps(sum + i, _mm_add_ps(_d, _mm_loadu_ps(sum + i)));
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
                        for (; i < innerF; i += F)
                            _mm_storel_epi64((__m128i*)(dst + i), _mm_packus_epi32(Float32ToBFloat16(
                                _mm_div_ps(_mm_loadu_ps(b + i), _mm_loadu_ps(sum + i))), K_ZERO));
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

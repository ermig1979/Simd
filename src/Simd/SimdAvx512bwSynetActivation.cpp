/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#include "Simd/SimdStore.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512bw
    {
        template<bool mask> SIMD_INLINE void SynetElu32f(const float * src, const Exp & exp, __m512 alpha, float * dst, size_t offset, __mmask16 tail = -1)
        {
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, exp.Elu((mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset)), alpha));
            else
                _mm512_storeu_ps(dst + offset, exp.Elu((mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset)), alpha));
        }

        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            __m512 _alpha = _mm512_set1_ps(alpha[0]);
            Exp exp;
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetElu32f<false>(src, exp, _alpha, dst, i + 0 * F);
                SynetElu32f<false>(src, exp, _alpha, dst, i + 1 * F);
                SynetElu32f<false>(src, exp, _alpha, dst, i + 2 * F);
                SynetElu32f<false>(src, exp, _alpha, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetElu32f<false>(src, exp, _alpha, dst, i);
            if(i < size)
                SynetElu32f<true>(src, exp, _alpha, dst, i, tail);

        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetGelu32f(const float* src, float* dst, size_t offset, __mmask16 tail = -1)
        {
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, Gelu((mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset))));
            else
                _mm512_storeu_ps(dst + offset, Gelu((mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset))));
        }

        void SynetGelu32f(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetGelu32f<false>(src, dst, i + 0 * F);
                SynetGelu32f<false>(src, dst, i + 1 * F);
                SynetGelu32f<false>(src, dst, i + 2 * F);
                SynetGelu32f<false>(src, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetGelu32f<false>(src, dst, i);
            if (i < size)
                SynetGelu32f<true>(src, dst, i, tail);

        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetHardSigmoid32f(const float* src, __m512 scale, __m512 shift, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _dst = SynetHardSigmoid32f(_src, scale, shift);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _dst);
            else
                _mm512_storeu_ps(dst + offset, _dst);
        }

        void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            __m512 _scale = _mm512_set1_ps(scale[0]);
            __m512 _shift = _mm512_set1_ps(shift[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHardSigmoid32f<false>(src, _scale, _shift, dst, i + 0 * F);
                SynetHardSigmoid32f<false>(src, _scale, _shift, dst, i + 1 * F);
                SynetHardSigmoid32f<false>(src, _scale, _shift, dst, i + 2 * F);
                SynetHardSigmoid32f<false>(src, _scale, _shift, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHardSigmoid32f<false>(src, _scale, _shift, dst, i);
            if (i < size)
                SynetHardSigmoid32f<true>(src, _scale, _shift, dst, i, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetHswish32f(const float * src, __m512 shift, __m512 scale, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _dst = _mm512_mul_ps(_mm512_mul_ps(_mm512_max_ps(_mm512_add_ps(_mm512_min_ps(_src, shift), shift), _mm512_setzero_ps()), scale), _src);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _dst);
            else
                _mm512_storeu_ps(dst + offset, _dst);
        }

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            __m512 _shift = _mm512_set1_ps(shift[0]);
            __m512 _scale = _mm512_set1_ps(scale[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHswish32f<false>(src, _shift, _scale, dst, i + 0 * F);
                SynetHswish32f<false>(src, _shift, _scale, dst, i + 1 * F);
                SynetHswish32f<false>(src, _shift, _scale, dst, i + 2 * F);
                SynetHswish32f<false>(src, _shift, _scale, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHswish32f<false>(src, _shift, _scale, dst, i);
            if (i < size)
                SynetHswish32f<true>(src, _shift, _scale, dst, i, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetMish32f(const float* src, __m512 threshold, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, Mish(_src, threshold));
            else
                _mm512_storeu_ps(dst + offset, Mish(_src, threshold));
        }

        void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            __m512 _threshold = _mm512_set1_ps(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetMish32f<false>(src, _threshold, dst, i + 0 * F);
                SynetMish32f<false>(src, _threshold, dst, i + 1 * F);
                SynetMish32f<false>(src, _threshold, dst, i + 2 * F);
                SynetMish32f<false>(src, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetMish32f<false>(src, _threshold, dst, i);
            if (i < size)
                SynetMish32f<true>(src, _threshold, dst, i, tail);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool mask> SIMD_INLINE void SynetPreluLayerForward(const float* src, const float* slope, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _slope = (mask ? _mm512_maskz_loadu_ps(tail, slope + offset) : _mm512_loadu_ps(slope + offset));
            __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), _src);
            __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), _src);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _mm512_add_ps(pos, _mm512_mul_ps(_slope, neg)));
            else
                _mm512_storeu_ps(dst + offset, _mm512_add_ps(pos, _mm512_mul_ps(_slope, neg)));
        }

        template <bool mask> SIMD_INLINE void SynetPreluLayerForward(const float* src, __m512 slope, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), _src);
            __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), _src);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _mm512_add_ps(pos, _mm512_mul_ps(slope, neg)));
            else
                _mm512_storeu_ps(dst + offset, _mm512_add_ps(pos, _mm512_mul_ps(slope, neg)));
        }

        void SynetPreluLayerForward(const float* src, const float* slope, size_t count, size_t size, float* dst, SimdBool trans)
        {
            if ((trans || size == 1) && count != 1)
            {
                size_t aligned = AlignLo(count, QF);
                size_t partial = AlignLo(count, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - count);
                for (size_t j = 0; j < size; ++j)
                {
                    size_t i = 0;
                    for (; i < aligned; i += QF)
                    {
                        SynetPreluLayerForward<false>(src, slope, dst, i + F * 0);
                        SynetPreluLayerForward<false>(src, slope, dst, i + F * 1);
                        SynetPreluLayerForward<false>(src, slope, dst, i + F * 2);
                        SynetPreluLayerForward<false>(src, slope, dst, i + F * 3);
                    }
                    for (; i < partial; i += F)
                        SynetPreluLayerForward<false>(src, slope, dst, i);
                    if (i < count)
                        SynetPreluLayerForward<true>(src, slope, dst, i, tail);
                    src += count;
                    dst += count;
                }
            }
            else
            {
                size_t aligned = AlignLo(size, QF);
                size_t partial = AlignLo(size, F);
                __mmask16 tail = __mmask16(-1) >> (F + partial - size);
                for (size_t i = 0; i < count; ++i)
                {
                    size_t j = 0;
                    __m512 _slope = _mm512_set1_ps(slope[i]);
                    for (; j < aligned; j += QF)
                    {
                        SynetPreluLayerForward<false>(src, _slope, dst, j + F * 0);
                        SynetPreluLayerForward<false>(src, _slope, dst, j + F * 1);
                        SynetPreluLayerForward<false>(src, _slope, dst, j + F * 2);
                        SynetPreluLayerForward<false>(src, _slope, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetPreluLayerForward<false>(src, _slope, dst, j);
                    if (i < count)
                        SynetPreluLayerForward<true>(src, _slope, dst, j, tail);
                    src += size;
                    dst += size;
                }
            }
        }

        void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            __mmask16 tail = TailMask16(spatial - partial);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                __m512 _slope = _mm512_set1_ps(slope[c]);
                for (; s < aligned; s += QF)
                {
                    SynetPreluLayerForward<false>(src, _slope, dst, s + F * 0);
                    SynetPreluLayerForward<false>(src, _slope, dst, s + F * 1);
                    SynetPreluLayerForward<false>(src, _slope, dst, s + F * 2);
                    SynetPreluLayerForward<false>(src, _slope, dst, s + F * 3);
                }
                for (; s < partial; s += F)
                    SynetPreluLayerForward<false>(src, _slope, dst, s);
                if (s < spatial)
                    SynetPreluLayerForward<true>(src, _slope, dst, s, tail);
                src += spatial;
                dst += spatial;
            }
        }

        void SynetPreluLayerForwardNhwc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            __mmask16 tail = TailMask16(channels - partial);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += QF)
                {
                    SynetPreluLayerForward<false>(src, slope, dst, c + F * 0);
                    SynetPreluLayerForward<false>(src, slope, dst, c + F * 1);
                    SynetPreluLayerForward<false>(src, slope, dst, c + F * 2);
                    SynetPreluLayerForward<false>(src, slope, dst, c + F * 3);
                }
                for (; c < partial; c += F)
                    SynetPreluLayerForward<false>(src, slope, dst, c);
                if (c < channels)
                    SynetPreluLayerForward<true>(src, slope, dst, c, tail);
                src += channels;
                dst += channels;
            }
        }

        void SynetPreluLayerForward(const float* src, const float* slope, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetPreluLayerForwardNchw(src, slope, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetPreluLayerForwardNhwc(src, slope, channels, spatial, dst);
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetRelu32f(const float* src, __m512 slope, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _dst = SynetRelu32f(_src, slope);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _dst);
            else
                _mm512_storeu_ps(dst + offset, _dst);
        }

        void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            __m512 _slope = _mm512_set1_ps(slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetRelu32f<false>(src, _slope, dst, i + 0 * F);
                SynetRelu32f<false>(src, _slope, dst, i + 1 * F);
                SynetRelu32f<false>(src, _slope, dst, i + 2 * F);
                SynetRelu32f<false>(src, _slope, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetRelu32f<false>(src, _slope, dst, i);
            if (i < size)
                SynetRelu32f<true>(src, _slope, dst, i, tail);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetRelu16b(const uint16_t* src, __m512 slope, uint16_t* dst, __mmask32 tail = -1)
        {
            __m512i _src = _mm512_maskz_loadu_epi16(tail, src);
            __m512 even = SynetRelu32f(BFloat16ToFloat32Even(_src), slope);
            __m512 odd = SynetRelu32f(BFloat16ToFloat32Odd(_src), slope);
            _mm512_mask_storeu_epi16(dst, tail, Float32ToBFloat16Interlived(even, odd));
        }

        void SynetRelu16b(const uint16_t* src, size_t size, const float* slope, uint16_t* dst)
        {
            __m512 _slope = _mm512_set1_ps(slope[0]);
            size_t sizeDF = AlignLo(size, DF);
            __mmask32 tail = TailMask32(size - sizeDF);

            size_t i = 0;
            for (; i < sizeDF; i += DF)
                SynetRelu16b(src + i, _slope, dst + i);
            if(tail)
                SynetRelu16b(src + i, _slope, dst + i, tail);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(lower[0] <= upper[0]);
            float min = *lower;
            float max = *upper;
            __m512 _min = _mm512_set1_ps(min);
            __m512 _max = _mm512_set1_ps(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                _mm512_storeu_ps(dst + i + 0 * F, _mm512_min_ps(_mm512_max_ps(_min, _mm512_loadu_ps(src + i + 0 * F)), _max));
                _mm512_storeu_ps(dst + i + 1 * F, _mm512_min_ps(_mm512_max_ps(_min, _mm512_loadu_ps(src + i + 1 * F)), _max));
                _mm512_storeu_ps(dst + i + 2 * F, _mm512_min_ps(_mm512_max_ps(_min, _mm512_loadu_ps(src + i + 2 * F)), _max));
                _mm512_storeu_ps(dst + i + 3 * F, _mm512_min_ps(_mm512_max_ps(_min, _mm512_loadu_ps(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                _mm512_storeu_ps(dst + i, _mm512_min_ps(_mm512_max_ps(_min, _mm512_loadu_ps(src + i)), _max));
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                _mm512_mask_storeu_ps(dst + i, tail, _mm512_min_ps(_mm512_max_ps(_min, _mm512_maskz_loadu_ps(tail, src + i)), _max));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetSigmoid32f(const float* src, const Exp& exp, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _dst = exp.Sigmoid(_src);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _dst);
            else
                _mm512_storeu_ps(dst + offset, _dst);
        }

        void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSigmoid32f<false>(src, exp, dst, i + 0 * F);
                SynetSigmoid32f<false>(src, exp, dst, i + 1 * F);
                SynetSigmoid32f<false>(src, exp, dst, i + 2 * F);
                SynetSigmoid32f<false>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSigmoid32f<false>(src, exp, dst, i);
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                SynetSigmoid32f<true>(src, exp, dst, i, tail);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetSoftplus32f(const float* src, __m512 beta, __m512 threshold, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _dst = Softplus(_src, beta, threshold);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _dst);
            else
                _mm512_storeu_ps(dst + offset, _dst);
        }

        void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            __m512 _beta = _mm512_set1_ps(beta[0]);
            __m512 _threshold = _mm512_set1_ps(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSoftplus32f<false>(src, _beta, _threshold, dst, i + 0 * F);
                SynetSoftplus32f<false>(src, _beta, _threshold, dst, i + 1 * F);
                SynetSoftplus32f<false>(src, _beta, _threshold, dst, i + 2 * F);
                SynetSoftplus32f<false>(src, _beta, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSoftplus32f<false>(src, _beta, _threshold, dst, i);
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                SynetSoftplus32f<true>(src, _beta, _threshold, dst, i, tail);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetSwish32f(const float* src, const Exp& exp, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _dst = exp.Swish(_src);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _dst);
            else
                _mm512_storeu_ps(dst + offset, _dst);
        }

        void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSwish32f<false>(src, exp, dst, i + 0 * F);
                SynetSwish32f<false>(src, exp, dst, i + 1 * F);
                SynetSwish32f<false>(src, exp, dst, i + 2 * F);
                SynetSwish32f<false>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSwish32f<false>(src, exp, dst, i);
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                SynetSwish32f<true>(src, exp, dst, i, tail);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<bool mask> SIMD_INLINE void SynetTanh32f(const float* src, const Exp& exp, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = (mask ? _mm512_maskz_loadu_ps(tail, src + offset) : _mm512_loadu_ps(src + offset));
            __m512 _dst = exp.Tanh(_src);
            if (mask)
                _mm512_mask_storeu_ps(dst + offset, tail, _dst);
            else
                _mm512_storeu_ps(dst + offset, _dst);
        }

        void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            Exp exp(-2.0f*slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetTanh32f<false>(src, exp, dst, i + 0 * F);
                SynetTanh32f<false>(src, exp, dst, i + 1 * F);
                SynetTanh32f<false>(src, exp, dst, i + 2 * F);
                SynetTanh32f<false>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetTanh32f<false>(src, exp, dst, i);
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                SynetTanh32f<true>(src, exp, dst, i, tail);
            }
        }

    }
#endif
}

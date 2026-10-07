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
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)  
    namespace Sse41
    {
        SIMD_INLINE void SynetElu32f(const float * src, const Exp & exp, __m128 alpha, float * dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, exp.Elu(_mm_loadu_ps(src + offset), alpha));
        }

        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            __m128 _alpha = _mm_set1_ps(alpha[0]);
            Exp exp;
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetElu32f(src, exp, _alpha, dst, i + 0 * F);
                SynetElu32f(src, exp, _alpha, dst, i + 1 * F);
                SynetElu32f(src, exp, _alpha, dst, i + 2 * F);
                SynetElu32f(src, exp, _alpha, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetElu32f(src, exp, _alpha, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetElu32f(src[i], alpha[0]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetGelu32f(const float* src, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, Gelu(_mm_loadu_ps(src + offset)));
        }

        void SynetGelu32f(const float* src, size_t size, float* dst)
        {
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetGelu32f(src, dst, i + 0 * F);
                SynetGelu32f(src, dst, i + 1 * F);
                SynetGelu32f(src, dst, i + 2 * F);
                SynetGelu32f(src, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetGelu32f(src, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::Gelu(src[i]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetHardSigmoid32f(const float* src, __m128 scale, __m128 shift, float* dst, size_t offset)
        {
            __m128 _src = _mm_loadu_ps(src + offset);
            __m128 _dst = SynetHardSigmoid32f(_src, scale, shift);
            _mm_storeu_ps(dst + offset, _dst);
        }

        void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            __m128 _scale = _mm_set1_ps(scale[0]);
            __m128 _shift = _mm_set1_ps(shift[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHardSigmoid32f(src, _scale, _shift, dst, i + 0 * F);
                SynetHardSigmoid32f(src, _scale, _shift, dst, i + 1 * F);
                SynetHardSigmoid32f(src, _scale, _shift, dst, i + 2 * F);
                SynetHardSigmoid32f(src, _scale, _shift, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHardSigmoid32f(src, _scale, _shift, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetHardSigmoid32f(src[i], scale[0], shift[0]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetHswish32f(const float* src, __m128 shift, __m128 scale, float* dst, size_t offset)
        {
            __m128 _src = _mm_loadu_ps(src + offset);
            __m128 _dst = SynetHswish32f(_src, shift, scale);
            _mm_storeu_ps(dst + offset, _dst);
        }

        void SynetHswish32f(const float* src, size_t size, const float* shift, const float* scale, float* dst)
        {
            __m128 _shift = _mm_set1_ps(shift[0]);
            __m128 _scale = _mm_set1_ps(scale[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHswish32f(src, _shift, _scale, dst, i + 0 * F);
                SynetHswish32f(src, _shift, _scale, dst, i + 1 * F);
                SynetHswish32f(src, _shift, _scale, dst, i + 2 * F);
                SynetHswish32f(src, _shift, _scale, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHswish32f(src, _shift, _scale, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetHswish32f(src[i], shift[0], scale[0]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetMish32f(const float* src, __m128 threshold, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, Mish(_mm_loadu_ps(src + offset), threshold));
        }

        void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            __m128 _threshold = _mm_set1_ps(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetMish32f(src, _threshold, dst, i + 0 * F);
                SynetMish32f(src, _threshold, dst, i + 1 * F);
                SynetMish32f(src, _threshold, dst, i + 2 * F);
                SynetMish32f(src, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetMish32f(src, _threshold, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetMish32f(src[i], threshold[0]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetPreluLayerForward(const float* src, const float* slope, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, SynetRelu32f(_mm_loadu_ps(src + offset), _mm_loadu_ps(slope + offset)));
        }

        SIMD_INLINE void SynetPreluLayerForward(const float* src, __m128 slope, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, SynetRelu32f(_mm_loadu_ps(src + offset), slope));
        }

        void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m128 _slope = _mm_set1_ps(slope[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetPreluLayerForward(src, _slope, dst, s + F * 0);
                        SynetPreluLayerForward(src, _slope, dst, s + F * 1);
                        SynetPreluLayerForward(src, _slope, dst, s + F * 2);
                        SynetPreluLayerForward(src, _slope, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetPreluLayerForward(src, _slope, dst, s);
                }
                for (; s < spatial; ++s)
                    dst[s] = Base::SynetRelu32f(src[s], slope[c]);
                src += spatial;
                dst += spatial;
            }
        }

        void SynetPreluLayerForwardNhwc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetPreluLayerForward(src, slope, dst, c + F * 0);
                        SynetPreluLayerForward(src, slope, dst, c + F * 1);
                        SynetPreluLayerForward(src, slope, dst, c + F * 2);
                        SynetPreluLayerForward(src, slope, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetPreluLayerForward(src, slope, dst, c);
                }
                for (; c < channels; ++c)
                    dst[c] = Base::SynetRelu32f(src[c], slope[c]);
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

        SIMD_INLINE void SynetRelu32f(const float* src, __m128 slope, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, SynetRelu32f(_mm_loadu_ps(src + offset), slope));
        }

        void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            __m128 _slope = _mm_set1_ps(slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetRelu32f(src, _slope, dst, i + 0 * F);
                SynetRelu32f(src, _slope, dst, i + 1 * F);
                SynetRelu32f(src, _slope, dst, i + 2 * F);
                SynetRelu32f(src, _slope, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetRelu32f(src, _slope, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetRelu32f(src[i], slope[0]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetRelu16b(const uint16_t* src, __m128 slope, uint16_t* dst)
        {
            __m128i _src = _mm_loadu_si128((__m128i*)src);
            __m128 even = SynetRelu32f(BFloat16ToFloat32Even(_src), slope);
            __m128 odd = SynetRelu32f(BFloat16ToFloat32Odd(_src), slope);
            _mm_storeu_si128((__m128i*)dst, Float32ToBFloat16Interlived(even, odd));
        }

        void SynetRelu16b(const uint16_t* src, size_t size, const float* slope, uint16_t* dst)
        {
            __m128 _slope = _mm_set1_ps(slope[0]);
            size_t sizeDF = AlignLo(size, DF);

            size_t i = 0;
            for (; i < sizeDF; i += DF)
                SynetRelu16b(src + i, _slope, dst + i);
            for (; i < size; ++i)
                dst[i] = Base::SynetRelu16b(src[i], slope[0]);
        }

        //-------------------------------------------------------------------------------------------------

        void SynetRestrictRange32f(const float* src, size_t size, const float* lower, const float* upper, float* dst)
        {
            assert(lower[0] <= upper[0]);
            float min = *lower;
            float max = *upper;
            __m128 _min = _mm_set1_ps(min);
            __m128 _max = _mm_set1_ps(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                _mm_storeu_ps(dst + i + 0 * F, _mm_min_ps(_mm_max_ps(_min, _mm_loadu_ps(src + i + 0 * F)), _max));
                _mm_storeu_ps(dst + i + 1 * F, _mm_min_ps(_mm_max_ps(_min, _mm_loadu_ps(src + i + 1 * F)), _max));
                _mm_storeu_ps(dst + i + 2 * F, _mm_min_ps(_mm_max_ps(_min, _mm_loadu_ps(src + i + 2 * F)), _max));
                _mm_storeu_ps(dst + i + 3 * F, _mm_min_ps(_mm_max_ps(_min, _mm_loadu_ps(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                _mm_storeu_ps(dst + i, _mm_min_ps(_mm_max_ps(_min, _mm_loadu_ps(src + i)), _max));
            for (; i < size; ++i)
                dst[i] = Simd::RestrictRange(src[i], min, max);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetSigmoid32f(const float* src, const Exp & exp, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, exp.Sigmoid(_mm_loadu_ps(src + offset)));
        }

        void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSigmoid32f(src, exp, dst, i + 0 * F);
                SynetSigmoid32f(src, exp, dst, i + 1 * F);
                SynetSigmoid32f(src, exp, dst, i + 2 * F);
                SynetSigmoid32f(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSigmoid32f(src, exp, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSigmoid32f(src[i], slope[0]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetSoftplus32f(const float* src, __m128 beta, __m128 threshold, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, Softplus(_mm_loadu_ps(src + offset), beta, threshold));
        }

        void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            __m128 _beta = _mm_set1_ps(beta[0]);
            __m128 _threshold = _mm_set1_ps(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSoftplus32f(src, _beta, _threshold, dst, i + 0 * F);
                SynetSoftplus32f(src, _beta, _threshold, dst, i + 1 * F);
                SynetSoftplus32f(src, _beta, _threshold, dst, i + 2 * F);
                SynetSoftplus32f(src, _beta, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSoftplus32f(src, _beta, _threshold, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSoftplus32f(src[i], beta[0], threshold[0]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetSwish32f(const float* src, const Exp& exp, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, exp.Swish(_mm_loadu_ps(src + offset)));
        }

        void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSwish32f(src, exp, dst, i + 0 * F);
                SynetSwish32f(src, exp, dst, i + 1 * F);
                SynetSwish32f(src, exp, dst, i + 2 * F);
                SynetSwish32f(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSwish32f(src, exp, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSwish32f(src[i], slope[0]);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void SynetTanh32f(const float* src, const Exp& exp, float* dst, size_t offset)
        {
            _mm_storeu_ps(dst + offset, exp.Tanh(_mm_loadu_ps(src + offset)));
        }

        void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            Exp exp(-2.0f*slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetTanh32f(src, exp, dst, i + 0 * F);
                SynetTanh32f(src, exp, dst, i + 1 * F);
                SynetTanh32f(src, exp, dst, i + 2 * F);
                SynetTanh32f(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetTanh32f(src, exp, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetTanh32f(src[i], slope[0]);
        }
    }
#endif
}

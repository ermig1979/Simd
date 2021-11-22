/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_AVX512F_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Avx512f
    {
        template<bool align, bool mask> SIMD_INLINE void SynetElu32f(const float * src, const Avx512f::Exp & exp, __m512 alpha, float * dst, size_t offset, __mmask16 tail = -1)
        {
            Avx512f::Store<align, mask>(dst + offset, exp.Elu(Avx512f::Load<align, mask>(src + offset, tail), alpha), tail);
        }

        template<bool align> void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            __m512 _alpha = _mm512_set1_ps(alpha[0]);
            Avx512f::Exp exp;
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetElu32f<align, false>(src, exp, _alpha, dst, i + 0 * F);
                SynetElu32f<align, false>(src, exp, _alpha, dst, i + 1 * F);
                SynetElu32f<align, false>(src, exp, _alpha, dst, i + 2 * F);
                SynetElu32f<align, false>(src, exp, _alpha, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetElu32f<align, false>(src, exp, _alpha, dst, i);
            if(i < size)
                SynetElu32f<align, true>(src, exp, _alpha, dst, i, tail);

        }

        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetElu32f<true>(src, size, alpha, dst);
            else
                SynetElu32f<false>(src, size, alpha, dst);
        }

        //-------------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetHardSigmoid32f(const float* src, __m512 scale, __m512 shift, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _dst = SynetHardSigmoid32f(_src, scale, shift);
            Store<align, mask>(dst + offset, _dst, tail);
        }

        template<bool align> void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            __m512 _scale = _mm512_set1_ps(scale[0]);
            __m512 _shift = _mm512_set1_ps(shift[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHardSigmoid32f<align, false>(src, _scale, _shift, dst, i + 0 * F);
                SynetHardSigmoid32f<align, false>(src, _scale, _shift, dst, i + 1 * F);
                SynetHardSigmoid32f<align, false>(src, _scale, _shift, dst, i + 2 * F);
                SynetHardSigmoid32f<align, false>(src, _scale, _shift, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHardSigmoid32f<align, false>(src, _scale, _shift, dst, i);
            if (i < size)
                SynetHardSigmoid32f<align, true>(src, _scale, _shift, dst, i, tail);
        }

        void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHardSigmoid32f<true>(src, size, scale, shift, dst);
            else
                SynetHardSigmoid32f<false>(src, size, scale, shift, dst);
        }

        //---------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetHswish32f(const float * src, __m512 shift, __m512 scale, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _dst = _mm512_mul_ps(_mm512_mul_ps(_mm512_max_ps(_mm512_add_ps(_mm512_min_ps(_src, shift), shift), _mm512_setzero_ps()), scale), _src);
            Store<align, mask>(dst + offset, _dst, tail);
        }

        template<bool align> void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            __m512 _shift = _mm512_set1_ps(shift[0]);
            __m512 _scale = _mm512_set1_ps(scale[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i + 0 * F);
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i + 1 * F);
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i + 2 * F);
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHswish32f<align, false>(src, _shift, _scale, dst, i);
            if (i < size)
                SynetHswish32f<align, true>(src, _shift, _scale, dst, i, tail);
        }

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHswish32f<true>(src, size, shift, scale, dst);
            else
                SynetHswish32f<false>(src, size, shift, scale, dst);
        }

        //---------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetMish32f(const float* src, __m512 threshold, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            Store<align, mask>(dst + offset, Mish(_src, threshold), tail);
        }

        template<bool align> void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            __m512 _threshold = _mm512_set1_ps(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetMish32f<align, false>(src, _threshold, dst, i + 0 * F);
                SynetMish32f<align, false>(src, _threshold, dst, i + 1 * F);
                SynetMish32f<align, false>(src, _threshold, dst, i + 2 * F);
                SynetMish32f<align, false>(src, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetMish32f<align, false>(src, _threshold, dst, i);
            if (i < size)
                SynetMish32f<align, true>(src, _threshold, dst, i, tail);
        }

        void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetMish32f<true>(src, size, threshold, dst);
            else
                SynetMish32f<false>(src, size, threshold, dst);
        }

        //---------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void SynetPreluLayerForward(const float* src, const float* slope, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _slope = Load<align, mask>(slope + offset, tail);
            __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), _src);
            __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), _src);
            Store<align, mask>(dst + offset, _mm512_add_ps(pos, _mm512_mul_ps(_slope, neg)), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetPreluLayerForward(const float* src, __m512 slope, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 pos = _mm512_max_ps(_mm512_setzero_ps(), _src);
            __m512 neg = _mm512_min_ps(_mm512_setzero_ps(), _src);
            Store<align, mask>(dst + offset, _mm512_add_ps(pos, _mm512_mul_ps(slope, neg)), tail);
        }

        template <bool align> void SynetPreluLayerForward(const float* src, const float* slope, size_t count, size_t size, float* dst, SimdBool trans)
        {
            if (align)
                assert(((trans || size == 1) && count != 1 ? Aligned(count) && Aligned(slope) : Aligned(size)) && Aligned(src) && Aligned(dst));
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
                        SynetPreluLayerForward<align, false>(src, slope, dst, i + F * 0);
                        SynetPreluLayerForward<align, false>(src, slope, dst, i + F * 1);
                        SynetPreluLayerForward<align, false>(src, slope, dst, i + F * 2);
                        SynetPreluLayerForward<align, false>(src, slope, dst, i + F * 3);
                    }
                    for (; i < partial; i += F)
                        SynetPreluLayerForward<align, false>(src, slope, dst, i);
                    if (i < count)
                        SynetPreluLayerForward<align, true>(src, slope, dst, i, tail);
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
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j + F * 0);
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j + F * 1);
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j + F * 2);
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j + F * 3);
                    }
                    for (; j < partial; j += F)
                        SynetPreluLayerForward<align, false>(src, _slope, dst, j);
                    if (i < count)
                        SynetPreluLayerForward<align, true>(src, _slope, dst, j, tail);
                    src += size;
                    dst += size;
                }
            }
        }

        template <bool align> void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            __mmask16 tail = TailMask16(spatial - partial);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                __m512 _slope = _mm512_set1_ps(slope[c]);
                for (; s < aligned; s += QF)
                {
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s + F * 0);
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s + F * 1);
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s + F * 2);
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s + F * 3);
                }
                for (; s < partial; s += F)
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s);
                if (s < spatial)
                    SynetPreluLayerForward<align, true>(src, _slope, dst, s, tail);
                src += spatial;
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(spatial, F) && Aligned(dst))
                SynetPreluLayerForwardNchw<true>(src, slope, channels, spatial, dst);
            else
                SynetPreluLayerForwardNchw<false>(src, slope, channels, spatial, dst);
        }

        template <bool align> void SynetPreluLayerForwardNhwc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(slope) && Aligned(channels, F) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            __mmask16 tail = TailMask16(channels - partial);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += QF)
                {
                    SynetPreluLayerForward<align, false>(src, slope, dst, c + F * 0);
                    SynetPreluLayerForward<align, false>(src, slope, dst, c + F * 1);
                    SynetPreluLayerForward<align, false>(src, slope, dst, c + F * 2);
                    SynetPreluLayerForward<align, false>(src, slope, dst, c + F * 3);
                }
                for (; c < partial; c += F)
                    SynetPreluLayerForward<align, false>(src, slope, dst, c);
                if (c < channels)
                    SynetPreluLayerForward<align, true>(src, slope, dst, c, tail);
                src += channels;
                dst += channels;
            }
        }

        SIMD_INLINE void SynetPreluLayerForwardNhwc(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(slope) && Aligned(channels, F) && Aligned(dst))
                SynetPreluLayerForwardNhwc<true>(src, slope, channels, spatial, dst);
            else
                SynetPreluLayerForwardNhwc<false>(src, slope, channels, spatial, dst);
        }

        template <bool align> void SynetPreluLayerForwardNchw16c(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4) * F;
            for (size_t c = 0; c < channels; c += F)
            {
                __m512 _slope = Load<false>(slope + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s + F * 0);
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s + F * 1);
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s + F * 2);
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetPreluLayerForward<align, false>(src, _slope, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetPreluLayerForwardNchw16c(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetPreluLayerForwardNchw16c<true>(src, slope, channels, spatial, dst);
            else
                SynetPreluLayerForwardNchw16c<false>(src, slope, channels, spatial, dst);
        }

        void SynetPreluLayerForward(const float* src, const float* slope, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetPreluLayerForwardNchw(src, slope, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetPreluLayerForwardNhwc(src, slope, channels, spatial, dst);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetPreluLayerForward(src, slope, channels, spatial, dst, format);
            else if (format == SimdTensorFormatNchw8c)
                Avx::SynetPreluLayerForward(src, slope, channels, spatial, dst, format);
            else if (format == SimdTensorFormatNchw16c)
                SynetPreluLayerForwardNchw16c(src, slope, channels, spatial, dst);
            else
                Base::SynetPreluLayerForward(src, slope, channels, spatial, dst, format);
        }

        //-------------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetRelu32f(const float* src, __m512 slope, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _dst = SynetRelu32f(_src, slope);
            Store<align, mask>(dst + offset, _dst, tail);
        }

        template<bool align> void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            __m512 _slope = _mm512_set1_ps(slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            __mmask16 tail = TailMask16(size - sizeF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetRelu32f<align, false>(src, _slope, dst, i + 0 * F);
                SynetRelu32f<align, false>(src, _slope, dst, i + 1 * F);
                SynetRelu32f<align, false>(src, _slope, dst, i + 2 * F);
                SynetRelu32f<align, false>(src, _slope, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetRelu32f<align, false>(src, _slope, dst, i);
            if (i < size)
                SynetRelu32f<align, true>(src, _slope, dst, i, tail);
        }

        void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRelu32f<true>(src, size, slope, dst);
            else
                SynetRelu32f<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            assert(lower[0] <= upper[0]);
            if (align)
                assert(Aligned(src) && Aligned(dst));
            float min = *lower;
            float max = *upper;
            __m512 _min = _mm512_set1_ps(min);
            __m512 _max = _mm512_set1_ps(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i + 0 * F)), _max));
                Store<align>(dst + i + 1 * F, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i + 1 * F)), _max));
                Store<align>(dst + i + 2 * F, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i + 2 * F)), _max));
                Store<align>(dst + i + 3 * F, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, _mm512_min_ps(_mm512_max_ps(_min, Load<align>(src + i)), _max));
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                Store<align, true>(dst + i, _mm512_min_ps(_mm512_max_ps(_min, (Load<align, true>(src + i, tail))), _max), tail);
            }
        }

        void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRestrictRange32f<true>(src, size, lower, upper, dst);
            else
                SynetRestrictRange32f<false>(src, size, lower, upper, dst);
        }

        //---------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetSigmoid32f(const float* src, const Avx512f::Exp& exp, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _dst = exp.Sigmoid(_src);
            Store<align, mask>(dst + offset, _dst, tail);
        }

        template<bool align> void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSigmoid32f<align, false>(src, exp, dst, i + 0 * F);
                SynetSigmoid32f<align, false>(src, exp, dst, i + 1 * F);
                SynetSigmoid32f<align, false>(src, exp, dst, i + 2 * F);
                SynetSigmoid32f<align, false>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSigmoid32f<align, false>(src, exp, dst, i);
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                SynetSigmoid32f<align, true>(src, exp, dst, i, tail);
            }
        }

        void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSigmoid32f<true>(src, size, slope, dst);
            else
                SynetSigmoid32f<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetSoftplus32f(const float* src, __m512 beta, __m512 threshold, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _dst = Softplus(_src, beta, threshold);
            Store<align, mask>(dst + offset, _dst, tail);
        }

        template<bool align> void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            __m512 _beta = _mm512_set1_ps(beta[0]);
            __m512 _threshold = _mm512_set1_ps(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSoftplus32f<align, false>(src, _beta, _threshold, dst, i + 0 * F);
                SynetSoftplus32f<align, false>(src, _beta, _threshold, dst, i + 1 * F);
                SynetSoftplus32f<align, false>(src, _beta, _threshold, dst, i + 2 * F);
                SynetSoftplus32f<align, false>(src, _beta, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSoftplus32f<align, false>(src, _beta, _threshold, dst, i);
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                SynetSoftplus32f<align, true>(src, _beta, _threshold, dst, i, tail);
            }
        }

        void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSoftplus32f<true>(src, size, beta, threshold, dst);
            else
                SynetSoftplus32f<false>(src, size, beta, threshold, dst);
        }

        //---------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetSwish32f(const float* src, const Avx512f::Exp& exp, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _dst = exp.Swish(_src);
            Store<align, mask>(dst + offset, _dst, tail);
        }

        template<bool align> void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSwish32f<align, false>(src, exp, dst, i + 0 * F);
                SynetSwish32f<align, false>(src, exp, dst, i + 1 * F);
                SynetSwish32f<align, false>(src, exp, dst, i + 2 * F);
                SynetSwish32f<align, false>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSwish32f<align, false>(src, exp, dst, i);
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                SynetSwish32f<align, true>(src, exp, dst, i, tail);
            }
        }

        void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSwish32f<true>(src, size, slope, dst);
            else
                SynetSwish32f<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template<bool align, bool mask> SIMD_INLINE void SynetTanh32f(const float* src, const Avx512f::Exp& exp, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Load<align, mask>(src + offset, tail);
            __m512 _dst = exp.Tanh(_src);
            Store<align, mask>(dst + offset, _dst, tail);
        }

        template<bool align> void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            Exp exp(-2.0f*slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetTanh32f<align, false>(src, exp, dst, i + 0 * F);
                SynetTanh32f<align, false>(src, exp, dst, i + 1 * F);
                SynetTanh32f<align, false>(src, exp, dst, i + 2 * F);
                SynetTanh32f<align, false>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetTanh32f<align, false>(src, exp, dst, i);
            if (i < size)
            {
                __mmask16 tail = TailMask16(size - i);
                SynetTanh32f<align, true>(src, exp, dst, i, tail);
            }
        }

        void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetTanh32f<true>(src, size, slope, dst);
            else
                SynetTanh32f<false>(src, size, slope, dst);
        }
    }
#endif// SIMD_AVX512F_ENABLE
}

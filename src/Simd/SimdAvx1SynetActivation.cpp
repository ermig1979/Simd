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
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_AVX_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx
    {
        template<bool align> SIMD_INLINE void SynetHardSigmoid32f(const float* src, __m256 scale, __m256 shift, float* dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 _dst = SynetHardSigmoid32f(_src, scale, shift);
            Store<align>(dst + offset, _dst);
        }

        template<bool align> void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            __m256 _scale = _mm256_set1_ps(scale[0]);
            __m256 _shift = _mm256_set1_ps(shift[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i + 0 * F);
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i + 1 * F);
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i + 2 * F);
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHardSigmoid32f<align>(src, _scale, _shift, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetHardSigmoid32f(src[i], scale[0], shift[0]);
        }

        void SynetHardSigmoid32f(const float* src, size_t size, const float* scale, const float* shift, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHardSigmoid32f<true>(src, size, scale, shift, dst);
            else
                SynetHardSigmoid32f<false>(src, size, scale, shift, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetHswish32f(const float * src, __m256 shift, __m256 scale, float * dst, size_t offset)
        {
            __m256 value = Load<align>(src + offset);
            Store<align>(dst + offset, _mm256_mul_ps(_mm256_mul_ps(_mm256_max_ps(_mm256_add_ps(_mm256_min_ps(value, shift), shift), _mm256_setzero_ps()), scale), value));
        }

        template<bool align> void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            __m256 _shift = _mm256_set1_ps(shift[0]);
            __m256 _scale = _mm256_set1_ps(scale[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 0 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 1 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 2 * F);
                SynetHswish32f<align>(src, _shift, _scale, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetHswish32f<align>(src, _shift, _scale, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetHswish32f(src[i], shift[0], scale[0]);
        }

        void SynetHswish32f(const float * src, size_t size, const float * shift, const float * scale, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetHswish32f<true>(src, size, shift, scale, dst);
            else
                SynetHswish32f<false>(src, size, shift, scale, dst);
        }
        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetPreluLayerForward(const float* src, const float* slope, float* dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 _slope = Load<align>(slope + offset);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), _src);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), _src);
            Store<align>(dst + offset, _mm256_add_ps(pos, _mm256_mul_ps(_slope, neg)));
        }

        template <bool align> SIMD_INLINE void SynetPreluLayerForward(const float* src, __m256 slope, float* dst, size_t offset)
        {
            __m256 _src = Load<align>(src + offset);
            __m256 pos = _mm256_max_ps(_mm256_setzero_ps(), _src);
            __m256 neg = _mm256_min_ps(_mm256_setzero_ps(), _src);
            Store<align>(dst + offset, _mm256_add_ps(pos, _mm256_mul_ps(slope, neg)));
        }

        template <bool align> void SynetPreluLayerForwardNchw(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m256 _slope = _mm256_set1_ps(slope[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetPreluLayerForward<align>(src, _slope, dst, s + F * 0);
                        SynetPreluLayerForward<align>(src, _slope, dst, s + F * 1);
                        SynetPreluLayerForward<align>(src, _slope, dst, s + F * 2);
                        SynetPreluLayerForward<align>(src, _slope, dst, s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetPreluLayerForward<align>(src, _slope, dst, s);
                }
                for (; s < spatial; ++s)
                    dst[s] = Base::SynetRelu32f(src[s], slope[c]);
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
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetPreluLayerForward<align>(src, slope, dst, c + F * 0);
                        SynetPreluLayerForward<align>(src, slope, dst, c + F * 1);
                        SynetPreluLayerForward<align>(src, slope, dst, c + F * 2);
                        SynetPreluLayerForward<align>(src, slope, dst, c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetPreluLayerForward<align>(src, slope, dst, c);
                }
                for (; c < channels; ++c)
                    dst[c] = Base::SynetRelu32f(src[c], slope[c]);
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

        template <bool align> void SynetPreluLayerForwardNchw8c(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4) * F;
            for (size_t c = 0; c < channels; c += F)
            {
                __m256 _slope = Load<false>(slope + c);
                size_t s = 0;
                for (; s < spatial4F; s += 4 * F)
                {
                    SynetPreluLayerForward<align>(src, _slope, dst, s + F * 0);
                    SynetPreluLayerForward<align>(src, _slope, dst, s + F * 1);
                    SynetPreluLayerForward<align>(src, _slope, dst, s + F * 2);
                    SynetPreluLayerForward<align>(src, _slope, dst, s + F * 3);
                }
                for (; s < spatialF; s += F)
                    SynetPreluLayerForward<align>(src, _slope, dst, s);
                src += spatialF;
                dst += spatialF;
            }
        }

        SIMD_INLINE void SynetPreluLayerForwardNchw8c(const float* src, const float* slope, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetPreluLayerForwardNchw8c<true>(src, slope, channels, spatial, dst);
            else
                SynetPreluLayerForwardNchw8c<false>(src, slope, channels, spatial, dst);
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
                SynetPreluLayerForwardNchw8c(src, slope, channels, spatial, dst);
            else
                Base::SynetPreluLayerForward(src, slope, channels, spatial, dst, format);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetRelu32f(const float* src, __m256 slope, float* dst, size_t offset)
        {
            Store<align>(dst + offset, SynetRelu32f(Load<align>(src + offset), slope));
        }

        template<bool align> void SynetRelu32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            __m256 _slope = _mm256_set1_ps(slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetRelu32f<align>(src, _slope, dst, i + 0 * F);
                SynetRelu32f<align>(src, _slope, dst, i + 1 * F);
                SynetRelu32f<align>(src, _slope, dst, i + 2 * F);
                SynetRelu32f<align>(src, _slope, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetRelu32f<align>(src, _slope, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetRelu32f(src[i], slope[0]);
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
            __m256 _min = _mm256_set1_ps(min);
            __m256 _max = _mm256_set1_ps(max);
            size_t sizeF = Simd::AlignLo(size, F);
            size_t sizeQF = Simd::AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                Store<align>(dst + i + 0 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 0 * F)), _max));
                Store<align>(dst + i + 1 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 1 * F)), _max));
                Store<align>(dst + i + 2 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 2 * F)), _max));
                Store<align>(dst + i + 3 * F, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i + 3 * F)), _max));
            }
            for (; i < sizeF; i += F)
                Store<align>(dst + i, _mm256_min_ps(_mm256_max_ps(_min, Load<align>(src + i)), _max));
            for (; i < size; ++i)
                dst[i] = Simd::RestrictRange(src[i], min, max);
        }

        void SynetRestrictRange32f(const float * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetRestrictRange32f<true>(src, size, lower, upper, dst);
            else
                SynetRestrictRange32f<false>(src, size, lower, upper, dst);
        }
    }
#endif// SIMD_AVX_ENABLE
}

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
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdSynetScale8i.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        template <bool align, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float * src, const float * scale, const float * bias, float * dst, size_t offset)
        {
            __m256 _src = Avx::Load<align>(src + offset);
            __m256 _scale = Avx::Load<align>(scale + offset);
            __m256 _bias = Avx::Load<align>(bias + offset);
            Avx::Store<align>(dst + offset, Fmadd<nofma>(_src, _scale, _bias));
        }

        template <bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, float* dst, size_t offset, __m256i tail)
        {
            __m256 _src = _mm256_maskload_ps(src + offset, tail);
            __m256 _scale = _mm256_maskload_ps(scale + offset, tail);
            __m256 _bias = _mm256_maskload_ps(bias + offset, tail);
            _mm256_maskstore_ps(dst + offset, tail, Fmadd<nofma>(_src, _scale, _bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float* src, const float* scale, float* dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, _mm256_mul_ps(Avx::Load<align>(src + offset), Avx::Load<align>(scale + offset)));
        }

        template <bool align, bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float* src, const __m256& scale, const __m256& bias, float* dst, size_t offset)
        {
            __m256 _src = Avx::Load<align>(src + offset);
            Avx::Store<align>(dst + offset, Fmadd<nofma>(_src, scale, bias));
        }

        template <bool nofma> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, const __m256 & bias, float * dst, size_t offset, __m256i tail)
        {
            __m256 _src = _mm256_maskload_ps(src + offset, tail);
            _mm256_maskstore_ps(dst + offset, tail, Fmadd<nofma>(_src, scale, bias));
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForward(const float * src, const __m256 & scale, float * dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, _mm256_mul_ps(Avx::Load<align>(src + offset), scale));
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNchw(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(width, F) && Aligned(dst));

            size_t widthQF = AlignLo(width, QF);
            size_t widthF = AlignLo(width, F);
            if (bias)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        if (widthF)
                        {
                            __m256 _scale = _mm256_set1_ps(scale[c]);
                            __m256 _bias = _mm256_set1_ps(bias[c]);
                            for (; w < widthQF; w += QF)
                            {
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 0);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 1);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 2);
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w + F * 3);
                            }
                            for (; w < widthF; w += F)
                                SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, w);
                        }
                        for (; w < width; ++w)
                            dst[w] = src[w] * scale[c] + bias[c];
                        src += width;
                        dst += width;
                    }
                }
            }
            else
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    for (size_t h = 0; h < height; ++h)
                    {
                        size_t w = 0;
                        if (widthF)
                        {
                            __m256 _scale = _mm256_set1_ps(scale[c]);
                            for (; w < widthQF; w += QF)
                            {
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 0);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 1);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 2);
                                SynetScaleLayerForward<align>(src, _scale, dst, w + F * 3);
                            }
                            for (; w < widthF; w += F)
                                SynetScaleLayerForward<align>(src, _scale, dst, w);
                        }
                        for (; w < width; ++w)
                            dst[w] = src[w] * scale[c];
                        src += width;
                        dst += width;
                    }
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (!(Base::FmaAvoid(compatibility) && bias))
            {
                width = height * width;
                height = 1;
                if (Aligned(src) && Aligned(width, F) && Aligned(dst))
                    SynetScaleLayerForwardNchw<true, false>(src, scale, bias, channels, height, width, dst);
                else
                    SynetScaleLayerForwardNchw<false, false>(src, scale, bias, channels, height, width, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(width, F) && Aligned(dst))
                    SynetScaleLayerForwardNchw<true, true>(src, scale, bias, channels, height, width, dst);
                else
                    SynetScaleLayerForwardNchw<false, true>(src, scale, bias, channels, height, width, dst);
            }
        }

        template <bool align, bool nofma, bool notail> void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsQF = AlignLo(channels, QF);
            if (bias)
            {
                size_t widthF = AlignLo(width, F);
                __m256i tail = LeftNotZero32i(channels - channelsF);
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, nofma>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<nofma>(src, scale, bias, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                    for (; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 0);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 1);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 2);
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align, notail>(src, scale, bias, dst, c);
                        if (c < channels)
                            SynetScaleLayerForward<notail>(src, scale, bias, dst, c, tail);
                        src += channels;
                        dst += channels;
                    }
                }
            }
            else
            {
                for (size_t h = 0; h < height; ++h)
                {
                    for (size_t w = 0; w < width; ++w)
                    {
                        size_t c = 0;
                        for (; c < channelsQF; c += QF)
                        {
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 0);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 1);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 2);
                            SynetScaleLayerForward<align>(src, scale, dst, c + F * 3);
                        }
                        for (; c < channelsF; c += F)
                            SynetScaleLayerForward<align>(src, scale, dst, c);
                        for (; c < channels; ++c)
                            dst[c] = src[c] * scale[c];
                        src += channels;
                        dst += channels;
                    }
                }
            }
        }

        template <bool align> SIMD_INLINE void SynetScaleLayerForwardNhwc(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility) && bias)
                SynetScaleLayerForwardNhwc<align, true, true>(src, scale, bias, channels, height, width, dst);
            else if(Base::FmaNoTail(compatibility) && bias)
                SynetScaleLayerForwardNhwc<align, false, true>(src, scale, bias, channels, height, width, dst);
            else
                SynetScaleLayerForwardNhwc<align, false, false>(src, scale, bias, channels, height, width, dst);
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNhwc3(const float * src, const float * scale, const float * bias, size_t height, size_t width, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(width));

            size_t width3 = width * 3;
            size_t widthF3 = AlignLo(width, F) * 3;
            if (bias)
            {
                float _scale[F * 3], _bias[F * 3];
                for (size_t i = 0; i < F; ++i)
                    for (size_t c = 0; c < 3; ++c)
                        _scale[i * 3 + c] = scale[c], _bias[i * 3 + c] = bias[c];
                __m256 _scale0 = Load<false>(_scale + 0 * F);
                __m256 _scale1 = Load<false>(_scale + 1 * F);
                __m256 _scale2 = Load<false>(_scale + 2 * F);
                __m256 _bias0 = Load<false>(_bias + 0 * F);
                __m256 _bias1 = Load<false>(_bias + 1 * F);
                __m256 _bias2 = Load<false>(_bias + 2 * F);                
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<align, nofma>(src, _scale0, _bias0, dst, w + F * 0);
                        SynetScaleLayerForward<align, nofma>(src, _scale1, _bias1, dst, w + F * 1);
                        SynetScaleLayerForward<align, nofma>(src, _scale2, _bias2, dst, w + F * 2);
                    }
                    for (; w < width3; w += 3)
                    {
                        dst[w + 0] = src[w + 0] * scale[0] + bias[0];
                        dst[w + 1] = src[w + 1] * scale[1] + bias[1];
                        dst[w + 2] = src[w + 2] * scale[2] + bias[2];
                    }
                    src += width3;
                    dst += width3;
                }
            }
            else
            {
                float _scale[F * 3];
                for (size_t i = 0; i < F; ++i)
                    for (size_t c = 0; c < 3; ++c)
                        _scale[i * 3 + c] = scale[c];
                __m256 _scale0 = Load<false>(_scale + 0 * F);
                __m256 _scale1 = Load<false>(_scale + 1 * F);
                __m256 _scale2 = Load<false>(_scale + 2 * F);                
                for (size_t h = 0; h < height; ++h)
                {
                    size_t w = 0;
                    for (; w < widthF3; w += F * 3)
                    {
                        SynetScaleLayerForward<align>(src, _scale0, dst, w + F * 0);
                        SynetScaleLayerForward<align>(src, _scale1, dst, w + F * 1);
                        SynetScaleLayerForward<align>(src, _scale2, dst, w + F * 2);
                    }
                    for (; w < width3; w += 3)
                    {
                        dst[w + 0] = src[w + 0] * scale[0];
                        dst[w + 1] = src[w + 1] * scale[1];
                        dst[w + 2] = src[w + 2] * scale[2];
                    }
                    src += width3;
                    dst += width3;
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNhwc(const float * src, const float * scale, const float * bias, size_t channels, size_t height, size_t width, float * dst, SimdSynetCompatibilityType compatibility)
        {
            if (!(Base::FmaNoTail(compatibility) && bias))
            {
                width = height * width;
                height = 1;
            }
            if (channels == 3)
            {
                if (Base::FmaAvoid(compatibility) && bias)
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(width))
                        SynetScaleLayerForwardNhwc3<true, true>(src, scale, bias, height, width, dst);
                    else
                        SynetScaleLayerForwardNhwc3<false, true>(src, scale, bias, height, width, dst);
                }
                else
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(width))
                        SynetScaleLayerForwardNhwc3<true, false>(src, scale, bias, height, width, dst);
                    else
                        SynetScaleLayerForwardNhwc3<false, false>(src, scale, bias, height, width, dst);
                }
            }
            else
            {
                if (Aligned(src) && Aligned(scale) && Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                    SynetScaleLayerForwardNhwc<true>(src, scale, bias, channels, height, width, dst, compatibility);
                else
                    SynetScaleLayerForwardNhwc<false>(src, scale, bias, channels, height, width, dst, compatibility);
            }
        }

        template <bool align, bool nofma> void SynetScaleLayerForwardNchw8c(const float * src, const float * scale, const float * bias, size_t channels, size_t spatial, float * dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            size_t spatialF = spatial * F;
            size_t spatial4F = AlignLo(spatial, 4)*F;
            if (bias)
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    __m256 _scale = Load<false>(scale + c);
                    __m256 _bias = Load<false>(bias + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 0);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 1);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 2);
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align, nofma>(src, _scale, _bias, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
            else
            {
                for (size_t c = 0; c < channels; c += F)
                {
                    __m256 _scale = Load<false>(scale + c);
                    size_t s = 0;
                    for (; s < spatial4F; s += 4 * F)
                    {
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 0);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 1);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 2);
                        SynetScaleLayerForward<align>(src, _scale, dst, s + F * 3);
                    }
                    for (; s < spatialF; s += F)
                        SynetScaleLayerForward<align>(src, _scale, dst, s);
                    src += spatialF;
                    dst += spatialF;
                }
            }
        }

        SIMD_INLINE void SynetScaleLayerForwardNchw8c(const float* src, const float* scale, const float* bias, size_t channels, size_t spatial, float* dst, SimdSynetCompatibilityType compatibility)
        {
            if (Base::FmaAvoid(compatibility))
            {
                if (Aligned(src) && Aligned(dst))
                    SynetScaleLayerForwardNchw8c<true, true>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNchw8c<false, true>(src, scale, bias, channels, spatial, dst);
            }
            else
            {
                if (Aligned(src) && Aligned(dst))
                    SynetScaleLayerForwardNchw8c<true, false>(src, scale, bias, channels, spatial, dst);
                else
                    SynetScaleLayerForwardNchw8c<false, false>(src, scale, bias, channels, spatial, dst);
            }
        }

        void SynetScaleLayerForward(const float* src, const float* scale, const float* bias, size_t channels, size_t height, size_t width, float* dst, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
                SynetScaleLayerForwardNchw(src, scale, bias, channels, width, height, dst, compatibility);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetScaleLayerForwardNhwc(src, scale, bias, channels, height, width, dst, compatibility);
            else if (format == SimdTensorFormatNchw4c)
                Sse2::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
            else if (format == SimdTensorFormatNchw8c)
                SynetScaleLayerForwardNchw8c(src, scale, bias, channels, spatial, dst, compatibility);
            else
                Base::SynetScaleLayerForward(src, scale, bias, channels, height, width, dst, format, compatibility);
        }

        //---------------------------------------------------------------------

        SynetScale8i::SynetScale8i(const Base::Scale8iParam& p)
            : Sse41::SynetScale8i(p)
        {
        }

        //---------------------------------------------------------------------

        template<int part> SIMD_INLINE __m256 Cvt8uTo32f(__m128i src)
        {
            return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(src, part * 8)));
        }

        template <bool nofma> SIMD_INLINE void ScaleNchwDF(const uint8_t* src, __m256 scale, __m256 shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m128i s0 = Sse2::Load<false>((__m128i*)(src + offset));
            __m256i d0 = _mm256_cvtps_epi32(Fmadd<nofma>(Cvt8uTo32f<0>(s0), scale, shift));
            __m256i d1 = _mm256_cvtps_epi32(Fmadd<nofma>(Cvt8uTo32f<1>(s0), scale, shift));
            Sse2::Store<false>((__m128i*)(dst + offset), _mm256_extracti128_si256(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO), upper), 0));
        }

        template <bool nofma> SIMD_INLINE void ScaleNchwF(const uint8_t* src, __m256 scale, __m256 shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m128i _src = _mm_loadl_epi64((__m128i*)(src + offset));
            __m256i d0 = _mm256_cvtps_epi32(Fmadd<nofma>(Cvt8uTo32f<0>(_src), scale, shift));
            *((int64_t*)(dst + offset)) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool nofma> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatialDF = AlignLo(spatial, DF);
            size_t spatialF = AlignLo(spatial, F);
            __m256i _upper = _mm256_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    __m256 _shift = _mm256_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        ScaleNchwDF<nofma>(src, _scale, _shift, _upper, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF<nofma>(src, _scale, _shift, _upper, dst, s);
                    if (s < spatial)
                        ScaleNchwF<nofma>(src, _scale, _shift, _upper, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template<int part, bool align, bool nofma> SIMD_INLINE __m256i ScaleNhwcF(__m128i value, const float* scale, const float* shift)
        {
            return _mm256_cvtps_epi32(Fmadd<nofma>(Cvt8uTo32f<part>(value), Load<align>(scale + part * F), Load<align>(shift + part * F)));
        }

        template <bool align, bool nofma> SIMD_INLINE void ScaleNhwcDF(const uint8_t* src, const float* scale, const float* shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m128i s0 = Sse2::Load<false>((__m128i*)(src + offset));
            __m256i d0 = ScaleNhwcF<0, align, nofma>(s0, scale + offset, shift + offset);
            __m256i d1 = ScaleNhwcF<1, align, nofma>(s0, scale + offset, shift + offset);
            Sse2::Store<false>((__m128i*)(dst + offset), _mm256_extracti128_si256(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO), upper), 0));
        }

        template <bool align, bool nofma> SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m128i s0 = _mm_loadl_epi64((__m128i*)(src + offset));
            __m256i d0 = ScaleNhwcF<0, align, nofma>(s0, scale + offset, shift + offset);
            *(int64_t*)(dst + offset) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool align, bool nofma> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));
            size_t channelsF = AlignLo(channels, F);
            size_t channelsDF = AlignLo(channels, DF);
            __m256i _upper = _mm256_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        ScaleNhwcDF<align, nofma>(src, scale, shift, _upper, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, nofma>(src, scale, shift, _upper, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, nofma>(src, scale, shift, _upper, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNhwc<false, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const uint8_t* src, __m256 scale, __m256 shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m128i _src = _mm_loadl_epi64((__m128i*)(src + offset));
            __m256i d0 = _mm256_cvtps_epi32(Fmadd<nofma>(Cvt8uTo32f<0>(_src), scale, shift));
            *(int64_t*)(dst + offset) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool nofma> void ScaleNhwc3(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m256 _scale0 = Load<false>(_scale + 0 * F);
            __m256 _scale1 = Load<false>(_scale + 1 * F);
            __m256 _scale2 = Load<false>(_scale + 2 * F);
            __m256 _shift0 = Load<false>(_shift + 0 * F);
            __m256 _shift1 = Load<false>(_shift + 1 * F);
            __m256 _shift2 = Load<false>(_shift + 2 * F);
            __m256i _upper = _mm256_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, _upper, dst, s + F * 0);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, _upper, dst, s + F * 1);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, _upper, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, _upper, dst, spatial3 - F * 3);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, _upper, dst, spatial3 - F * 2);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, _upper, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, uint8_t* dst)
        {
            const Base::Scale8iParam& p = _param;
            bool nofma = Base::FmaAvoid(p.compatibility);
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
            {
                if(nofma)
                    ScaleNchw<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNchw<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
            {
                if (nofma)
                    ScaleNhwc3<true>(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc3<false>(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else
                Sse41::SynetScale8i::Scale(src, dst);
        }

        //---------------------------------------------------------------------

        template <bool nofma> SIMD_INLINE void ScaleNchwDF(const uint8_t* src, __m256 scale, __m256 shift, float* dst, size_t offset)
        {
            __m128i s0 = Sse2::Load<false>((__m128i*)(src + offset));
            Avx::Store<false>(dst + offset + 0, Fmadd<nofma>(Cvt8uTo32f<0>(s0), scale, shift));
            Avx::Store<false>(dst + offset + F, Fmadd<nofma>(Cvt8uTo32f<1>(s0), scale, shift));
        }

        template <bool nofma> SIMD_INLINE void ScaleNchwF(const uint8_t* src, __m256 scale, __m256 shift, float* dst, size_t offset)
        {
            __m128i s0 = _mm_loadl_epi64((__m128i*)(src + offset));
            Avx::Store<false>(dst + offset + 0, Fmadd<nofma>(Cvt8uTo32f<0>(s0), scale, shift));
        }

        template <bool nofma> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            assert(spatial >= F);

            size_t spatialDF = AlignLo(spatial, DF);
            size_t spatialF = AlignLo(spatial, F);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    __m256 _shift = _mm256_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        ScaleNchwDF<nofma>(src, _scale, _shift, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF<nofma>(src, _scale, _shift, dst, s);
                    if (s < spatial)
                        ScaleNchwF<nofma>(src, _scale, _shift, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template<int part, bool align, bool nofma> SIMD_INLINE void ScaleNhwcF(__m128i value, const float* scale, const float* shift, float* dst)
        {
            return Avx::Store<false>(dst + part * F, Fmadd<nofma>(Cvt8uTo32f<part>(value), Avx::Load<align>(scale + part * F), Avx::Load<align>(shift + part * F)));
        }

        template <bool align, bool nofma> SIMD_INLINE void ScaleNhwcDF(const uint8_t* src, const float* scale, const float* shift, float* dst, size_t offset)
        {
            __m128i s0 = Sse2::Load<false>((__m128i*)(src + offset));
            ScaleNhwcF<0, align, nofma>(s0, scale + offset, shift + offset, dst + offset);
            ScaleNhwcF<1, align, nofma>(s0, scale + offset, shift + offset, dst + offset);
        }

        template <bool align, bool nofma> SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, float* dst, size_t offset)
        {
            __m128i s0 = _mm_loadl_epi64((__m128i*)(src + offset));
            ScaleNhwcF<0, align, nofma>(s0, scale + offset, shift + offset, dst + offset);
        }

        template <bool align, bool nofma> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsDF = AlignLo(channels, DF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        ScaleNhwcDF<align, nofma>(src, scale, shift, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, nofma>(src, scale, shift, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, nofma>(src, scale, shift, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true, nofma>(src, scale, shift, batch, channels, spatial, dst);
            else
                ScaleNhwc<false, nofma>(src, scale, shift, batch, channels, spatial, dst);
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const uint8_t* src, __m256 scale, __m256 shift, float* dst, size_t offset)
        {
            __m128i s0 = _mm_loadl_epi64((__m128i*)(src + offset));
            Avx::Store<false>(dst + offset, Fmadd<nofma>(Cvt8uTo32f<0>(s0), scale, shift));
        }

        template <bool nofma> void ScaleNhwc3(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t spatial, float* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m256 _scale0 = Load<false>(_scale + 0 * F);
            __m256 _scale1 = Load<false>(_scale + 1 * F);
            __m256 _scale2 = Load<false>(_scale + 2 * F);
            __m256 _shift0 = Load<false>(_shift + 0 * F);
            __m256 _shift1 = Load<false>(_shift + 1 * F);
            __m256 _shift2 = Load<false>(_shift + 2 * F);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, dst, s + F * 0);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, dst, s + F * 1);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, dst, spatial3 - F * 3);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, dst, spatial3 - F * 2);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, float* dst)
        {
            const Base::Scale8iParam& p = _param;
            bool nofma = Base::FmaAvoid(p.compatibility);
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
            {
                if (nofma)
                    ScaleNchw<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
                else
                    ScaleNchw<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
            {
                if (nofma)
                    ScaleNhwc3<true>(src, _scale.data, _shift.data, p.batch, p.spatial, dst);
                else
                    ScaleNhwc3<false>(src, _scale.data, _shift.data, p.batch, p.spatial, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            }
            else
                Sse41::SynetScale8i::Scale(src, dst);
        }

        //---------------------------------------------------------------------

        template <bool align, bool nofma> SIMD_INLINE void ScaleNchwDF(const float* src, __m256 scale, __m256 shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m256i d0 = _mm256_cvtps_epi32(Fmadd<nofma>(Load<align>(src + offset + 0), scale, shift));
            __m256i d1 = _mm256_cvtps_epi32(Fmadd<nofma>(Load<align>(src + offset + F), scale, shift));
            Sse2::Store<false>((__m128i*)(dst + offset), _mm256_extracti128_si256(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO), upper), 0));
        }

        template <bool align, bool nofma> SIMD_INLINE void ScaleNchwF(const float* src, __m256 scale, __m256 shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m256i d0 = _mm256_cvtps_epi32(Fmadd<nofma>(Load<align>(src + offset + 0), scale, shift));
            *((int64_t*)(dst + offset)) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool align, bool nofma> void ScaleNchw(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(src) && Aligned(spatial, A));

            size_t spatialDF = AlignLo(spatial, DF);
            size_t spatialF = AlignLo(spatial, F);
            __m256i _upper = _mm256_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m256 _scale = _mm256_set1_ps(scale[c]);
                    __m256 _shift = _mm256_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        ScaleNchwDF<align, nofma>(src, _scale, _shift, _upper, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF<align, nofma>(src, _scale, _shift, _upper, dst, s);
                    if (s < spatial)
                        ScaleNchwF<align, nofma>(src, _scale, _shift, _upper, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void ScaleNchw(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(src) && Aligned(spatial, A))
                ScaleNchw<true, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNchw<false, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template<int part, bool align, bool nofma> SIMD_INLINE __m256i ScaleNhwcF(const float* src, const float* scale, const float* shift)
        {
            return _mm256_cvtps_epi32(Fmadd<nofma>(Load<false>(src + part * F), Load<align>(scale + part * F), Load<align>(shift + part * F)));
        }

        template <bool align, bool nofma> SIMD_INLINE void ScaleNhwcDF(const float* src, const float* scale, const float* shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m256i d0 = ScaleNhwcF<0, align, nofma>(src + offset, scale + offset, shift + offset);
            __m256i d1 = ScaleNhwcF<1, align, nofma>(src + offset, scale + offset, shift + offset);
            Sse2::Store<false>((__m128i*)(dst + offset), _mm256_extracti128_si256(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(d0, d1), K_ZERO), upper), 0));
        }

        template <bool align, bool nofma> SIMD_INLINE void ScaleNhwcF(const float* src, const float* scale, const float* shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m256i d0 = ScaleNhwcF<0, align, nofma>(src + offset, scale + offset, shift + offset);
            *(int64_t*)(dst + offset) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool align, bool nofma> void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsDF = AlignLo(channels, DF);
            __m256i _upper = _mm256_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        ScaleNhwcDF<align, nofma>(src, scale, shift, _upper, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, nofma>(src, scale, shift, _upper, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, nofma>(src, scale, shift, _upper, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNhwc<false, nofma>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const float* src, __m256 scale, __m256 shift, __m256i upper, uint8_t* dst, size_t offset)
        {
            __m256i d0 = _mm256_cvtps_epi32(Fmadd<nofma>(Load<false>(src + offset), scale, shift));
            *(int64_t*)(dst + offset) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool nofma> void ScaleNhwc3(const float* src, const float* scale, const float* shift, size_t batch, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m256 _scale0 = Load<false>(_scale + 0 * F);
            __m256 _scale1 = Load<false>(_scale + 1 * F);
            __m256 _scale2 = Load<false>(_scale + 2 * F);
            __m256 _shift0 = Load<false>(_shift + 0 * F);
            __m256 _shift1 = Load<false>(_shift + 1 * F);
            __m256 _shift2 = Load<false>(_shift + 2 * F);
            __m256i _upper = _mm256_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, _upper, dst, s + F * 0);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, _upper, dst, s + F * 1);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, _upper, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3<nofma>(src, _scale0, _shift0, _upper, dst, spatial3 - F * 3);
                    ScaleNhwc3<nofma>(src, _scale1, _shift1, _upper, dst, spatial3 - F * 2);
                    ScaleNhwc3<nofma>(src, _scale2, _shift2, _upper, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const float* src, uint8_t* dst)
        {
            const Base::Scale8iParam& p = _param;
            bool nofma = Base::FmaAvoid(p.compatibility);
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
            {
                if (nofma)
                    ScaleNchw<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNchw<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
            {
                if (nofma)
                    ScaleNhwc3<true>(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc3<false>(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
            }
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else
                Sse41::SynetScale8i::Scale(src, dst);
        }

        //---------------------------------------------------------------------

        void SynetScale8i::Scale(const float* src, float* dst)
        {
            const Base::Scale8iParam& p = _param;
            for (size_t b = 0; b < p.batch; ++b)
            {
                if(Base::FmaAvoid(p.compatibility) && p.format == SimdTensorFormatNchw)
                    Avx::SynetScaleLayerForward(src, _scale.data, _shift.data, p.channels, 1, p.spatial, dst, p.format, p.compatibility);
                else
                    Avx2::SynetScaleLayerForward(src, _scale.data, _shift.data, p.channels, 1, p.spatial, dst, p.format, p.compatibility);
                src += p.channels * p.spatial;
                dst += p.channels * p.spatial;
            }
        }

        //---------------------------------------------------------------------

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            Base::Scale8iParam param(batch, channels, spatial, srcType, dstType, format, compatibility);
            if (!param.Valid())
                return NULL;
            return new Avx2::SynetScale8i(param);
        }
    }
#endif// SIMD_AVX2_ENABLE
}

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
#include "Simd/SimdArray.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdSynetScale8i.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)      
    namespace Sse41
    {
        SynetScale8i::SynetScale8i(const Base::Scale8iParam& p)
            : Base::SynetScale8i(p)
        {
        }

        //---------------------------------------------------------------------

        template<int part> SIMD_INLINE __m128 Cvt8uTo32f(__m128i src)
        {
            return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(src, part * 4)));
        }

        template <bool align> SIMD_INLINE void ScaleNchwA(const uint8_t* src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i _src = Load<align>((__m128i*)(src + offset));
            __m128i d0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Cvt8uTo32f<0>(_src), scale), shift));
            __m128i d1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Cvt8uTo32f<1>(_src), scale), shift));
            __m128i d2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Cvt8uTo32f<2>(_src), scale), shift));
            __m128i d3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Cvt8uTo32f<3>(_src), scale), shift));
            Store<align>((__m128i*)(dst + offset), _mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)), upper));
        }

        SIMD_INLINE void ScaleNchwF(const uint8_t* src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i _src = _mm_cvtsi32_si128(*(int32_t*)(src + offset));
            __m128i d0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Cvt8uTo32f<0>(_src), scale), shift));
            *(int32_t*)(dst + offset) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool align> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(src) && Aligned(spatial, A) && Aligned(dst));

            size_t spatialA = AlignLo(spatial, A);
            size_t spatialF = AlignLo(spatial, F);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _scale = _mm_set1_ps(scale[c]);
                    __m128 _shift = _mm_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        ScaleNchwA<align>(src, _scale, _shift, _upper, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF(src, _scale, _shift, _upper, dst, s);
                    if (s < spatial)
                        ScaleNchwF(src, _scale, _shift, _upper, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        SIMD_INLINE void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(src) && Aligned(spatial, A) && Aligned(dst))
                ScaleNchw<true>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNchw<false>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template<int part, bool align> SIMD_INLINE __m128i ScaleNhwcF(__m128i value, const float* scale, const float* shift)
        {
            return _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Cvt8uTo32f<part>(value), Load<align>(scale + part * F)), Load<align>(shift + part * F)));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcA(const uint8_t* src, const float* scale, const float* shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i _src = Load<false>((__m128i*)(src + offset));
            __m128i d0 = ScaleNhwcF<0, align>(_src, scale + offset, shift + offset);
            __m128i d1 = ScaleNhwcF<1, align>(_src, scale + offset, shift + offset);
            __m128i d2 = ScaleNhwcF<2, align>(_src, scale + offset, shift + offset);
            __m128i d3 = ScaleNhwcF<3, align>(_src, scale + offset, shift + offset);
            Store<false>((__m128i*)(dst + offset), _mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)), upper));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float * scale, const float* shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i _src = _mm_cvtsi32_si128(*(int32_t*)(src + offset));
            __m128i d0 = ScaleNhwcF<0, align>(_src, scale + offset, shift + offset);
            *(int32_t*)(dst + offset) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool align> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsA = AlignLo(channels, A);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsA; c += A)
                        ScaleNhwcA<align>(src, scale, shift, _upper, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align>(src, scale, shift, _upper, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false>(src, scale, shift, _upper, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        SIMD_INLINE void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNhwc<false>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        SIMD_INLINE void ScaleNhwc3(const uint8_t* src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i _src = _mm_cvtsi32_si128(*(int32_t*)(src + offset));
            __m128i d0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Cvt8uTo32f<0>(_src), scale), shift));
            *(int32_t*)(dst + offset) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO), upper));
        }

        void ScaleNhwc3(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m128 _scale0 = Load<false>(_scale + 0 * F);
            __m128 _scale1 = Load<false>(_scale + 1 * F);
            __m128 _scale2 = Load<false>(_scale + 2 * F);
            __m128 _shift0 = Load<false>(_shift + 0 * F);
            __m128 _shift1 = Load<false>(_shift + 1 * F);
            __m128 _shift2 = Load<false>(_shift + 2 * F);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, _upper, dst, s + F * 0);
                    ScaleNhwc3(src, _scale1, _shift1, _upper, dst, s + F * 1);
                    ScaleNhwc3(src, _scale2, _shift2, _upper, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, _upper, dst, spatial3 - F * 3);
                    ScaleNhwc3(src, _scale1, _shift1, _upper, dst, spatial3 - F * 2);
                    ScaleNhwc3(src, _scale2, _shift2, _upper, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, uint8_t* dst)
        {
            const Base::Scale8iParam& p = _param;
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
                ScaleNchw(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
                ScaleNhwc3(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
                ScaleNhwc(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            else
                Base::SynetScale8i::Scale(src, dst);
        }

        //---------------------------------------------------------------------

        template <bool align> SIMD_INLINE void ScaleNchwA(const uint8_t* src, __m128 scale, __m128 shift, float* dst, size_t offset)
        {
            __m128i _src = Load<align>((__m128i*)(src + offset));
            Store<align>(dst + offset + 0 * F, _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<0>(_src), scale), shift));
            Store<align>(dst + offset + 1 * F, _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<1>(_src), scale), shift));
            Store<align>(dst + offset + 2 * F, _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<2>(_src), scale), shift));
            Store<align>(dst + offset + 3 * F, _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<3>(_src), scale), shift));
        }

        SIMD_INLINE void ScaleNchwF(const uint8_t* src, __m128 scale, __m128 shift, float* dst, size_t offset)
        {
            __m128i _src = _mm_cvtsi32_si128(*(int32_t*)(src + offset));
            Store<false>(dst + offset + 0 * F, _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<0>(_src), scale), shift));
        }

        template <bool align> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(src) && Aligned(spatial, A) && Aligned(dst));

            size_t spatialA = AlignLo(spatial, A);
            size_t spatialF = AlignLo(spatial, F);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _scale = _mm_set1_ps(scale[c]);
                    __m128 _shift = _mm_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        ScaleNchwA<align>(src, _scale, _shift, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF(src, _scale, _shift, dst, s);
                    if (s < spatial)
                        ScaleNchwF(src, _scale, _shift, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        SIMD_INLINE void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(src) && Aligned(spatial, A) && Aligned(dst))
                ScaleNchw<true>(src, scale, shift, batch, channels, spatial, dst);
            else
                ScaleNchw<false>(src, scale, shift, batch, channels, spatial, dst);
        }

        template<int part, bool align> SIMD_INLINE void ScaleNhwcF(__m128i value, const float* scale, const float* shift, float * dst)
        {
            return Store<false>(dst + part * F, _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<part>(value), Load<align>(scale + part * F)), Load<align>(shift + part * F)));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcA(const uint8_t* src, const float* scale, const float* shift, float* dst, size_t offset)
        {
            __m128i _src = Load<false>((__m128i*)(src + offset));
            ScaleNhwcF<0, align>(_src, scale + offset, shift + offset, dst + offset);
            ScaleNhwcF<1, align>(_src, scale + offset, shift + offset, dst + offset);
            ScaleNhwcF<2, align>(_src, scale + offset, shift + offset, dst + offset);
            ScaleNhwcF<3, align>(_src, scale + offset, shift + offset, dst + offset);
        }

        template <bool align> SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, float* dst, size_t offset)
        {
            __m128i _src = _mm_cvtsi32_si128(*(int32_t*)(src + offset));
            ScaleNhwcF<0, align>(_src, scale + offset, shift + offset, dst + offset);
        }

        template <bool align> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsA = AlignLo(channels, A);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += A)
                        ScaleNhwcA<align>(src, scale, shift, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align>(src, scale, shift, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false>(src, scale, shift, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        SIMD_INLINE void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true>(src, scale, shift, batch, channels, spatial, dst);
            else
                ScaleNhwc<false>(src, scale, shift, batch, channels, spatial, dst);
        }

        SIMD_INLINE void ScaleNhwc3(const uint8_t* src, __m128 scale, __m128 shift, float* dst, size_t offset)
        {
            __m128i _src = _mm_cvtsi32_si128(*(int32_t*)(src + offset));
            Store<false>(dst + offset, _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<0>(_src), scale), shift));
        }

        void ScaleNhwc3(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t spatial, float* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m128 _scale0 = Load<false>(_scale + 0 * F);
            __m128 _scale1 = Load<false>(_scale + 1 * F);
            __m128 _scale2 = Load<false>(_scale + 2 * F);
            __m128 _shift0 = Load<false>(_shift + 0 * F);
            __m128 _shift1 = Load<false>(_shift + 1 * F);
            __m128 _shift2 = Load<false>(_shift + 2 * F);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, dst, s + F * 0);
                    ScaleNhwc3(src, _scale1, _shift1, dst, s + F * 1);
                    ScaleNhwc3(src, _scale2, _shift2, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, dst, spatial3 - F * 3);
                    ScaleNhwc3(src, _scale1, _shift1, dst, spatial3 - F * 2);
                    ScaleNhwc3(src, _scale2, _shift2, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const uint8_t* src, float* dst)
        {
            const Base::Scale8iParam& p = _param;
            if (p.format == SimdTensorFormatNchw && p.spatial >= F && 0)
                ScaleNchw(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
                ScaleNhwc3(src, _scale.data, _shift.data, p.batch, p.spatial, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
                ScaleNhwc(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            else
                Base::SynetScale8i::Scale(src, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE __m128i ScaleNhwcF(const float* src, __m128 scale, __m128 shift, size_t offset)
        {
            return _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Load<align>(src + offset), scale), shift));
        }

        template <bool align> SIMD_INLINE void ScaleNchwA(const float* src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i d0 = ScaleNhwcF<align>(src, scale, shift, offset + 0 * F);
            __m128i d1 = ScaleNhwcF<align>(src, scale, shift, offset + 1 * F);
            __m128i d2 = ScaleNhwcF<align>(src, scale, shift, offset + 2 * F);
            __m128i d3 = ScaleNhwcF<align>(src, scale, shift, offset + 3 * F);
            Store<align>((__m128i*)(dst + offset), _mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)), upper));
        }

        template <bool align> SIMD_INLINE void ScaleNchwF(const float* src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i d0 = ScaleNhwcF<align>(src, scale, shift, offset);
            *(int32_t*)(dst + offset) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool align> void ScaleNchw(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(src) && Aligned(spatial, A) && Aligned(dst));

            size_t spatialA = AlignLo(spatial, A);
            size_t spatialF = AlignLo(spatial, F);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _scale = _mm_set1_ps(scale[c]);
                    __m128 _shift = _mm_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        ScaleNchwA<align>(src, _scale, _shift, _upper, dst, s);
                    for (; s < spatialF; s += F)
                        ScaleNchwF<align>(src, _scale, _shift, _upper, dst, s);
                    if (s < spatial)
                        ScaleNchwF<false>(src, _scale, _shift, _upper, dst, spatial - F);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        SIMD_INLINE void ScaleNchw(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(src) && Aligned(spatial, A) && Aligned(dst))
                ScaleNchw<true>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNchw<false>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        template<bool align> SIMD_INLINE __m128i ScaleNhwcF(const float * src, const float* scale, const float* shift, size_t offset)
        {
            return _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Load<false>(src + offset), Load<align>(scale + offset)), Load<align>(shift + offset)));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcA(const float* src, const float* scale, const float* shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i d0 = ScaleNhwcF<align>(src, scale, shift, offset + 0 * F);
            __m128i d1 = ScaleNhwcF<align>(src, scale, shift, offset + 1 * F);
            __m128i d2 = ScaleNhwcF<align>(src, scale, shift, offset + 2 * F);
            __m128i d3 = ScaleNhwcF<align>(src, scale, shift, offset + 3 * F);
            Store<false>((__m128i*)(dst + offset), _mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, d1), _mm_packs_epi32(d2, d3)), upper));
        }

        template <bool align> SIMD_INLINE void ScaleNhwcF(const float* src, const float* scale, const float* shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i d0 = ScaleNhwcF<align>(src, scale, shift, offset + 0 * F);
            *(int32_t*)(dst + offset) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO), upper));
        }

        template <bool align> void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsA = AlignLo(channels, A);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += A)
                        ScaleNhwcA<align>(src, scale, shift, _upper, dst, c);
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align>(src, scale, shift, _upper, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false>(src, scale, shift, _upper, dst, channels - F);
                    src += channels;
                    dst += channels;
                }
            }
        }

        SIMD_INLINE void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (Aligned(scale) && Aligned(shift))
                ScaleNhwc<true>(src, scale, shift, batch, channels, spatial, upper, dst);
            else
                ScaleNhwc<false>(src, scale, shift, batch, channels, spatial, upper, dst);
        }

        SIMD_INLINE void ScaleNhwc3(const float* src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m128i d0 = ScaleNhwcF<false>(src, scale, shift, offset + 0 * F);
            *(int32_t*)(dst + offset) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(d0, K_ZERO), K_ZERO), upper));
        }

        void ScaleNhwc3(const float* src, const float* scale, const float* shift, size_t batch, size_t spatial, int upper, uint8_t* dst)
        {
            assert(spatial >= F);

            size_t spatial3 = spatial * 3;
            size_t spatialF3 = AlignLo(spatial, F) * 3;
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];
            __m128 _scale0 = Load<false>(_scale + 0 * F);
            __m128 _scale1 = Load<false>(_scale + 1 * F);
            __m128 _scale2 = Load<false>(_scale + 2 * F);
            __m128 _shift0 = Load<false>(_shift + 0 * F);
            __m128 _shift1 = Load<false>(_shift + 1 * F);
            __m128 _shift2 = Load<false>(_shift + 2 * F);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatialF3; s += F * 3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, _upper, dst, s + F * 0);
                    ScaleNhwc3(src, _scale1, _shift1, _upper, dst, s + F * 1);
                    ScaleNhwc3(src, _scale2, _shift2, _upper, dst, s + F * 2);
                }
                if (s < spatial3)
                {
                    ScaleNhwc3(src, _scale0, _shift0, _upper, dst, spatial3 - F * 3);
                    ScaleNhwc3(src, _scale1, _shift1, _upper, dst, spatial3 - F * 2);
                    ScaleNhwc3(src, _scale2, _shift2, _upper, dst, spatial3 - F * 1);
                }
                src += spatial3;
                dst += spatial3;
            }
        }

        void SynetScale8i::Scale(const float* src, uint8_t* dst)
        {
            const Base::Scale8iParam& p = _param;
            if (p.format == SimdTensorFormatNchw && p.spatial >= F)
                ScaleNchw(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels == 3 && p.spatial >= F)
                ScaleNhwc3(src, _scale.data, _shift.data, p.batch, p.spatial, _dstCvt.uMax, dst);
            else if (p.format == SimdTensorFormatNhwc && p.channels >= F)
                ScaleNhwc(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            else
                Base::SynetScale8i::Scale(src, dst);
        }

        //---------------------------------------------------------------------

        void SynetScale8i::Scale(const float* src, float* dst)
        {
            const Base::Scale8iParam& p = _param;
            for (size_t b = 0; b < p.batch; ++b)
            {
                Sse2::SynetScaleLayerForward(src, _scale.data, _shift.data, p.channels, 1, p.spatial, dst, p.format, p.compatibility);
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
            return new Sse41::SynetScale8i(param);
        }
    }
#endif// SIMD_SSE41_ENABLE
}

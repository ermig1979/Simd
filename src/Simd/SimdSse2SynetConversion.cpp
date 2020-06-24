/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdMath.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template <bool align> SIMD_INLINE void SynetConvert32fTo8u(const float * src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst)
        {
            __m128i i32 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Sse::Load<align>(src), scale), shift));
            *((int32_t*)dst) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32, K_ZERO), K_ZERO), upper));
        }

        template <bool align> SIMD_INLINE void SynetConvert32fTo8uNchw(const float* src, __m128 scale, __m128 shift, __m128i upper, uint8_t* dst)
        {
            __m128i i32_0 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Sse::Load<align>(src + 0 * F), scale), shift));
            __m128i i32_1 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Sse::Load<align>(src + 1 * F), scale), shift));
            __m128i i32_2 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Sse::Load<align>(src + 2 * F), scale), shift));
            __m128i i32_3 = _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(Sse::Load<align>(src + 3 * F), scale), shift));
            Store<align>((__m128i*)dst, _mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(i32_0, i32_1), _mm_packs_epi32(i32_2, i32_3)), upper));
        }        
        
        template <bool align> void SynetConvert32fTo8uNchw(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(spatial, A));

            size_t spatialF = AlignLo(spatial, F);
            size_t spatialA = AlignLo(spatial, A);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m128 _scale = _mm_set1_ps(scale[c]);
                    __m128 _shift = _mm_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        SynetConvert32fTo8uNchw<align>(src + s, _scale, _shift, _upper, dst + s);
                    for (; s < spatialF; s += F)
                        SynetConvert32fTo8u<align>(src + s, _scale, _shift, _upper, dst + s);
                    for (; s < spatial; s += 1)
                            dst[s] = Base::SynetConvert32fTo8u(src[s], scale[c], shift[c], 0, upper);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align> void SynetConvert32fTo8uNhwc(const float* src, size_t batch, size_t channels, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(channels, A) && Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        SynetConvert32fTo8u<align>(src + c, Sse::Load<align>(scale + c), Sse::Load<align>(shift + c), _upper, dst + c);
                    for (; c < channels; ++c)
                        dst[c] = Base::SynetConvert32fTo8u(src[c], scale[c], shift[c], 0, upper);
                    src += channels;
                    dst += channels;
                }
            }
        }

        template <bool align> void SynetConvert32fTo8uNhwc3(const float* src, size_t batch, size_t spatial, const float* scale, const float* shift, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst) && Aligned(spatial, A));

            size_t spatial3 = spatial * 3;
            size_t spatial3F = AlignLo(spatial, F) * 3;
            __m128i _upper = _mm_set1_epi8(upper);
            float _scale[F * 3], _shift[F * 3];
            for (size_t i = 0; i < F; ++i)
                for (size_t c = 0; c < 3; ++c)
                    _scale[i * 3 + c] = scale[c], _shift[i * 3 + c] = shift[c];

            __m128 _scale0 = Sse::Load<false>(_scale + 0 * F);
            __m128 _scale1 = Sse::Load<false>(_scale + 1 * F);
            __m128 _scale2 = Sse::Load<false>(_scale + 2 * F);
            __m128 _shift0 = Sse::Load<false>(_shift + 0 * F);
            __m128 _shift1 = Sse::Load<false>(_shift + 1 * F);
            __m128 _shift2 = Sse::Load<false>(_shift + 2 * F);

            for (size_t b = 0; b < batch; ++b)
            {
                size_t s = 0;
                for (; s < spatial3F; s += 3 * F)
                {
                    SynetConvert32fTo8u<align>(src + 0 * F, _scale0, _shift0, _upper, dst + 0 * F);
                    SynetConvert32fTo8u<align>(src + 1 * F, _scale1, _shift1, _upper, dst + 1 * F);
                    SynetConvert32fTo8u<align>(src + 2 * F, _scale2, _shift2, _upper, dst + 2 * F);
                    src += 3 * F;
                    dst += 3 * F;
                }
                for (; s < spatial3; s += 3)
                {
                    dst[0] = Base::SynetConvert32fTo8u(src[0], scale[0], shift[0], 0, upper);
                    dst[1] = Base::SynetConvert32fTo8u(src[1], scale[1], shift[1], 0, upper);
                    dst[2] = Base::SynetConvert32fTo8u(src[2], scale[2], shift[2], 0, upper);
                    src += 3;
                    dst += 3;
                }
            }
        }

        void SynetConvert32fTo8u(const float* src, size_t batch, size_t channels, size_t height, size_t width, SimdTensorFormatType format, const float* scale, const float* shift, uint8_t* dst, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            size_t spatial = height * width;
            if (Base::NchwCompatible(channels, spatial, format))
            {
                if(Aligned(src) && Aligned(dst) && Aligned(spatial, A))
                    SynetConvert32fTo8uNchw<true>(src, batch, channels, spatial, scale, shift, upper, dst);
                else
                    SynetConvert32fTo8uNchw<false>(src, batch, channels, spatial, scale, shift, upper, dst);
            }
            else if (Base::NhwcCompatible(channels, spatial, format))
            {
                if (channels == 3)
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(spatial, A))
                        SynetConvert32fTo8uNhwc3<true>(src, batch, spatial, scale, shift, upper, dst);
                    else
                        SynetConvert32fTo8uNhwc3<false>(src, batch, spatial, scale, shift, upper, dst);
                }
                else
                {
                    if (Aligned(src) && Aligned(dst) && Aligned(channels, A) && Aligned(scale) && Aligned(shift))
                        SynetConvert32fTo8uNhwc<true>(src, batch, channels, spatial, scale, shift, upper, dst);
                    else
                        SynetConvert32fTo8uNhwc<false>(src, batch, channels, spatial, scale, shift, upper, dst);
                }
            }
            else
                assert(0);
        }
    }
#endif// SIMD_SSE2_ENABLE
}

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

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) && defined(SIMD_SYNET_ENABLE)   
    namespace Sse41
    {
        template<int part> SIMD_INLINE __m128 Cvt8uTo32f(__m128i src)
        {
            return _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_srli_si128(src, part * 4)));
        }

        template<int part> SIMD_INLINE __m128i SynetAdd8iNchw(__m128i a, __m128i b, __m128 scale[3], __m128 shift[3])
        {
            __m128 _a = _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<part>(a), scale[0]), shift[0]);
            __m128 _b = _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<part>(b), scale[1]), shift[1]);
            return _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_a, _b), scale[2]), shift[2]));
        }

        template <bool align> SIMD_INLINE void SynetAdd8iNchwA(const uint8_t* a, const uint8_t* b, __m128 scale[3], __m128 shift[3], __m128i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = Load<align>((__m128i*)(a + offset));
            __m128i _b = Load<align>((__m128i*)(b + offset));
            __m128i c0 = SynetAdd8iNchw<0>(_a, _b, scale, shift);
            __m128i c1 = SynetAdd8iNchw<1>(_a, _b, scale, shift);
            __m128i c2 = SynetAdd8iNchw<2>(_a, _b, scale, shift);
            __m128i c3 = SynetAdd8iNchw<3>(_a, _b, scale, shift);
            Store<align>((__m128i*)(c + offset), _mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(c0, c1), _mm_packs_epi32(c2, c3)), upper));
        }

        SIMD_INLINE void SynetAdd8iNchwF(const uint8_t* a, const uint8_t* b,  __m128 scale[3], __m128 shift[3], __m128i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = _mm_cvtsi32_si128(*(int32_t*)(a + offset));
            __m128i _b = _mm_cvtsi32_si128(*(int32_t*)(b + offset));
            __m128i c0 = SynetAdd8iNchw<0>(_a, _b, scale, shift);
            *(int32_t*)(c + offset) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(c0, K_ZERO), K_ZERO), upper));
        }

        template <bool align> void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            assert(spatial >= F);
            if (align)
                assert(Aligned(aData) && Aligned(bData) && Aligned(cData) && Aligned(spatial, A));

            size_t spatialA = AlignLo(spatial, A);
            size_t spatialF = AlignLo(spatial, F);
            __m128i _upper = _mm_set1_epi8(upper);
            __m128 scale[3], shift[3];
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    scale[0] = _mm_set1_ps(aScale[c]);
                    shift[0] = _mm_set1_ps(aShift[c]);
                    scale[1] = _mm_set1_ps(bScale[c]);
                    shift[1] = _mm_set1_ps(bShift[c]);
                    scale[2] = _mm_set1_ps(cScale[c]);
                    shift[2] = _mm_set1_ps(cShift[c]);
                    size_t s = 0;
                    for (; s < spatialA; s += A)
                        SynetAdd8iNchwA<align>(aData, bData, scale, shift, _upper, cData, s);
                    for (; s < spatialF; s += F)
                        SynetAdd8iNchwF(aData, bData, scale, shift, _upper, cData, s);
                    if (s < spatial)
                        SynetAdd8iNchwF(aData, bData, scale, shift, _upper, cData, spatial - F);
                    aData += spatial;
                    bData += spatial;
                    cData += spatial;
                }
            }
        }

        SIMD_INLINE void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (Aligned(aData) && Aligned(bData) && Aligned(cData) && Aligned(spatial, A))
                SynetAdd8iNchw<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                SynetAdd8iNchw<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
        }

        template<int part, bool align> SIMD_INLINE __m128i SynetAdd8iNhwc(__m128i a, const float* aScale, const float* aShift,
            __m128i b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, size_t offset)
        {
            __m128 _a = _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<part>(a), Load<align>(aScale + offset)), Load<align>(aShift + offset));
            __m128 _b = _mm_add_ps(_mm_mul_ps(Cvt8uTo32f<part>(b), Load<align>(bScale + offset)), Load<align>(bShift + offset));
            return _mm_cvtps_epi32(_mm_add_ps(_mm_mul_ps(_mm_add_ps(_a, _b), Load<align>(cScale + offset)), Load<align>(cShift + offset)));
        }

        template <bool align> SIMD_INLINE void SynetAdd8iNhwcA(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, __m128i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = Load<false>((__m128i*)(a + offset));
            __m128i _b = Load<false>((__m128i*)(b + offset));
            __m128i c0 = SynetAdd8iNhwc<0, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 0 * F);
            __m128i c1 = SynetAdd8iNhwc<1, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 1 * F);
            __m128i c2 = SynetAdd8iNhwc<2, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 2 * F);
            __m128i c3 = SynetAdd8iNhwc<3, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 3 * F);
            Store<false>((__m128i*)(c + offset), _mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(c0, c1), _mm_packs_epi32(c2, c3)), upper));
        }

        template <bool align> SIMD_INLINE void SynetAdd8iNhwcF(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, __m128i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = _mm_cvtsi32_si128(*(int32_t*)(a + offset));
            __m128i _b = _mm_cvtsi32_si128(*(int32_t*)(b + offset));
            __m128i c0 = SynetAdd8iNhwc<0, align>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 0 * F);
            *(int32_t*)(c + offset) = _mm_cvtsi128_si32(_mm_min_epu8(_mm_packus_epi16(_mm_packs_epi32(c0, K_ZERO), K_ZERO), upper));
        }

        template <bool align> void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsA = AlignLo(channels, A);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsA; c += A)
                        SynetAdd8iNhwcA<align>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c);
                    for (; c < channelsF; c += F)
                        SynetAdd8iNhwcF<align>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c);
                    if (c < channels)
                        SynetAdd8iNhwcF<false>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, channels - F);
                    aData += channels;
                    bData += channels;
                    cData += channels;
                }
            }
        }

        SIMD_INLINE void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift))
                SynetAdd8iNhwc<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                SynetAdd8iNhwc<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
        }

        void SynetAdd8i(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            if (format == SimdTensorFormatNchw && spatial >= F)
                SynetAdd8iNchw(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else if (format == SimdTensorFormatNhwc && channels >= F)
                SynetAdd8iNhwc(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                Base::SynetAdd8i(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, format, compatibility);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align> SIMD_INLINE void SynetAddBias(const float* bias, float* dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), Load<align>(bias)));
        }

        template <bool align> SIMD_INLINE void SynetAddBias(__m128 bias, float* dst)
        {
            Store<align>(dst, _mm_add_ps(Load<align>(dst), bias));
        }

        template <bool align> void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                if (partial)
                {
                    __m128 _bias = _mm_set1_ps(bias[c]);
                    for (; s < aligned; s += QF)
                    {
                        SynetAddBias<align>(_bias, dst + s + F * 0);
                        SynetAddBias<align>(_bias, dst + s + F * 1);
                        SynetAddBias<align>(_bias, dst + s + F * 2);
                        SynetAddBias<align>(_bias, dst + s + F * 3);
                    }
                    for (; s < partial; s += F)
                        SynetAddBias<align>(_bias, dst + s);
                }
                for (; s < spatial; ++s)
                    dst[s] += bias[c];
                dst += spatial;
            }
        }

        SIMD_INLINE void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(spatial, F) && Aligned(dst))
                SynetAddBiasNchw<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNchw<false>(bias, channels, spatial, dst);
        }

        template <bool align> void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(channels, F) && Aligned(bias) && Aligned(dst));

            size_t aligned = AlignLo(channels, QF);
            size_t partial = AlignLo(channels, F);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                if (partial)
                {
                    for (; c < aligned; c += QF)
                    {
                        SynetAddBias<align>(bias + c + F * 0, dst + c + F * 0);
                        SynetAddBias<align>(bias + c + F * 1, dst + c + F * 1);
                        SynetAddBias<align>(bias + c + F * 2, dst + c + F * 2);
                        SynetAddBias<align>(bias + c + F * 3, dst + c + F * 3);
                    }
                    for (; c < partial; c += F)
                        SynetAddBias<align>(bias + c, dst + c);
                }
                for (; c < channels; ++c)
                    dst[c] += bias[c];
                dst += channels;
            }
        }

        SIMD_INLINE void SynetAddBiasNhwc(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (Aligned(bias) && Aligned(channels, F) && Aligned(dst))
                SynetAddBiasNhwc<true>(bias, channels, spatial, dst);
            else
                SynetAddBiasNhwc<false>(bias, channels, spatial, dst);
        }

        void SynetAddBias(const float* bias, size_t channels, size_t spatial, float* dst, SimdTensorFormatType format)
        {
            if (Base::NchwCompatible(channels, spatial, format))
                SynetAddBiasNchw(bias, channels, spatial, dst);
            else if (Base::NhwcCompatible(channels, spatial, format))
                SynetAddBiasNhwc(bias, channels, spatial, dst);
            else
                assert(0);
        }
    }
#endif
}

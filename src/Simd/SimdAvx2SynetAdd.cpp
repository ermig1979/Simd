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
#include "Simd/SimdSynet.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdGather.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        template<int part> SIMD_INLINE __m256 Cvt8uTo32f(__m128i src)
        {
            return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_srli_si128(src, part * 8)));
        }

        template<bool nofma, int part> SIMD_INLINE __m256i SynetAdd8iNchw(__m128i a, __m128i b, __m256 scale[3], __m256 shift[3])
        {
            __m256 _a = Fmadd<nofma>(Cvt8uTo32f<part>(a), scale[0], shift[0]);
            __m256 _b = Fmadd<nofma>(Cvt8uTo32f<part>(b), scale[1], shift[1]);
            return _mm256_cvtps_epi32(Fmadd<nofma>(_mm256_add_ps(_a, _b), scale[2], shift[2]));
        }

        template <bool nofma> SIMD_INLINE void SynetAdd8iNchwDF(const uint8_t* a, const uint8_t* b, __m256 scale[3], __m256 shift[3], __m256i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = Sse41::Load<false>((__m128i*)(a + offset));
            __m128i _b = Sse41::Load<false>((__m128i*)(b + offset));
            __m256i c0 = SynetAdd8iNchw<nofma, 0>(_a, _b, scale, shift);
            __m256i c1 = SynetAdd8iNchw<nofma, 1>(_a, _b, scale, shift);
            Sse41::Store<false>((__m128i*)(c + offset), _mm256_extracti128_si256(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(c0, c1), K_ZERO), upper), 0));
        }

        template<bool nofma> SIMD_INLINE void SynetAdd8iNchwF(const uint8_t* a, const uint8_t* b, __m256 scale[3], __m256 shift[3], __m256i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = _mm_loadl_epi64((__m128i*)(a + offset));
            __m128i _b = _mm_loadl_epi64((__m128i*)(b + offset));
            __m256i c0 = SynetAdd8iNchw<nofma, 0>(_a, _b, scale, shift);
            *(int64_t*)(c + offset) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(c0, K_ZERO), K_ZERO), upper));
        }

        template <bool nofma> void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            assert(spatial >= F);

            size_t spatialDF = AlignLo(spatial, DF);
            size_t spatialF = AlignLo(spatial, F);
            __m256i _upper = _mm256_set1_epi8(upper);
            __m256 scale[3], shift[3];
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    scale[0] = _mm256_set1_ps(aScale[c]);
                    shift[0] = _mm256_set1_ps(aShift[c]);
                    scale[1] = _mm256_set1_ps(bScale[c]);
                    shift[1] = _mm256_set1_ps(bShift[c]);
                    scale[2] = _mm256_set1_ps(cScale[c]);
                    shift[2] = _mm256_set1_ps(cShift[c]);
                    size_t s = 0;
                    for (; s < spatialDF; s += DF)
                        SynetAdd8iNchwDF<nofma>(aData, bData, scale, shift, _upper, cData, s);
                    for (; s < spatialF; s += F)
                        SynetAdd8iNchwF<nofma>(aData, bData, scale, shift, _upper, cData, s);
                    if (s < spatial)
                        SynetAdd8iNchwF<nofma>(aData, bData, scale, shift, _upper, cData, spatial - F);
                    aData += spatial;
                    bData += spatial;
                    cData += spatial;
                }
            }
        }

        template<int part, bool align, bool nofma> SIMD_INLINE __m256i SynetAdd8iNhwc(__m128i a, const float* aScale, const float* aShift,
            __m128i b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, size_t offset)
        {
            __m256 _a = Fmadd<nofma>(Cvt8uTo32f<part>(a), Load<align>(aScale + offset), Load<align>(aShift + offset));
            __m256 _b = Fmadd<nofma>(Cvt8uTo32f<part>(b), Load<align>(bScale + offset), Load<align>(bShift + offset));
            return _mm256_cvtps_epi32(Fmadd<nofma>(_mm256_add_ps(_a, _b), Load<align>(cScale + offset), Load<align>(cShift + offset)));
        }

        template <bool align, bool nofma> SIMD_INLINE void SynetAdd8iNhwcDF(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, __m256i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = Sse41::Load<false>((__m128i*)(a + offset));
            __m128i _b = Sse41::Load<false>((__m128i*)(b + offset));
            __m256i c0 = SynetAdd8iNhwc<0, align, nofma>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 0 * F);
            __m256i c1 = SynetAdd8iNhwc<1, align, nofma>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 1 * F);
            Sse41::Store<false>((__m128i*)(c + offset), _mm256_extracti128_si256(_mm256_min_epu8(PackI16ToU8(PackI32ToI16(c0, c1), K_ZERO), upper), 0));
        }

        template <bool align, bool nofma> SIMD_INLINE void SynetAdd8iNhwcF(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, __m256i upper, uint8_t* c, size_t offset)
        {
            __m128i _a = _mm_loadl_epi64((__m128i*)(a + offset));
            __m128i _b = _mm_loadl_epi64((__m128i*)(b + offset));
            __m256i c0 = SynetAdd8iNhwc<0, align, nofma>(_a, aScale, aShift, _b, bScale, bShift, cScale, cShift, offset + 0 * F);
            *(int64_t*)(c + offset) = Extract64i<0>(_mm256_min_epu8(_mm256_packus_epi16(PackI32ToI16(c0, K_ZERO), K_ZERO), upper));
        }

        template <bool align, bool nofma> void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            assert(channels >= F);
            if (align)
                assert(Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift));

            size_t channelsF = AlignLo(channels, F);
            size_t channelsDF = AlignLo(channels, DF);
            __m256i _upper = _mm256_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsDF; c += DF)
                        SynetAdd8iNhwcDF<align, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c);
                    for (; c < channelsF; c += F)
                        SynetAdd8iNhwcF<align, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c);
                    if (c < channels)
                        SynetAdd8iNhwcF<false, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, channels - F);
                    aData += channels;
                    bData += channels;
                    cData += channels;
                }
            }
        }

        template <bool nofma> SIMD_INLINE void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift))
                SynetAdd8iNhwc<true, nofma>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            else
                SynetAdd8iNhwc<false, nofma>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
        }

        void SynetAdd8i(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility)
        {
            int upper = Base::Narrowed(compatibility) ? Base::U8_NARROWED_MAX : Base::U8_PRECISE_MAX;
            bool nofma = Base::FmaAvoid(compatibility);
            if (format == SimdTensorFormatNchw && spatial >= F)
            {
                if(nofma)
                    SynetAdd8iNchw<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
                else
                    SynetAdd8iNchw<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            }
            else if (format == SimdTensorFormatNhwc && channels >= F)
            {
                if (nofma)
                    SynetAdd8iNhwc<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
                else
                    SynetAdd8iNhwc<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            }
            else
                Sse41::SynetAdd8i(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, format, compatibility);
        }
    }
#endif// SIMD_AVX2_ENABLE
}

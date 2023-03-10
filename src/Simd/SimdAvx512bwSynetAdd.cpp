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
#include "Simd/SimdStore.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdAvx1.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdInterleave.h"
#include "Simd/SimdDeinterleave.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)     
    namespace Avx512bw
    {
        template<bool mask, bool nofma> SIMD_INLINE void SynetAdd8iNchwF(const uint8_t* a, const uint8_t* b, __m512 scale[3], __m512 shift[3], __m128i upper, uint8_t* c, size_t offset, __mmask16 tail = -1)
        {
            __m512 _a = Fmadd<nofma>(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(a + offset, tail)))), scale[0], shift[0]);
            __m512 _b = Fmadd<nofma>(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(b + offset, tail)))), scale[1], shift[1]);
            __m512i c32 = _mm512_cvtps_epi32(Fmadd<nofma>(_mm512_add_ps(_a, _b), scale[2], shift[2]));
            __m512i c8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(c32, K_ZERO), K_ZERO));
            Store<false, mask>(c + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(c8, 0), upper), tail);
        }

        template <bool nofma> void SynetAdd8iNchw(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tailF = TailMask16(spatial - spatialF);
            __m128i _upper = _mm_set1_epi8(upper);
            __m512 scale[3], shift[3];
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    scale[0] = _mm512_set1_ps(aScale[c]);
                    shift[0] = _mm512_set1_ps(aShift[c]);
                    scale[1] = _mm512_set1_ps(bScale[c]);
                    shift[1] = _mm512_set1_ps(bShift[c]);
                    scale[2] = _mm512_set1_ps(cScale[c]);
                    shift[2] = _mm512_set1_ps(cShift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        SynetAdd8iNchwF<false, nofma>(aData, bData, scale, shift, _upper, cData, s);
                    if (s < spatial)
                        SynetAdd8iNchwF<true, nofma>(aData, bData, scale, shift, _upper, cData, s, tailF);
                    aData += spatial;
                    bData += spatial;
                    cData += spatial;
                }
            }
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void SynetAdd8iNhwcF(const uint8_t* a, const float* aScale, const float* aShift,
            const uint8_t* b, const float* bScale, const float* bShift, const float* cScale, const float* cShift, __m128i upper, uint8_t* c, size_t offset, __mmask16 tail = -1)
        {
            __m512 _a = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(a + offset, tail))));
            __m512 _aScale = Load<align, mask>(aScale + offset, tail);
            __m512 _aShift = Load<align, mask>(aShift + offset, tail);
            _a = Fmadd<nofma>(_a, _aScale, _aShift);
            __m512 _b = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(b + offset, tail))));
            __m512 _bScale = Load<align, mask>(bScale + offset, tail);
            __m512 _bShift = Load<align, mask>(bShift + offset, tail);
            _b = Fmadd<nofma>(_b, _bScale, _bShift);
            __m512 _cScale = Load<align, mask>(cScale + offset, tail);
            __m512 _cShift = Load<align, mask>(cShift + offset, tail);
            __m512i c32 = _mm512_cvtps_epi32(Fmadd<nofma>(_mm512_add_ps(_a, _b), _cScale, _cShift));
            __m512i c8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(c32, K_ZERO), K_ZERO));
            Store<false, mask>(c + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(c8, 0), upper), tail);
        }

        template <bool align, bool nofma> void SynetAdd8iNhwc(const uint8_t* aData, const float* aScale, const float* aShift, const uint8_t* bData, const float* bScale, const float* bShift,
            uint8_t* cData, const float* cScale, const float* cShift, size_t batch, size_t channels, size_t spatial, int upper)
        {
            if (align)
                assert(Aligned(aScale) && Aligned(aShift) && Aligned(bScale) && Aligned(bShift) && Aligned(cScale) && Aligned(cShift));

            size_t channelsF = AlignLo(channels, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        SynetAdd8iNhwcF<align, false, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c);
                    if (c < channels)
                        SynetAdd8iNhwcF<align, true, nofma>(aData, aScale, aShift, bData, bScale, bShift, cScale, cShift, _upper, cData, c, tailF);
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
            if (format == SimdTensorFormatNchw && spatial > HF)
            {
                if (nofma)
                    SynetAdd8iNchw<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
                else
                    SynetAdd8iNchw<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            }
            else if (format == SimdTensorFormatNhwc && channels > HF)
            {
                if (nofma)
                    SynetAdd8iNhwc<true>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
                else
                    SynetAdd8iNhwc<false>(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, upper);
            }
            else
                Avx2::SynetAdd8i(aData, aScale, aShift, bData, bScale, bShift, cData, cScale, cShift, batch, channels, spatial, format, compatibility);
        }

        //-------------------------------------------------------------------------------------------------

        template <bool align, bool mask> SIMD_INLINE void SynetAddBias(const __m512& bias, float* dst, __mmask16 tail = -1)
        {
            Store<align, mask>(dst, _mm512_add_ps((Load<align, mask>(dst, tail)), bias), tail);
        }

        template <bool align, bool mask> SIMD_INLINE void SynetAddBias(const float* bias, float* dst, __mmask16 tail = -1)
        {
            __m512 _bias = Load<align, mask>(bias, tail);
            __m512 _dst = Load<align, mask>(dst, tail);
            Store<align, mask>(dst, _mm512_add_ps(_dst, _bias), tail);
        }

        template <bool align> void SynetAddBiasNchw(const float* bias, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(spatial, F) && Aligned(dst));

            size_t aligned = AlignLo(spatial, QF);
            size_t partial = AlignLo(spatial, F);
            __mmask16 tail = TailMask16(spatial - partial);
            for (size_t c = 0; c < channels; ++c)
            {
                size_t s = 0;
                __m512 _bias = _mm512_set1_ps(bias[c]);
                for (; s < aligned; s += QF)
                {
                    SynetAddBias<align, false>(_bias, dst + s + F * 0);
                    SynetAddBias<align, false>(_bias, dst + s + F * 1);
                    SynetAddBias<align, false>(_bias, dst + s + F * 2);
                    SynetAddBias<align, false>(_bias, dst + s + F * 3);
                }
                for (; s < partial; s += F)
                    SynetAddBias<align, false>(_bias, dst + s);
                if (s < spatial)
                    SynetAddBias<align, true>(_bias, dst + s, tail);
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
            __mmask16 tail = TailMask16(channels - partial);
            for (size_t s = 0; s < spatial; ++s)
            {
                size_t c = 0;
                for (; c < aligned; c += QF)
                {
                    SynetAddBias<align, false>(bias + c + F * 0, dst + c + F * 0);
                    SynetAddBias<align, false>(bias + c + F * 1, dst + c + F * 1);
                    SynetAddBias<align, false>(bias + c + F * 2, dst + c + F * 2);
                    SynetAddBias<align, false>(bias + c + F * 3, dst + c + F * 3);
                }
                for (; c < partial; c += F)
                    SynetAddBias<align, false>(bias + c, dst + c);
                if (c < channels)
                    SynetAddBias<align, true>(bias + c, dst + c, tail);
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

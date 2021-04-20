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
#include "Simd/SimdAvx512f.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdSynetScale8i.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx512bw
    {
        SynetScale8i::SynetScale8i(const Base::Scale8iParam& p)
            : Avx2::SynetScale8i(p)
        {
        }

        //---------------------------------------------------------------------

        template <bool mask, bool nofma> SIMD_INLINE void ScaleNchwF(const uint8_t* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(src + offset, tail))));
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Store<false, mask>(dst + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool nofma> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tailF = TailMask16(spatial - spatialF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        ScaleNchwF<false, nofma>(src, _scale, _shift, _upper, dst, s);
                    if (s < spatial)
                        ScaleNchwF<true, nofma>(src, _scale, _shift, _upper, dst, s, tailF);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, __m128i upper, uint8_t* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(src + offset, tail))));
            __m512 _scale = Avx512f::Load<align, mask>(scale + offset, tail);
            __m512 _shift = Avx512f::Load<align, mask>(shift + offset, tail);
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, _scale, _shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Store<false, mask>(dst + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool align, bool nofma> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, false, nofma>(src, scale, shift, _upper, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, true, nofma>(src, scale, shift, _upper, dst, c, tailF);
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

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const uint8_t* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(Sse2::Load<false>((__m128i*)(src + offset))));
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Sse2::Store<false>((__m128i*)(dst + offset), _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper));
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
            __m512 _scale0 = Avx512f::Load<false>(_scale + 0 * F);
            __m512 _scale1 = Avx512f::Load<false>(_scale + 1 * F);
            __m512 _scale2 = Avx512f::Load<false>(_scale + 2 * F);
            __m512 _shift0 = Avx512f::Load<false>(_shift + 0 * F);
            __m512 _shift1 = Avx512f::Load<false>(_shift + 1 * F);
            __m512 _shift2 = Avx512f::Load<false>(_shift + 2 * F);
            __m128i _upper = _mm_set1_epi8(upper);
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
            if (p.format == SimdTensorFormatNchw && p.spatial > HF)
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
            else if (p.format == SimdTensorFormatNhwc && p.channels != 3)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else
                Avx2::SynetScale8i::Scale(src, dst);
        }


        //---------------------------------------------------------------------

        template <bool mask, bool nofma> SIMD_INLINE void ScaleNchwF(const uint8_t* src, __m512 scale, __m512 shift, float* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(src + offset, tail))));
            Avx512f::Store<false, mask>(dst + offset, Fmadd<nofma>(_src, scale, shift), tail);
        }

        template <bool nofma> void ScaleNchw(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tailF = TailMask16(spatial - spatialF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        ScaleNchwF<false, nofma>(src, _scale, _shift, dst, s);
                    if (s < spatial)
                        ScaleNchwF<true, nofma>(src, _scale, _shift, dst, s, tailF);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void ScaleNhwcF(const uint8_t* src, const float* scale, const float* shift, float * dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32((Load<false, mask>(src + offset, tail))));
            __m512 _scale = Avx512f::Load<align, mask>(scale + offset, tail);
            __m512 _shift = Avx512f::Load<align, mask>(shift + offset, tail);
            Avx512f::Store<false, mask>(dst + offset, Fmadd<nofma>(_src, _scale, _shift), tail);
        }

        template <bool align, bool nofma> void ScaleNhwc(const uint8_t* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, float* dst)
        {
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, false, nofma>(src, scale, shift, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, true, nofma>(src, scale, shift, dst, c, tailF);
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

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const uint8_t* src, __m512 scale, __m512 shift, float* dst, size_t offset)
        {
            __m512 _src = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(Sse2::Load<false>((__m128i*)(src + offset))));
            Avx512f::Store<false>(dst + offset, Fmadd<nofma>(_src, scale, shift));
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
            __m512 _scale0 = Avx512f::Load<false>(_scale + 0 * F);
            __m512 _scale1 = Avx512f::Load<false>(_scale + 1 * F);
            __m512 _scale2 = Avx512f::Load<false>(_scale + 2 * F);
            __m512 _shift0 = Avx512f::Load<false>(_shift + 0 * F);
            __m512 _shift1 = Avx512f::Load<false>(_shift + 1 * F);
            __m512 _shift2 = Avx512f::Load<false>(_shift + 2 * F);
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
            else if (p.format == SimdTensorFormatNhwc && p.channels != 3)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, dst);
            }
            else
                Avx2::SynetScale8i::Scale(src, dst);
        }

        //---------------------------------------------------------------------

        template <bool mask, bool nofma> SIMD_INLINE void ScaleNchwF(const float* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Avx512f::Load<false, mask>(src + offset, tail);
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Store<false, mask>(dst + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool nofma> void ScaleNchw(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            size_t spatialF = AlignLo(spatial, F);
            __mmask16 tailF = TailMask16(spatial - spatialF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t c = 0; c < channels; ++c)
                {
                    __m512 _scale = _mm512_set1_ps(scale[c]);
                    __m512 _shift = _mm512_set1_ps(shift[c]);
                    size_t s = 0;
                    for (; s < spatialF; s += F)
                        ScaleNchwF<false, nofma>(src, _scale, _shift, _upper, dst, s);
                    if (s < spatial)
                        ScaleNchwF<true, nofma>(src, _scale, _shift, _upper, dst, s, tailF);
                    src += spatial;
                    dst += spatial;
                }
            }
        }

        template <bool align, bool mask, bool nofma> SIMD_INLINE void ScaleNhwcF(const float* src, const float* scale, const float* shift, __m128i upper, uint8_t* dst, size_t offset, __mmask16 tail = -1)
        {
            __m512 _src = Avx512f::Load<false, mask>(src + offset, tail);
            __m512 _scale = Avx512f::Load<align, mask>(scale + offset, tail);
            __m512 _shift = Avx512f::Load<align, mask>(shift + offset, tail);
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, _scale, _shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Store<false, mask>(dst + offset, _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper), tail);
        }

        template <bool align, bool nofma> void ScaleNhwc(const float* src, const float* scale, const float* shift, size_t batch, size_t channels, size_t spatial, int upper, uint8_t* dst)
        {
            if (align)
                assert(Aligned(scale) && Aligned(shift));

            size_t channelsF = AlignLo(channels, F);
            __mmask16 tailF = TailMask16(channels - channelsF);
            __m128i _upper = _mm_set1_epi8(upper);
            for (size_t b = 0; b < batch; ++b)
            {
                for (size_t s = 0; s < spatial; ++s)
                {
                    size_t c = 0;
                    for (; c < channelsF; c += F)
                        ScaleNhwcF<align, false, nofma>(src, scale, shift, _upper, dst, c);
                    if (c < channels)
                        ScaleNhwcF<false, true, nofma>(src, scale, shift, _upper, dst, c, tailF);
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

        template <bool nofma> SIMD_INLINE void ScaleNhwc3(const float* src, __m512 scale, __m512 shift, __m128i upper, uint8_t* dst, size_t offset)
        {
            __m512 _src = Avx512f::Load<false>(src + offset);
            __m512i _dst = _mm512_cvtps_epi32(Fmadd<nofma>(_src, scale, shift));
            __m512i u8 = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(_dst, K_ZERO), K_ZERO));
            Sse2::Store<false>((__m128i*)(dst + offset), _mm_min_epu8(_mm512_extracti32x4_epi32(u8, 0), upper));
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
            __m512 _scale0 = Avx512f::Load<false>(_scale + 0 * F);
            __m512 _scale1 = Avx512f::Load<false>(_scale + 1 * F);
            __m512 _scale2 = Avx512f::Load<false>(_scale + 2 * F);
            __m512 _shift0 = Avx512f::Load<false>(_shift + 0 * F);
            __m512 _shift1 = Avx512f::Load<false>(_shift + 1 * F);
            __m512 _shift2 = Avx512f::Load<false>(_shift + 2 * F);
            __m128i _upper = _mm_set1_epi8(upper);
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
            else if (p.format == SimdTensorFormatNhwc && p.channels != 3)
            {
                if (nofma)
                    ScaleNhwc<true>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
                else
                    ScaleNhwc<false>(src, _scale.data, _shift.data, p.batch, p.channels, p.spatial, _dstCvt.uMax, dst);
            }
            else
                Avx2::SynetScale8i::Scale(src, dst);
        }

        //---------------------------------------------------------------------

        void SynetScale8i::Scale(const float* src, float* dst)
        {
            const Base::Scale8iParam& p = _param;
            for (size_t b = 0; b < p.batch; ++b)
            {
                Avx512f::SynetScaleLayerForward(src, _scale.data, _shift.data, p.channels, 1, p.spatial, dst, p.format, p.compatibility);
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
            return new Avx512bw::SynetScale8i(param);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

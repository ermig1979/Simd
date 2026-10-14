/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        template <bool mask> SIMD_INLINE __m512i Float32ToUint8(const float * src, const __m512 & lower, const __m512 & upper, const __m512 & boost, __mmask16 tail = -1)
        {
            __m512 value = mask ? _mm512_maskz_loadu_ps(tail, src) : _mm512_loadu_ps(src);
            return _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_sub_ps(_mm512_min_ps(_mm512_max_ps(value, lower), upper), lower), boost));
        }

        template <bool mask> SIMD_INLINE void Float32ToUint8(const float * src, const __m512 & lower, const __m512 & upper, const __m512 & boost, uint8_t * dst, const __mmask16 * srcTails, __mmask64 dstTail)
        {
            __m512i d0 = Float32ToUint8<mask>(src + F * 0, lower, upper, boost, srcTails[0]);
            __m512i d1 = Float32ToUint8<mask>(src + F * 1, lower, upper, boost, srcTails[1]);
            __m512i d2 = Float32ToUint8<mask>(src + F * 2, lower, upper, boost, srcTails[2]);
            __m512i d3 = Float32ToUint8<mask>(src + F * 3, lower, upper, boost, srcTails[3]);
            __m512i packed = _mm512_permutexvar_epi32(K32_PERMUTE_FOR_TWO_UNPACK, _mm512_packus_epi16(_mm512_packs_epi32(d0, d1), _mm512_packs_epi32(d2, d3)));
            if (mask)
                _mm512_mask_storeu_epi8(dst, dstTail, packed);
            else
                _mm512_storeu_si512(dst, packed);
        }

        void Float32ToUint8(const float * src, size_t size, const float * lower, const float * upper, uint8_t * dst)
        {
            __m512 _lower = _mm512_set1_ps(lower[0]);
            __m512 _upper = _mm512_set1_ps(upper[0]);
            __m512 boost = _mm512_set1_ps(255.0f / (upper[0] - lower[0]));

            size_t alignedSize = AlignLo(size, A);
            __mmask16 srcTailMasks[4];
            for (size_t c = 0; c < 4; ++c)
                srcTailMasks[c] = TailMask16(size - alignedSize - F*c);
            __mmask64 dstTailMask = TailMask64(size - alignedSize);

            size_t i = 0;
            for (; i < alignedSize; i += A)
                Float32ToUint8<false>(src + i, _lower, _upper, boost, dst + i, srcTailMasks, dstTailMask);
            if (i < size)
                Float32ToUint8<true>(src + i, _lower, _upper, boost, dst + i, srcTailMasks, dstTailMask);
        }

        template <bool mask> SIMD_INLINE void Uint8ToFloat32(const __m128i & value, const __m512 & lower, const __m512 & boost, float * dst, __mmask16 tail)
        {
            __m512 result = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(value)), boost), lower);
            if (mask)
                _mm512_mask_storeu_ps(dst, tail, result);
            else
                _mm512_storeu_ps(dst, result);
        }

        template <bool mask> SIMD_INLINE void Uint8ToFloat32(const uint8_t * src, const __m512 & lower, const __m512 & boost, float * dst, __mmask64 srcTail, const __mmask16 * dstTails)
        {
            __m512i _src = mask ? _mm512_maskz_loadu_epi8(srcTail, src) : _mm512_loadu_si512(src);
            Uint8ToFloat32<mask>(_mm512_extracti32x4_epi32(_src, 0), lower, boost, dst + 0 * F, dstTails[0]);
            Uint8ToFloat32<mask>(_mm512_extracti32x4_epi32(_src, 1), lower, boost, dst + 1 * F, dstTails[1]);
            Uint8ToFloat32<mask>(_mm512_extracti32x4_epi32(_src, 2), lower, boost, dst + 2 * F, dstTails[2]);
            Uint8ToFloat32<mask>(_mm512_extracti32x4_epi32(_src, 3), lower, boost, dst + 3 * F, dstTails[3]);
        }

        void Uint8ToFloat32(const uint8_t * src, size_t size, const float * lower, const float * upper, float * dst)
        {
            __m512 _lower = _mm512_set1_ps(lower[0]);
            __m512 boost = _mm512_set1_ps((upper[0] - lower[0]) / 255.0f);

            size_t alignedSize = AlignLo(size, A);
            __mmask64 srcTailMask = TailMask64(size - alignedSize);
            __mmask16 dstTailMasks[4];
            for (size_t c = 0; c < 4; ++c)
                dstTailMasks[c] = TailMask16(size - alignedSize - F*c);

            size_t i = 0;
            for (; i < alignedSize; i += A)
                Uint8ToFloat32<false>(src + i, _lower, boost, dst + i, srcTailMask, dstTailMasks);
            if (i < size)
                Uint8ToFloat32<true>(src + i, _lower, boost, dst + i, srcTailMask, dstTailMasks);
        }

        void CosineDistance32f(const float * a, const float * b, size_t size, float * distance)
        {
            size_t partialAlignedSize = AlignLo(size, F);
            size_t fullAlignedSize = AlignLo(size, DF);
            size_t i = 0;
            __m512 _aa[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
            __m512 _ab[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
            __m512 _bb[2] = { _mm512_setzero_ps(), _mm512_setzero_ps() };
            if (fullAlignedSize)
            {
                for (; i < fullAlignedSize; i += DF)
                {
                    __m512 a0 = _mm512_loadu_ps(a + i + 0 * F);
                    __m512 b0 = _mm512_loadu_ps(b + i + 0 * F);
                    _aa[0] = _mm512_fmadd_ps(a0, a0, _aa[0]);
                    _ab[0] = _mm512_fmadd_ps(a0, b0, _ab[0]);
                    _bb[0] = _mm512_fmadd_ps(b0, b0, _bb[0]);
                    __m512 a1 = _mm512_loadu_ps(a + i + 1 * F);
                    __m512 b1 = _mm512_loadu_ps(b + i + 1 * F);
                    _aa[1] = _mm512_fmadd_ps(a1, a1, _aa[1]);
                    _ab[1] = _mm512_fmadd_ps(a1, b1, _ab[1]);
                    _bb[1] = _mm512_fmadd_ps(b1, b1, _bb[1]);
                }
                _aa[0] = _mm512_add_ps(_aa[0], _aa[1]);
                _ab[0] = _mm512_add_ps(_ab[0], _ab[1]);
                _bb[0] = _mm512_add_ps(_bb[0], _bb[1]);
            }
            for (; i < partialAlignedSize; i += F)
            {
                __m512 a0 = _mm512_loadu_ps(a + i);
                __m512 b0 = _mm512_loadu_ps(b + i);
                _aa[0] = _mm512_fmadd_ps(a0, a0, _aa[0]);
                _ab[0] = _mm512_fmadd_ps(a0, b0, _ab[0]);
                _bb[0] = _mm512_fmadd_ps(b0, b0, _bb[0]);
            }
            float aa = ExtractSum(_aa[0]), ab = ExtractSum(_ab[0]), bb = ExtractSum(_bb[0]);
            for (; i < size; ++i)
            {
                float _a = a[i];
                float _b = b[i];
                aa += _a * _a;
                ab += _a * _b;
                bb += _b * _b;
            }
            *distance = 1.0f - ab / ::sqrt(aa*bb);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

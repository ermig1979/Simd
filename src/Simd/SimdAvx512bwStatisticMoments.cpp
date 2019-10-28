/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdExtract.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        const __m512i K16_I = SIMD_MM512_SETR_EPI16(
            0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 
            0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
            0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
            0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37);

        SIMD_INLINE void GetObjectMoments16(__m512i src, __m512i col, __m512i & sx, __m512i & sxx)
        {
            sx = _mm512_add_epi32(sx, _mm512_madd_epi16(col, src));
            sxx = _mm512_add_epi32(sxx, _mm512_madd_epi16(src, _mm512_mullo_epi16(col, col)));
        }

        SIMD_INLINE void GetObjectMoments8(__m512i src, __mmask64 mask, __m512i& col, uint64_t & n, __m512i & s, __m512i & sx, __m512i & sxx)
        {
            n += Popcnt64(mask);
            __m512i _mask = _mm512_maskz_set1_epi8(mask, -1);
            src = _mm512_and_si512(src, _mask);
            s = _mm512_add_epi64(s, _mm512_sad_epu8(src, K_ZERO));
            GetObjectMoments16(_mm512_unpacklo_epi8(src, K_ZERO), col, sx, sxx);
            col = _mm512_add_epi16(col, K16_0008);
            GetObjectMoments16(_mm512_unpackhi_epi8(src, K_ZERO), col, sx, sxx);
            col = _mm512_add_epi16(col, K16_0038);
        }

        template <bool align> void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t index,
            uint64_t & n, __m512i & s, __m512i & sx, __m512i & sy, __m512i & sxx, __m512i& sxy, __m512i& syy)
        {
            size_t widthA = AlignLo(width, A);
            const size_t B = AlignLo(181, A);
            size_t widthB = AlignLoAny(width, B);
            __mmask64 tail = TailMask64(width - widthA);
            const __m512i _index = _mm512_set1_epi8(index);

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colB = 0; colB < width;)
                {
                    size_t colE = Simd::Min(colB + B, widthA);
                    __m512i _col = K16_I;
                    __m512i _s = _mm512_setzero_si512();
                    __m512i _sx = _mm512_setzero_si512();
                    __m512i _sxx = _mm512_setzero_si512();
                    if (mask == NULL)
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            __m512i _src = Load<align>(src + col);
                            GetObjectMoments8(_src, -1, _col, n, _s, _sx, _sxx);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            __m512i _src = Load<align, true>(src + widthA, tail);
                            GetObjectMoments8(_src, tail, _col, n, _s, _sx, _sxx);
                            colE = width;
                        }                        
                    }
                    else if (src == NULL)
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            __mmask64 _mask = _mm512_cmpeq_epi8_mask(Load<align>(mask + col), _index);
                            GetObjectMoments8(K8_01, _mask, _col, n, _s, _sx, _sxx);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            __mmask64 _mask = _mm512_cmpeq_epi8_mask((Load<align, true>(mask + widthA, tail)), _index)&tail;
                            GetObjectMoments8(K8_01, _mask, _col, n, _s, _sx, _sxx);
                            colE = width;
                        }
                    }
                    else
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            __m512i _src = Load<align>(src + col);
                            __mmask64 _mask = _mm512_cmpeq_epi8_mask(Load<align>(mask + col), _index);
                            GetObjectMoments8(_src, _mask, _col, n, _s, _sx, _sxx);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            __m512i _src = Load<align, true>(src + widthA, tail);
                            __mmask64 _mask = _mm512_cmpeq_epi8_mask((Load<align, true>(mask + widthA, tail)), _index) & tail;
                            GetObjectMoments8(_src, _mask, _col, n, _s, _sx, _sxx);
                            colE = width;
                        }
                    }
                    _sx = HorizontalSum32(_sx);
                    _sxx = HorizontalSum32(_sxx);

                    __m512i _y = _mm512_set1_epi32((int32_t)row);
                    __m512i _x0 = _mm512_set1_epi32((int32_t)colB);

                    s = _mm512_add_epi64(s, _s);

                    sx = _mm512_add_epi64(sx, _sx);
                    __m512i _sx0 = _mm512_mul_epu32(_s, _x0);
                    sx = _mm512_add_epi64(sx, _sx0);

                    __m512i _sy = _mm512_mul_epu32(_s, _y);
                    sy = _mm512_add_epi64(sy, _sy);

                    sxx = _mm512_add_epi64(sxx, _sxx);
                    sxx = _mm512_add_epi64(sxx, _mm512_mul_epu32(_sx, _mm512_add_epi64(_x0, _x0)));
                    sxx = _mm512_add_epi64(sxx, _mm512_mul_epu32(_sx0, _x0));

                    sxy = _mm512_add_epi64(sxy, _mm512_mul_epu32(_sx, _y));
                    sxy = _mm512_add_epi64(sxy, _mm512_mul_epu32(_sx0, _y));

                    syy = _mm512_add_epi64(syy, _mm512_mul_epu32(_sy, _y));

                    colB = colE;
                }
                if(src)
                    src += srcStride;
                if(mask)
                    mask += maskStride;
            }
        }

        template<bool align> void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            assert(src || mask);
            if (align)
                assert((src == NULL || (Aligned(src) && Aligned(srcStride))) && (mask == NULL || (Aligned(mask) && Aligned(maskStride))));

            *n = 0;
            __m512i _s = _mm512_setzero_si512();
            __m512i _sx = _mm512_setzero_si512();
            __m512i _sy = _mm512_setzero_si512();
            __m512i _sxx = _mm512_setzero_si512();
            __m512i _sxy = _mm512_setzero_si512();
            __m512i _syy = _mm512_setzero_si512();

            GetObjectMoments<align>(src, srcStride, width, height, mask, maskStride, index, *n, _s, _sx, _sy, _sxx, _sxy, _syy);

            *s = ExtractSum<uint64_t>(_s);
            *sx = ExtractSum<uint64_t>(_sx);
            *sy = ExtractSum<uint64_t>(_sy);
            *sxx = ExtractSum<uint64_t>(_sxx);
            *sxy = ExtractSum<uint64_t>(_sxy);
            *syy = ExtractSum<uint64_t>(_syy);
        }

        void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t* mask, size_t maskStride, uint8_t index,
            uint64_t* n, uint64_t* s, uint64_t* sx, uint64_t* sy, uint64_t* sxx, uint64_t* sxy, uint64_t* syy)
        {
            if ((src == NULL || (Aligned(src) && Aligned(srcStride))) && (mask == NULL || (Aligned(mask) && Aligned(maskStride))))
                GetObjectMoments<true>(src, srcStride, width, height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
            else
                GetObjectMoments<false>(src, srcStride, width, height, mask, maskStride, index, n, s, sx, sy, sxx, sxy, syy);
        }

        void GetMoments(const uint8_t* mask, size_t stride, size_t width, size_t height, uint8_t index,
            uint64_t* area, uint64_t* x, uint64_t* y, uint64_t* xx, uint64_t* xy, uint64_t* yy)
        {
            uint64_t stub;
            GetObjectMoments(NULL, 0, width, height, mask, stride, index, &stub, area, x, y, xx, xy, yy);
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

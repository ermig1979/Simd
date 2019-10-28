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
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        SIMD_INLINE void GetObjectMoments16(__m128i src, __m128i col, __m128i & sx, __m128i & sxx)
        {
            sx = _mm_add_epi32(sx, _mm_madd_epi16(col, src));
            sxx = _mm_add_epi32(sxx, _mm_madd_epi16(src, _mm_mullo_epi16(col, col)));
        }

        SIMD_INLINE void GetObjectMoments8(__m128i src, __m128i mask, __m128i& col, __m128i & n, __m128i & s, __m128i & sx, __m128i & sxx)
        {
            src = _mm_and_si128(src, mask);
            n = _mm_add_epi64(n, _mm_sad_epu8(_mm_and_si128(K8_01, mask), K_ZERO));
            s = _mm_add_epi64(s, _mm_sad_epu8(src, K_ZERO));
            GetObjectMoments16(_mm_unpacklo_epi8(src, K_ZERO), col, sx, sxx);
            col = _mm_add_epi16(col, K16_0008);
            GetObjectMoments16(_mm_unpackhi_epi8(src, K_ZERO), col, sx, sxx);
            col = _mm_add_epi16(col, K16_0008);
        }

        template <bool align> void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t index,
            __m128i & n, __m128i & s, __m128i & sx, __m128i & sy, __m128i & sxx, __m128i& sxy, __m128i& syy)
        {
            size_t widthA = AlignLo(width, A);
            const size_t B = AlignLo(181, A);
            size_t widthB = AlignLoAny(width, B);
            __m128i tailMask = ShiftLeft(K_INV_ZERO, A - width + widthA);

            const __m128i K16_I = _mm_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7);
            const __m128i _index = _mm_set1_epi8(index);
            const __m128i tailCol = _mm_add_epi16(K16_I, _mm_set1_epi16((int16_t)(width - A - widthB)));

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colB = 0; colB < width;)
                {
                    size_t colE = Simd::Min(colB + B, widthA);
                    __m128i _col = K16_I;
                    __m128i _n = _mm_setzero_si128();
                    __m128i _s = _mm_setzero_si128();
                    __m128i _sx = _mm_setzero_si128();
                    __m128i _sxx = _mm_setzero_si128();
                    if (mask == NULL)
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            __m128i _src = Load<align>((__m128i*)(src + col));
                            GetObjectMoments8(_src, K_INV_ZERO, _col, _n, _s, _sx, _sxx);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            __m128i _src = Load<false>((__m128i*)(src + width - A));
                            _col = tailCol;
                            GetObjectMoments8(_src, tailMask, _col, _n, _s, _sx, _sxx);
                            colE = width;
                        }                        
                    }
                    else if (src == NULL)
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            __m128i _mask = _mm_cmpeq_epi8(Load<align>((__m128i*)(mask + col)), _index);
                            GetObjectMoments8(K8_01, _mask, _col, _n, _s, _sx, _sxx);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            __m128i _mask = _mm_and_si128(_mm_cmpeq_epi8(Load<false>((__m128i*)(mask + width - A)), _index), tailMask);
                            _col = tailCol;
                            GetObjectMoments8(K8_01, _mask, _col, _n, _s, _sx, _sxx);
                            colE = width;
                        }
                    }
                    else
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            __m128i _src = Load<align>((__m128i*)(src + col));
                            __m128i _mask = _mm_cmpeq_epi8(Load<align>((__m128i*)(mask + col)), _index);
                            GetObjectMoments8(_src, _mask, _col, _n, _s, _sx, _sxx);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            __m128i _mask = _mm_and_si128(_mm_cmpeq_epi8(Load<false>((__m128i*)(mask + width - A)), _index), tailMask);
                            __m128i _src = Load<false>((__m128i*)(src + width - A));
                            _col = tailCol;
                            GetObjectMoments8(_src, _mask, _col, _n, _s, _sx, _sxx);
                            colE = width;
                        }
                    }
                    _sx = HorizontalSum32(_sx);
                    _sxx = HorizontalSum32(_sxx);

                    __m128i _y = _mm_set1_epi32((int32_t)row);
                    __m128i _x0 = _mm_set1_epi32((int32_t)colB);

                    n = _mm_add_epi64(n, _n);

                    s = _mm_add_epi64(s, _s);

                    sx = _mm_add_epi64(sx, _sx);
                    __m128i _sx0 = _mm_mul_epu32(_s, _x0);
                    sx = _mm_add_epi64(sx, _sx0);

                    __m128i _sy = _mm_mul_epu32(_s, _y);
                    sy = _mm_add_epi64(sy, _sy);

                    sxx = _mm_add_epi64(sxx, _sxx);
                    sxx = _mm_add_epi64(sxx, _mm_mul_epu32(_sx, _mm_add_epi64(_x0, _x0)));
                    sxx = _mm_add_epi64(sxx, _mm_mul_epu32(_sx0, _x0));

                    sxy = _mm_add_epi64(sxy, _mm_mul_epu32(_sx, _y));
                    sxy = _mm_add_epi64(sxy, _mm_mul_epu32(_sx0, _y));

                    syy = _mm_add_epi64(syy, _mm_mul_epu32(_sy, _y));

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
            assert(width >= A && (src || mask));
            if (align)
                assert((src == NULL || (Aligned(src) && Aligned(srcStride))) && (mask == NULL || (Aligned(mask) && Aligned(maskStride))));

            __m128i _n = _mm_setzero_si128();
            __m128i _s = _mm_setzero_si128();
            __m128i _sx = _mm_setzero_si128();
            __m128i _sy = _mm_setzero_si128();
            __m128i _sxx = _mm_setzero_si128();
            __m128i _sxy = _mm_setzero_si128();
            __m128i _syy = _mm_setzero_si128();

            GetObjectMoments<align>(src, srcStride, width, height, mask, maskStride, index, _n, _s, _sx, _sy, _sxx, _sxy, _syy);

            *n = ExtractInt64Sum(_n);
            *s = ExtractInt64Sum(_s);
            *sx = ExtractInt64Sum(_sx);
            *sy = ExtractInt64Sum(_sy);
            *sxx = ExtractInt64Sum(_sxx);
            *sxy = ExtractInt64Sum(_sxy);
            *syy = ExtractInt64Sum(_syy);
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
#endif// SIMD_SSE2_ENABLE
}

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
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        SIMD_INLINE void GetObjectMoments(uint16x4_t src, uint16x4_t col, uint32x4_t & sx, uint32x4_t & sxx)
        {
            sx = vmlal_u16(sx, src, col);
            sxx = vmlal_u16(sxx, src, vmul_u16(col, col));
        }

        SIMD_INLINE void GetObjectMoments(uint16x8_t src, uint16x8_t col, uint16x8_t & s, uint32x4_t& sx, uint32x4_t& sxx)
        {
            s = vaddq_u16(s, src);
            GetObjectMoments(Half<0>(src), Half<0>(col), sx, sxx);
            GetObjectMoments(Half<1>(src), Half<1>(col), sx, sxx);
        }

        SIMD_INLINE void GetObjectMoments(uint8x16_t src, uint8x16_t mask, uint16x8_t & col, uint8x16_t & n, uint16x8_t & s, uint32x4_t & sx, uint32x4_t & sxx)
        {
            src = vandq_u8(src, mask);
            n = vaddq_u8(n, vandq_u8(K8_01, mask));
            GetObjectMoments(UnpackU8<0>(src), col, s, sx, sxx);
            col = vaddq_u16(col, K16_0008);
            GetObjectMoments(UnpackU8<1>(src), col, s, sx, sxx);
            col = vaddq_u16(col, K16_0008);
        }

        template <bool align> void GetObjectMoments(const uint8_t* src, size_t srcStride, size_t width, size_t height, const uint8_t * mask, size_t maskStride, uint8_t index,
            uint64x2_t & n, uint64x2_t & s, uint64x2_t & sx, uint64x2_t & sy, uint64x2_t & sxx, uint64x2_t& sxy, uint64x2_t& syy)
        {
            size_t widthA = AlignLo(width, A);
            const size_t B = AlignLo(181, A);
            size_t widthB = AlignLoAny(width, B);
            uint8x16_t tailMask = ShiftLeft(K8_FF, A - width + widthA);

            const uint16x8_t K16_I = SIMD_VEC_SETR_EPI16(0, 1, 2, 3, 4, 5, 6, 7);
            const uint8x16_t _index = vdupq_n_u8(index);
            const uint16x8_t tailCol = vaddq_u16(K16_I, vdupq_n_u16((uint16_t)(width - A - widthB))); 

            for (size_t row = 0; row < height; ++row)
            {
                for (size_t colB = 0; colB < width;)
                {
                    size_t colE = Simd::Min(colB + B, widthA);
                    uint16x8_t _col = K16_I;
                    uint8x16_t _n8 = K8_00;
                    uint16x8_t _s16 = K16_0000;
                    uint32x4_t _sx32 = K32_00000000;
                    uint32x4_t _sxx32 = K32_00000000;
                    if (mask == NULL)
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            uint8x16_t _src = Load<align>(src + col);
                            GetObjectMoments(_src, K8_FF, _col, _n8, _s16, _sx32, _sxx32);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            uint8x16_t _src = Load<false>(src + width - A);
                            _col = tailCol;
                            GetObjectMoments(_src, tailMask, _col, _n8, _s16, _sx32, _sxx32);
                            colE = width;
                        }                        
                    }
                    else if (src == NULL)
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            uint8x16_t _mask = vceqq_u8(Load<align>(mask + col), _index);
                            GetObjectMoments(K8_01, _mask, _col, _n8, _s16, _sx32, _sxx32);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            uint8x16_t _mask = vandq_u8(vceqq_u8(Load<false>(mask + width - A), _index), tailMask);
                            _col = tailCol;
                            GetObjectMoments(K8_01, _mask, _col, _n8, _s16, _sx32, _sxx32);
                            colE = width;
                        }
                    }
                    else
                    {
                        for (size_t col = colB; col < colE; col += A)
                        {
                            uint8x16_t _src = Load<align>(src + col);
                            uint8x16_t _mask = vceqq_u8(Load<align>(mask + col), _index);
                            GetObjectMoments(_src, _mask, _col, _n8, _s16, _sx32, _sxx32);
                        }
                        if (colB == widthB && widthA < width)
                        {
                            uint8x16_t _mask = vandq_u8(vceqq_u8(Load<false>(mask + width - A), _index), tailMask);
                            uint8x16_t _src = Load<false>(src + width - A);
                            _col = tailCol;
                            GetObjectMoments(_src, _mask, _col, _n8, _s16, _sx32, _sxx32);
                            colE = width;
                        }
                    }
                    uint32x2_t _s = vmovn_u64(vpaddlq_u32(vpaddlq_u16(_s16)));
                    uint32x2_t _sx = vpadd_u32(Half<0>(_sx32), Half<1>(_sx32));
                    uint32x2_t _sxx = vpadd_u32(Half<0>(_sxx32), Half<1>(_sxx32));
                    uint32x2_t _y = vdup_n_u32((uint32_t)row);
                    uint32x2_t _x = vdup_n_u32((uint32_t)colB);

                    n = vaddq_u64(n, vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(_n8))));

                    s = vaddq_u64(s, vpaddlq_u32(vpaddlq_u16(_s16)));

                    sx = vaddw_u32(sx, _sx);
                    sx = vmlal_u32(sx, _s, _x);

                    sy = vmlal_u32(sy, _s, _y);

                    sxx = vaddw_u32(sxx, _sxx);
                    sxx = vmlal_u32(sxx, _sx, vadd_u32(_x, _x));
                    sxx = vmlal_u32(sxx, _s, vmul_u32(_x, _x));

                    sxy = vmlal_u32(sxy, _sx, _y);
                    sxy = vmlal_u32(sxy, _s, vmul_u32(_x, _y));

                    syy = vmlal_u32(syy, _s, vmul_u32(_y, _y));

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

            uint64x2_t _n = vdupq_n_u64(0);
            uint64x2_t _s = vdupq_n_u64(0);
            uint64x2_t _sx = vdupq_n_u64(0);
            uint64x2_t _sy = vdupq_n_u64(0);
            uint64x2_t _sxx = vdupq_n_u64(0);
            uint64x2_t _sxy = vdupq_n_u64(0);
            uint64x2_t _syy = vdupq_n_u64(0);

            GetObjectMoments<align>(src, srcStride, width, height, mask, maskStride, index, _n, _s, _sx, _sy, _sxx, _sxy, _syy);

            *n = ExtractSum64u(_n);
            *s = ExtractSum64u(_s);
            *sx = ExtractSum64u(_sx);
            *sy = ExtractSum64u(_sy);
            *sxx = ExtractSum64u(_sxx);
            *sxy = ExtractSum64u(_sxy);
            *syy = ExtractSum64u(_syy);
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
#endif// SIMD_NEON_ENABLE
}

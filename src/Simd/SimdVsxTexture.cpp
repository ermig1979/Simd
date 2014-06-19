/*
* Simd Library.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
#include "Simd/SimdVsx.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdConst.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCompare.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdLog.h"

namespace Simd
{
#ifdef SIMD_VSX_ENABLE  
    namespace Vsx
    {
        SIMD_INLINE v128_s16 TextureBoostedSaturatedGradient16(v128_s16 a, v128_s16 b, const v128_s16 & saturation, const v128_s16 & boost)
        {
            return vec_mladd(vec_max((v128_s16)K16_0000, vec_add(saturation, vec_min(vec_sub(b, a), saturation))), boost, (v128_s16)K16_0000);
        }

        SIMD_INLINE v128_u8 TextureBoostedSaturatedGradient8(v128_u8 a, v128_u8 b, const v128_s16 & saturation, const v128_s16 & boost) 
        {
            v128_s16 lo = TextureBoostedSaturatedGradient16((v128_s16)UnpackLoU8(a), (v128_s16)UnpackLoU8(b), saturation, boost);
            v128_s16 hi = TextureBoostedSaturatedGradient16((v128_s16)UnpackHiU8(a), (v128_s16)UnpackHiU8(b), saturation, boost);
            return vec_packsu(lo, hi);
        }

        template<bool align, bool first> 
        SIMD_INLINE void TextureBoostedSaturatedGradient(const uint8_t * src, size_t stride, const v128_s16 & saturation, const v128_s16 & boost,
            Storer<align> & dx, Storer<align> & dy)
        {
            const v128_u8 s10 = Load<false>(src - 1);
            const v128_u8 s12 = Load<false>(src + 1);
            const v128_u8 s01 = Load<align>(src - stride);
            const v128_u8 s21 = Load<align>(src + stride);
            Store<align, first>(dx, TextureBoostedSaturatedGradient8(s10, s12, saturation, boost));
            Store<align, first>(dy, TextureBoostedSaturatedGradient8(s01, s21, saturation, boost));
        }

        template<bool align> void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            assert(width >= A && int(2)*saturation*boost <= 0xFF);
            if(align)
            {
                assert(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride));
            }

            size_t alignedWidth = AlignLo(width, A);
            v128_s16 _saturation = SIMD_VEC_SET1_EPI16(saturation);
            v128_s16 _boost = SIMD_VEC_SET1_EPI16(boost);

            memset(dx, 0, width);
            memset(dy, 0, width);
            src += srcStride;
            dx += dxStride;
            dy += dyStride;
            for (size_t row = 2; row < height; ++row)
            {
                Storer<align> _dx(dx), _dy(dy);
                TextureBoostedSaturatedGradient<align, true>(src, srcStride, _saturation, _boost, _dx, _dy);
                for (size_t col = A; col < alignedWidth; col += A)
                    TextureBoostedSaturatedGradient<align, false>(src + col, srcStride, _saturation, _boost, _dx, _dy);
                _dx.Flush();
                _dy.Flush();

                if(width != alignedWidth)
                {
                    Storer<false> _dx(dx + width - A), _dy(dy + width - A);
                    TextureBoostedSaturatedGradient<false, true>(src + width - A, srcStride, _saturation, _boost, _dx, _dy);
                    _dx.Flush();
                    _dy.Flush();
                }

                dx[0] = 0;
                dy[0] = 0;
                dx[width - 1] = 0;
                dy[width - 1] = 0;

                src += srcStride;
                dx += dxStride;
                dy += dyStride;
            }
            memset(dx, 0, width);
            memset(dy, 0, width);
        }

        void TextureBoostedSaturatedGradient(const uint8_t * src, size_t srcStride, size_t width, size_t height, 
            uint8_t saturation, uint8_t boost, uint8_t * dx, size_t dxStride, uint8_t * dy, size_t dyStride)
        {
            if(Aligned(src) && Aligned(srcStride) && Aligned(dx) && Aligned(dxStride) && Aligned(dy) && Aligned(dyStride))
                TextureBoostedSaturatedGradient<true>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
            else
                TextureBoostedSaturatedGradient<false>(src, srcStride, width, height, saturation, boost, dx, dxStride, dy, dyStride);
        }
    }
#endif// SIMD_VSX_ENABLE
}
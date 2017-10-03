/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

namespace Simd
{
#ifdef SIMD_VMX_ENABLE  
    namespace Vmx
    {
        SIMD_INLINE v128_u16 AlphaBlending(const v128_u16 & foreground, const v128_u16 & background, const v128_u16 & alpha)
        {
            return DivideBy255(vec_mladd(foreground, alpha, vec_mladd(background, vec_sub(K16_00FF, alpha), K16_0000)));
        }

        template <bool align, bool first>
        SIMD_INLINE void AlphaBlending(const Loader<align> & foreground, const Loader<align> & background, const v128_u8 & alpha, Storer<align> & dst)
        {
            v128_u8 _foreground = Load<align, first>(foreground);
            v128_u8 _background = Load<align, first>(background);
            v128_u16 lo = AlphaBlending(UnpackLoU8(_foreground), UnpackLoU8(_background), UnpackLoU8(alpha));
            v128_u16 hi = AlphaBlending(UnpackHiU8(_foreground), UnpackHiU8(_background), UnpackHiU8(alpha));
            Store<align, first>(dst, vec_pack(lo, hi));
        }

        template <bool align, bool first, size_t channelCount> struct AlphaBlender
        {
            void operator()(const Loader<align> & foreground, const Loader<align> & background, const v128_u8 & alpha, Storer<align> & dst);
        };

        template <bool align, bool first> struct AlphaBlender<align, first, 1>
        {
            SIMD_INLINE void operator()(const Loader<align> & foreground, const Loader<align> & background, const v128_u8 & alpha, Storer<align> & dst)
            {
                AlphaBlending<align, first>(foreground, background, alpha, dst);
            }
        };

        template <bool align, bool first> struct AlphaBlender<align, first, 2>
        {
            SIMD_INLINE void operator()(const Loader<align> & foreground, const Loader<align> & background, const v128_u8 & alpha, Storer<align> & dst)
            {
                AlphaBlending<align, first>(foreground, background, (v128_u8)UnpackLoU8(alpha, alpha), dst);
                AlphaBlending<align, false>(foreground, background, (v128_u8)UnpackHiU8(alpha, alpha), dst);
            }
        };

        template <bool align, bool first> struct AlphaBlender<align, first, 3>
        {
            SIMD_INLINE void operator()(const Loader<align> & foreground, const Loader<align> & background, const v128_u8 & alpha, Storer<align> & dst)
            {
                AlphaBlending<align, first>(foreground, background, vec_perm(alpha, K8_00, K8_PERM_GRAY_TO_BGR_0), dst);
                AlphaBlending<align, false>(foreground, background, vec_perm(alpha, K8_00, K8_PERM_GRAY_TO_BGR_1), dst);
                AlphaBlending<align, false>(foreground, background, vec_perm(alpha, K8_00, K8_PERM_GRAY_TO_BGR_2), dst);
            }
        };

        template <bool align, bool first> struct AlphaBlender<align, first, 4>
        {
            SIMD_INLINE void operator()(const Loader<align> & foreground, const Loader<align> & background, const v128_u8 & alpha, Storer<align> & dst)
            {
                v128_u8 lo = (v128_u8)UnpackLoU8(alpha, alpha);
                AlphaBlending<align, first>(foreground, background, (v128_u8)UnpackLoU8(lo, lo), dst);
                AlphaBlending<align, false>(foreground, background, (v128_u8)UnpackHiU8(lo, lo), dst);
                v128_u8 hi = (v128_u8)UnpackHiU8(alpha, alpha);
                AlphaBlending<align, false>(foreground, background, (v128_u8)UnpackLoU8(hi, hi), dst);
                AlphaBlending<align, false>(foreground, background, (v128_u8)UnpackHiU8(hi, hi), dst);
            }
        };

        template <bool align, size_t channelCount> void AlphaBlending(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * alpha, size_t alphaStride, uint8_t * dst, size_t dstStride)
        {
            size_t alignedWidth = AlignLo(width, A);
            v128_u8 tailMask = ShiftLeft(K8_FF, A - width + alignedWidth);
            size_t step = channelCount*A;
            for (size_t row = 0; row < height; ++row)
            {
                Loader<align> foreground(src), background(dst);
                Storer<align> _dst(dst);
                AlphaBlender<align, true, channelCount>()(foreground, background, Load<align>(alpha), _dst);
                for (size_t col = A, offset = 0; col < alignedWidth; col += A, offset += step)
                    AlphaBlender<align, false, channelCount>()(foreground, background, Load<align>(alpha + col), _dst);
                Flush(_dst);
                if (alignedWidth != width)
                {
                    size_t offset = +(width - A)*channelCount;
                    Loader<false> foreground(src + offset), background(dst + offset);
                    Storer<false> _dst(dst + offset);
                    v128_u8 _alpha = vec_and(Load<false>(alpha + width - A), tailMask);
                    AlphaBlender<false, true, channelCount>()(foreground, background, _alpha, _dst);
                    Flush(_dst);
                }
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        template <bool align> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            assert(width >= A);
            if (align)
            {
                assert(Aligned(src) && Aligned(srcStride));
                assert(Aligned(alpha) && Aligned(alphaStride));
                assert(Aligned(dst) && Aligned(dstStride));
            }

            switch (channelCount)
            {
            case 1: AlphaBlending<align, 1>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 2: AlphaBlending<align, 2>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 3: AlphaBlending<align, 3>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 4: AlphaBlending<align, 4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            default:
                assert(0);
            }
        }

        void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            if (Aligned(src) && Aligned(srcStride) && Aligned(alpha) && Aligned(alphaStride) && Aligned(dst) && Aligned(dstStride))
                AlphaBlending<true>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
            else
                AlphaBlending<false>(src, srcStride, width, height, channelCount, alpha, alphaStride, dst, dstStride);
        }
    }
#endif// SIMD_VMX_ENABLE
}

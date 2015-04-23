/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
#include "Simd/SimdMath.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int AlphaBlending(int src, int dst, int alpha)
        {
            return DivideBy255(src*alpha + dst*(0xFF - alpha));
        }

        template <size_t channelCount> void AlphaBlending(const uint8_t *src, int alpha, uint8_t *dst);

        template <> SIMD_INLINE void AlphaBlending<1>(const uint8_t *src, int alpha, uint8_t *dst)
        {
            dst[0] = AlphaBlending(src[0], dst[0], alpha);
        }

        template <> SIMD_INLINE void AlphaBlending<2>(const uint8_t *src, int alpha, uint8_t *dst)
        {
            dst[0] = AlphaBlending(src[0], dst[0], alpha);
            dst[1] = AlphaBlending(src[1], dst[1], alpha);
        }

        template <> SIMD_INLINE void AlphaBlending<3>(const uint8_t *src, int alpha, uint8_t *dst)
        {
            dst[0] = AlphaBlending(src[0], dst[0], alpha);
            dst[1] = AlphaBlending(src[1], dst[1], alpha);
            dst[2] = AlphaBlending(src[2], dst[2], alpha);
        }

        template <> SIMD_INLINE void AlphaBlending<4>(const uint8_t *src, int alpha, uint8_t *dst)
        {
            dst[0] = AlphaBlending(src[0], dst[0], alpha);
            dst[1] = AlphaBlending(src[1], dst[1], alpha);
            dst[2] = AlphaBlending(src[2], dst[2], alpha);
            dst[3] = AlphaBlending(src[3], dst[3], alpha);
        }

        template <size_t channelCount> void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, 
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            for(size_t row = 0; row < height; ++row)
            {
                for(size_t col = 0, offset = 0; col < width; ++col, offset += channelCount)
                    AlphaBlending<channelCount>(src + offset, alpha[col], dst + offset);
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        void AlphaBlending(const uint8_t *src, size_t srcStride, size_t width, size_t height, size_t channelCount, 
            const uint8_t *alpha, size_t alphaStride, uint8_t *dst, size_t dstStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            switch(channelCount)
            {
            case 1 : AlphaBlending<1>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 2 : AlphaBlending<2>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 3 : AlphaBlending<3>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 4 : AlphaBlending<4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            }
        }
    }
}
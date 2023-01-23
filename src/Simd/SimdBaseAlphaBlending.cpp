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

#include "Simd/SimdAlphaBlending.h"
#include "Simd/SimdYuvToBgr.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int AlphaBlending(int src, int dst, int alpha)
        {
            return DivideBy255(src*alpha + dst*(0xFF - alpha));
        }

        template <size_t channelCount> void AlphaBlending(const uint8_t * src, int alpha, uint8_t * dst);

        template <> SIMD_INLINE void AlphaBlending<1>(const uint8_t * src, int alpha, uint8_t * dst)
        {
            dst[0] = AlphaBlending(src[0], dst[0], alpha);
        }

        template <> SIMD_INLINE void AlphaBlending<2>(const uint8_t * src, int alpha, uint8_t * dst)
        {
            dst[0] = AlphaBlending(src[0], dst[0], alpha);
            dst[1] = AlphaBlending(src[1], dst[1], alpha);
        }

        template <> SIMD_INLINE void AlphaBlending<3>(const uint8_t * src, int alpha, uint8_t * dst)
        {
            dst[0] = AlphaBlending(src[0], dst[0], alpha);
            dst[1] = AlphaBlending(src[1], dst[1], alpha);
            dst[2] = AlphaBlending(src[2], dst[2], alpha);
        }

        template <> SIMD_INLINE void AlphaBlending<4>(const uint8_t * src, int alpha, uint8_t * dst)
        {
            dst[0] = AlphaBlending(src[0], dst[0], alpha);
            dst[1] = AlphaBlending(src[1], dst[1], alpha);
            dst[2] = AlphaBlending(src[2], dst[2], alpha);
            dst[3] = AlphaBlending(src[3], dst[3], alpha);
        }

        template <size_t channelCount> void AlphaBlending(const uint8_t * src, size_t srcStride, size_t width, size_t height,
            const uint8_t * alpha, size_t alphaStride, uint8_t * dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += channelCount)
                    AlphaBlending<channelCount>(src + offset, alpha[col], dst + offset);
                src += srcStride;
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        void AlphaBlending(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t channelCount,
            const uint8_t * alpha, size_t alphaStride, uint8_t * dst, size_t dstStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: AlphaBlending<1>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 2: AlphaBlending<2>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 3: AlphaBlending<3>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            case 4: AlphaBlending<4>(src, srcStride, width, height, alpha, alphaStride, dst, dstStride); break;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <size_t channelCount> void AlphaBlending2x(const uint8_t* src0, int alpha0, const uint8_t* src1, int alpha1, uint8_t* dst);

        template <> SIMD_INLINE void AlphaBlending2x<1>(const uint8_t* src0, int alpha0, const uint8_t* src1, int alpha1, uint8_t* dst)
        {
            dst[0] = AlphaBlending(src1[0], AlphaBlending(src0[0], dst[0], alpha0), alpha1);
        }

        template <> SIMD_INLINE void AlphaBlending2x<2>(const uint8_t* src0, int alpha0, const uint8_t* src1, int alpha1, uint8_t* dst)
        {
            dst[0] = AlphaBlending(src1[0], AlphaBlending(src0[0], dst[0], alpha0), alpha1);
            dst[1] = AlphaBlending(src1[1], AlphaBlending(src0[1], dst[1], alpha0), alpha1);
        }

        template <> SIMD_INLINE void AlphaBlending2x<3>(const uint8_t* src0, int alpha0, const uint8_t* src1, int alpha1, uint8_t* dst)
        {
            dst[0] = AlphaBlending(src1[0], AlphaBlending(src0[0], dst[0], alpha0), alpha1);
            dst[1] = AlphaBlending(src1[1], AlphaBlending(src0[1], dst[1], alpha0), alpha1);
            dst[2] = AlphaBlending(src1[2], AlphaBlending(src0[2], dst[2], alpha0), alpha1);
        }

        template <> SIMD_INLINE void AlphaBlending2x<4>(const uint8_t* src0, int alpha0, const uint8_t* src1, int alpha1, uint8_t* dst)
        {
            dst[0] = AlphaBlending(src1[0], AlphaBlending(src0[0], dst[0], alpha0), alpha1);
            dst[1] = AlphaBlending(src1[1], AlphaBlending(src0[1], dst[1], alpha0), alpha1);
            dst[2] = AlphaBlending(src1[2], AlphaBlending(src0[2], dst[2], alpha0), alpha1);
            dst[3] = AlphaBlending(src1[3], AlphaBlending(src0[3], dst[3], alpha0), alpha1);
        }

        template <size_t channelCount> void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += channelCount)
                    AlphaBlending2x<channelCount>(src0 + offset, alpha0[col], src1 + offset, alpha1[col], dst + offset);
                src0 += src0Stride;
                alpha0 += alpha0Stride;
                src1 += src1Stride;
                alpha1 += alpha1Stride;
                dst += dstStride;
            }
        }

        void AlphaBlending2x(const uint8_t* src0, size_t src0Stride, const uint8_t* alpha0, size_t alpha0Stride,
            const uint8_t* src1, size_t src1Stride, const uint8_t* alpha1, size_t alpha1Stride,
            size_t width, size_t height, size_t channelCount, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: AlphaBlending2x<1>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 2: AlphaBlending2x<2>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 3: AlphaBlending2x<3>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            case 4: AlphaBlending2x<4>(src0, src0Stride, alpha0, alpha0Stride, src1, src1Stride, alpha1, alpha1Stride, width, height, dst, dstStride); break;
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <class T> SIMD_INLINE void AlphaBlendingBgraToYuv420p(const uint8_t* bgra0, size_t bgraStride, uint8_t* y0, size_t yStride, uint8_t* u, uint8_t* v)
        {
            const uint8_t* bgra1 = bgra0 + bgraStride;
            uint8_t* y1 = y0 + yStride;

            y0[0] = AlphaBlending(BgrToY<T>(bgra0[0], bgra0[1], bgra0[2]), y0[0], bgra0[3]);
            y0[1] = AlphaBlending(BgrToY<T>(bgra0[4], bgra0[5], bgra0[6]), y0[1], bgra0[7]);
            y1[0] = AlphaBlending(BgrToY<T>(bgra1[0], bgra1[1], bgra1[2]), y1[0], bgra1[3]);
            y1[1] = AlphaBlending(BgrToY<T>(bgra1[4], bgra1[5], bgra1[6]), y1[1], bgra1[7]);

            int b = Average(bgra0[0], bgra0[4], bgra1[0], bgra1[4]);
            int g = Average(bgra0[1], bgra0[5], bgra1[1], bgra1[5]);
            int r = Average(bgra0[2], bgra0[6], bgra1[2], bgra1[6]);
            int a = Average(bgra0[3], bgra0[7], bgra1[3], bgra1[7]);
            u[0] = AlphaBlending(BgrToU<T>(b, g, r), u[0], a);
            v[0] = AlphaBlending(BgrToV<T>(b, g, r), v[0], a);
        }

        template <class T> void AlphaBlendingBgraToYuv420p(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride)
        {
            assert((width % 2 == 0) && (height % 2 == 0) && (width >= 2) && (height >= 2));

            for (size_t row = 0; row < height; row += 2)
            {
                for (size_t colUV = 0, colY = 0, colBgra = 0; colY < width; colY += 2, colUV++, colBgra += 8)
                    AlphaBlendingBgraToYuv420p<T>(bgra + colBgra, bgraStride, y + colY, yStride, u + colUV, v + colUV);
                bgra += 2 * bgraStride;
                y += 2 * yStride;
                u += uStride;
                v += vStride;
            }
        }

        void AlphaBlendingBgraToYuv420p(const uint8_t* bgra, size_t bgraStride, size_t width, size_t height,
            uint8_t* y, size_t yStride, uint8_t* u, size_t uStride, uint8_t* v, size_t vStride, SimdYuvType yuvType)
        {
            switch (yuvType)
            {
            case SimdYuvBt601: AlphaBlendingBgraToYuv420p<Base::Bt601>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt709: AlphaBlendingBgraToYuv420p<Base::Bt709>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvBt2020: AlphaBlendingBgraToYuv420p<Base::Bt2020>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            case SimdYuvTrect871: AlphaBlendingBgraToYuv420p<Base::Trect871>(bgra, bgraStride, width, height, y, yStride, u, uStride, v, vStride); break;
            default:
                assert(0);
            }
        }

        //-------------------------------------------------------------------------------------------------

        template <size_t channelCount> void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t alpha, uint8_t* dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += channelCount)
                    AlphaBlending<channelCount>(src + offset, alpha, dst + offset);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaBlendingUniform(const uint8_t* src, size_t srcStride, size_t width, size_t height, size_t channelCount, uint8_t alpha, uint8_t* dst, size_t dstStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: AlphaBlendingUniform<1>(src, srcStride, width, height, alpha, dst, dstStride); break;
            case 2: AlphaBlendingUniform<2>(src, srcStride, width, height, alpha, dst, dstStride); break;
            case 3: AlphaBlendingUniform<3>(src, srcStride, width, height, alpha, dst, dstStride); break;
            case 4: AlphaBlendingUniform<4>(src, srcStride, width, height, alpha, dst, dstStride); break;
            }
        }

        //-----------------------------------------------------------------------------------------

        template <size_t channelCount> void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, const uint8_t * alpha, size_t alphaStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, offset = 0; col < width; ++col, offset += channelCount)
                    AlphaBlending<channelCount>(channel, alpha[col], dst + offset);
                alpha += alphaStride;
                dst += dstStride;
            }
        }

        void AlphaFilling(uint8_t * dst, size_t dstStride, size_t width, size_t height, const uint8_t * channel, size_t channelCount, const uint8_t * alpha, size_t alphaStride)
        {
            assert(channelCount >= 1 && channelCount <= 4);

            switch (channelCount)
            {
            case 1: AlphaFilling<1>(dst, dstStride, width, height, channel, alpha, alphaStride); break;
            case 2: AlphaFilling<2>(dst, dstStride, width, height, channel, alpha, alphaStride); break;
            case 3: AlphaFilling<3>(dst, dstStride, width, height, channel, alpha, alphaStride); break;
            case 4: AlphaFilling<4>(dst, dstStride, width, height, channel, alpha, alphaStride); break;
            }
        }

        //-----------------------------------------------------------------------------------------

        template<bool argb> void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, end = width*4; col < end; col += 4)
                    AlphaPremultiply<argb>(src + col, dst + col);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaPremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdBool argb)
        {
            if (argb)
                AlphaPremultiply<true>(src, srcStride, width, height, dst, dstStride);
            else
                AlphaPremultiply<false>(src, srcStride, width, height, dst, dstStride);
        }

        //-----------------------------------------------------------------------------------------

        template<bool argb> void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride)
        {
            for (size_t row = 0; row < height; ++row)
            {
                for (size_t col = 0, end = width * 4; col < end; col += 4)
                    AlphaUnpremultiply<argb>(src + col, dst + col);
                src += srcStride;
                dst += dstStride;
            }
        }

        void AlphaUnpremultiply(const uint8_t* src, size_t srcStride, size_t width, size_t height, uint8_t* dst, size_t dstStride, SimdBool argb)
        {
            if (argb)
                AlphaUnpremultiply<true>(src, srcStride, width, height, dst, dstStride);
            else
                AlphaUnpremultiply<false>(src, srcStride, width, height, dst, dstStride);
        }
    }
}

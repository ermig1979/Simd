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
#ifndef __SimdWarpAffine_h__
#define __SimdWarpAffine_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"

namespace Simd
{
    struct WarpAffParam
    {
        static const int BorderSizeMax = 4 * 1;

        SimdWarpAffineFlags flags;
        float mat[6];
        uint8_t border[BorderSizeMax];
        size_t srcW, srcH, dstW, dstH, channels, align;

        WarpAffParam(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float * mat, SimdWarpAffineFlags flags, const uint8_t * border, size_t align)
        {
            this->srcW = srcW;
            this->srcH = srcH;
            this->dstW = dstW;
            this->dstH = dstH;
            this->channels = channels;
            memcpy(this->mat, mat, 6 * sizeof(float));
            this->flags = flags;
            if (border && (flags & SimdWarpAffineBorderMask) == SimdWarpAffineInterpBilinear)
                memcpy(this->border, border, this->ChannelSize());
            else
                memset(this->border, 0, BorderSizeMax);
            this->align = align;
        }

        bool Valid() const
        {
            return true;
        }

        bool IsNearest() const
        {
            return (flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpNearest;
        }

        bool IsByteBilinear() const
        {
            return (flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpBilinear && (SimdWarpAffineChannelMask & flags) == SimdWarpAffineChannelByte;
        }

        size_t ChannelSize() const
        {
            switch (SimdWarpAffineChannelMask & flags)
            {
            case SimdWarpAffineChannelByte: return 1;
            default:
                assert(0); return 0;
            }
        }

        size_t PixelSize() const
        {
            return ChannelSize() * channels;
        }
    };

    //---------------------------------------------------------------------------------------------

    class WarpAffine : Deletable
    {
    public:
        WarpAffine(const WarpAffParam & param)
            : _param(param)
        {
        }

        virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) = 0;

    protected:
        WarpAffParam _param;
    };

    //---------------------------------------------------------------------------------------------

    namespace Base
    {
        class WarpAffineNearest : public WarpAffine
        {
        public:
            WarpAffineNearest(const WarpAffParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        class WarpAffineByteBilinear : public WarpAffine
        {
        public:
            WarpAffineByteBilinear(const WarpAffParam & param);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride);
        };

        //---------------------------------------------------------------------------------------------

        void * WarpAffineInit(size_t srcW, size_t srcH, size_t dstW, size_t dstH, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border);
    }
}
#endif//__SimdWarpAffine_h__

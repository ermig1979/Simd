/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#ifndef __SimdResizer_h__
#define __SimdResizer_h__

#include "Simd/SimdArray.h"

namespace Simd
{
    class Resizer : Deletable
    {
        SimdResizeChannelType _type;
        SimdResizeMethodType _method;

    public:
        Resizer(SimdResizeChannelType type, SimdResizeMethodType method)
            : _type(type)
            , _method(method)
        {
        }

        SimdResizeChannelType Type() const { return _type; }
        SimdResizeMethodType Method() const { return _method; }

        virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) const = 0;
    };

    namespace Base
    {
        class ResizerByteBilinear : Resizer
        {
            size_t _sx, _sy, _dx, _dy, _cn, _rs;
            Array32i _ax, _ix, _ay, _iy;
        public:
            ResizerByteBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels);

            static void EstimateIndexAlpha(size_t srcSize, size_t dstSize, int32_t * indices, int32_t * alphas, size_t channels);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) const;
        };

        class ResizerFloatBilinear : Resizer
        {
        protected:
            size_t _sx, _sy, _dx, _dy, _cn, _rs;
            Array32i _ix, _iy;
            Array32f _ax, _ay;

            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride) const;

        public:
            ResizerFloatBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, size_t align, bool caffeInterp);

            virtual void Run(const uint8_t * src, size_t srcStride, uint8_t * dst, size_t dstStride) const;

            static void EstimateIndexAlpha(size_t srcSize, size_t dstSize, int32_t * indices, float * alphas, size_t channels, bool caffeInterp);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }

#ifdef SIMD_SSE_ENABLE    
    namespace Sse
    {
        class ResizerFloatBilinear : Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride) const;
        public:
            ResizerFloatBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, bool caffeInterp);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_SSE_ENABLE 

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        class ResizerFloatBilinear : Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride) const;
        public:
            ResizerFloatBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, bool caffeInterp);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX_ENABLE 

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class ResizerFloatBilinear : Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride) const;
        public:
            ResizerFloatBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, bool caffeInterp);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX2_ENABLE 

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        class ResizerFloatBilinear : Base::ResizerFloatBilinear
        {
            virtual void Run(const float * src, size_t srcStride, float * dst, size_t dstStride) const;
        public:
            ResizerFloatBilinear(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, bool caffeInterp);
        };

        void * ResizerInit(size_t srcX, size_t srcY, size_t dstX, size_t dstY, size_t channels, SimdResizeChannelType type, SimdResizeMethodType method);
    }
#endif //SIMD_AVX512F_ENABLE 
}
#endif//__SimdResizer_h__

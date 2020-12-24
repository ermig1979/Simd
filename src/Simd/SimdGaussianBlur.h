/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#ifndef __SimdGaussianBlur_h__
#define __SimdGaussianBlur_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"

namespace Simd
{
    struct BlurParam
    {
        size_t width;
        size_t height;
        size_t channels;
        float radius;
        size_t align;

        BlurParam(size_t w, size_t h, size_t c, const float* r, size_t a);
    };

    namespace Base
    {
        class GaussianBlur : Deletable
        {
        public:
            GaussianBlur(const BlurParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);

        protected:
            BlurParam _param;
            size_t _half, _kernel, _edge, _start, _size, _stride;
            Array8u _buf;
            Array32f _weight, _rows;
        };

        void * GaussianBlurInit(size_t width, size_t height, size_t channels, const float* radius);
    }

//#ifdef SIMD_SSE41_ENABLE    
//    namespace Sse41
//    {
//        class GaussianBlur : Simd::GaussianBlur
//        {
//        public:
//            GaussianBlur(const BlurParam& param);
//
//            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
//        };
//
//        void* GaussianBlurInit(size_t width, size_t height, size_t channels, const float* radius);
//    }
//#endif //SIMD_SSE41_ENABLE
}
#endif//__SimdGaussianBlur_h__

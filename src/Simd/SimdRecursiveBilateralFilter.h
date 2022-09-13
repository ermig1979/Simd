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
#ifndef __SimdRecursiveBilateralFilter_h__
#define __SimdRecursiveBilateralFilter_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"
#include "Simd/SimdCopyPixel.h"

namespace Simd
{
    struct RbfParam
    {
        size_t width;
        size_t height;
        size_t channels;
        float spatial;
        float range;
        size_t align;

        RbfParam(size_t w, size_t h, size_t c, const float* s, const float * r, size_t a);
        bool Valid() const;
    };

    class RecursiveBilateralFilter : Deletable
    {
    public:
        RecursiveBilateralFilter(const RbfParam& param);

        virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride) = 0;

    protected:
        RbfParam _param;
    };

    namespace Base
    {
        struct RbfAlg
        {
            float alpha;
            Array32f ranges;
            Array32f fb0, cb0, fb1, cb1;
        };

        class RecursiveBilateralFilterDefault : public Simd::RecursiveBilateralFilter
        {
        public:
            RecursiveBilateralFilterDefault(const RbfParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);

        protected:
            typedef void (*FilterPtr)(const RbfParam& p, RbfAlg& a, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);
            RbfAlg _alg;
            FilterPtr _hFilter, _vFilter;

            void InitAlg();
            void InitBuf();
        };

        void * RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class RecursiveBilateralFilterDefault : public Base::RecursiveBilateralFilterDefault
        {
        public:
            RecursiveBilateralFilterDefault(const RbfParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);

        protected:
        };

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
    }
#endif
}
#endif//__SimdRecursiveBilateralFilter_h__

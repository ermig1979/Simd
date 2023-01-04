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
    enum RbfDiffType
    {
        RbfDiffAvg = SimdRecursiveBilateralFilterDiffAvg,
        RbfDiffMax = SimdRecursiveBilateralFilterDiffMax,
        RbfDiffSum = SimdRecursiveBilateralFilterDiffSum,
    };    
    
    SIMD_INLINE bool Precise(SimdRecursiveBilateralFilterFlags flags)
    {
        return flags & SimdRecursiveBilateralFilterPrecise;
    }

    SIMD_INLINE bool FmaAvoid(SimdRecursiveBilateralFilterFlags flags)
    {
        return flags & SimdRecursiveBilateralFilterFmaAvoid;
    }

    SIMD_INLINE RbfDiffType DiffType(SimdRecursiveBilateralFilterFlags flags)
    {
        return (RbfDiffType)(flags & SimdRecursiveBilateralFilterDiffMask);
    }

    //-----------------------------------------------------------------------------------------

    struct RbfParam
    {
        size_t width;
        size_t height;
        size_t channels;
        float spatial;
        float range;
        SimdRecursiveBilateralFilterFlags flags;
        size_t align;

        RbfParam(size_t w, size_t h, size_t c, const float* s, const float * r, SimdRecursiveBilateralFilterFlags f, size_t a);
        bool Valid() const;

        float alpha;
        float ranges[256];

        void Init();
    };

    //-----------------------------------------------------------------------------------------

    class RecursiveBilateralFilter : Deletable
    {
    public:
        RecursiveBilateralFilter(const RbfParam& param);

        virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride) = 0;

        size_t BufferSize() const
        {
            return _buffer.RawSize();
        }

        typedef void (*FilterPtr)(const RbfParam& p, float* buf, const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);

    protected:
        RbfParam _param;
        Array8u _buffer;
        FilterPtr _hFilter, _vFilter;
    };

    //-----------------------------------------------------------------------------------------

    namespace Base
    {
        template<RbfDiffType type> SIMD_INLINE int Diff(int ch0, int ch1)
        {
            switch (type)
            {
            case RbfDiffAvg: return Average(ch0, ch1);
            case RbfDiffMax: return Max(ch0, ch1);
            case RbfDiffSum: return Max(ch0 + ch1, 255);
            default:
                assert(0); return 0;
            }
        }

        template<RbfDiffType type> SIMD_INLINE int Diff(int ch0, int ch1, int ch2)
        {
            switch (type)
            {
            case RbfDiffAvg: return Average(ch1, Average(ch0, ch2));
            case RbfDiffMax: return Max(Max(ch0, ch1), ch2);
            case RbfDiffSum: return Max(ch0 + ch1 + ch2, 255);
            default:
                assert(0); return 0;
            }
        }

        template<int channels, RbfDiffType type> SIMD_INLINE int Diff(const uint8_t* src1, const uint8_t* src2)
        {
            switch (channels)
            {
            case 1: return ::abs(src1[0] - src2[0]);
            case 2: return Diff<type>(::abs(src1[0] - src2[0]), ::abs(src1[1] - src2[1]));
            case 3: 
            case 4: return Diff<type>(::abs(src1[0] - src2[0]), ::abs(src1[1] - src2[1]), ::abs(src1[2] - src2[2]));
            default:
                assert(0); return 0;
            }
        }

        //-----------------------------------------------------------------------------------------

        class RecursiveBilateralFilterPrecize : public Simd::RecursiveBilateralFilter
        {
        public:
            RecursiveBilateralFilterPrecize(const RbfParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);

        protected:
            float* GetBuffer();
        };

        class RecursiveBilateralFilterFast : public Simd::RecursiveBilateralFilter
        {
        public:
            RecursiveBilateralFilterFast(const RbfParam& param);

            virtual void Run(const uint8_t* src, size_t srcStride, uint8_t* dst, size_t dstStride);

        protected:
            uint8_t* GetBuffer();
        private:
            size_t _size, _stride;
        };

        void * RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange, SimdRecursiveBilateralFilterFlags flags);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class RecursiveBilateralFilterPrecize : public Base::RecursiveBilateralFilterPrecize
        {
        public:
            RecursiveBilateralFilterPrecize(const RbfParam& param);
        };

        class RecursiveBilateralFilterFast : public Base::RecursiveBilateralFilterFast
        {
        public:
            RecursiveBilateralFilterFast(const RbfParam& param);
        };

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange, SimdRecursiveBilateralFilterFlags flags);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class RecursiveBilateralFilterFast : public Sse41::RecursiveBilateralFilterFast
        {
        public:
            RecursiveBilateralFilterFast(const RbfParam& param);
        };

        void* RecursiveBilateralFilterInit(size_t width, size_t height, size_t channels, const float* sigmaSpatial, const float* sigmaRange, SimdRecursiveBilateralFilterFlags flags);
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
#endif

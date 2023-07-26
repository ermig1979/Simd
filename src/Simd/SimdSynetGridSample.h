/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#ifndef __SimdSynetGridSample_h__
#define __SimdSynetGridSample_h__

#include "Simd/SimdMath.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdArray.h"

namespace Simd
{
    struct GridSample2dParam
    {
        size_t batch, channels, srcH, srcW, dstH, dstW;
        SimdTensorDataType type;
        SimdGridSampleInterpType interp;
        SimdGridSamplePaddingType padding;
        SimdBool align;

        SIMD_INLINE GridSample2dParam(size_t b, size_t c, size_t sh, size_t sw, size_t dh, size_t dw,
            SimdTensorDataType t, SimdGridSampleInterpType i, SimdGridSamplePaddingType p, SimdBool a)
            : batch(b)
            , channels(c)
            , srcH(sh)
            , srcW(sw)
            , dstH(dh)
            , dstW(dw)
            , type(t)
            , interp(i)
            , padding(p)
            , align(a)
        {
        }

        SIMD_INLINE bool Valid() const
        {
            return true;
        }

        bool Is32fBlZ() const
        {
            return type == SimdTensorData32f && interp == SimdGridSampleInterpBilinear && padding == SimdGridSamplePaddingZeros;
        }
    };

    //-------------------------------------------------------------------------------------------------

    class SynetGridSample2d : public Deletable
    {
    public:
        SynetGridSample2d(const GridSample2dParam& param)
            : _param(param)
        {
        }

        virtual size_t InternalBufferSize() const
        {
            return 0;
        }

        virtual void Forward(const uint8_t* src, const uint8_t* grd, uint8_t* dst) = 0;

    protected:
        GridSample2dParam _param;
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetGridSample2dRef : public Simd::SynetGridSample2d
        {
        public:
            SynetGridSample2dRef(const GridSample2dParam & param);

            virtual void Forward(const uint8_t* src, const uint8_t* grd, uint8_t* dst);

            typedef void (*GridSample2dPtr)(const uint8_t* src8, size_t batch, size_t channels, size_t srcH, size_t srcW, const uint8_t* grd8, size_t dstH, size_t dstW, uint8_t* dst8);

        protected:
            GridSample2dPtr _gridSample2d;
        };

        class SynetGridSample2d32fBlZ : public Simd::SynetGridSample2d
        {
        public:
            SynetGridSample2d32fBlZ(const GridSample2dParam& param);

            virtual size_t InternalBufferSize() const;

            virtual void Forward(const uint8_t* src, const uint8_t* grd, uint8_t* dst);

            typedef void (*IndexCoeffsPtr)(const float* grd, size_t dstS, int srcH, int srcW, int padW, uint32_t* idx, float * dy, float *dx);
            typedef void (*BilinearInterpPtr)(const float* pad, size_t dstS, int padW, uint32_t* idx, float * dy, float* dx, float * dst);

        protected:
            Array32f _padded, _coeffs;
            Array32u _index;
            size_t _padH, _padW, _srcS, _dstS;
            IndexCoeffsPtr _indexCoeffs;
            BilinearInterpPtr _bilinearInterp;
        };

        //-------------------------------------------------------------------------------------------------

        void * SynetGridSample2dInit(size_t batch, size_t channels, size_t srcH, size_t srcW, size_t dstH, size_t dstW,
            SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetGridSample2d32fBlZ : public Base::SynetGridSample2d32fBlZ
        {
        public:
            SynetGridSample2d32fBlZ(const GridSample2dParam& param);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetGridSample2dInit(size_t batch, size_t channels, size_t srcH, size_t srcW, size_t dstH, size_t dstW,
            SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetGridSample2d32fBlZ : public Sse41::SynetGridSample2d32fBlZ
        {
        public:
            SynetGridSample2d32fBlZ(const GridSample2dParam& param);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetGridSample2dInit(size_t batch, size_t channels, size_t srcH, size_t srcW, size_t dstH, size_t dstW,
            SimdTensorDataType type, SimdGridSampleInterpType interp, SimdGridSamplePaddingType padding, SimdBool align);
    }
#endif
}

#endif

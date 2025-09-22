/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#ifndef __SimdSynetQuantizedAdd_h__
#define __SimdSynetQuantizedAdd_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"

#include <vector>

namespace Simd
{
    typedef std::vector<size_t> Shape;

    struct QuantizedAddParam
    {
        Shape aShape, bShape;
        SimdTensorDataType aType, bType, dType;
        float aScale, bScale, dScale, actParams[2];
        size_t aZero, bZero, dZero;
        SimdConvolutionActivationType actType;

        QuantizedAddParam(
            const size_t* as, size_t ac, SimdTensorDataType at, const float* aSc, int32_t aZr, 
            const size_t* bs, size_t bc, SimdTensorDataType bt, const float* bSc, int32_t bZr,
            SimdConvolutionActivationType act, const float *ap, SimdTensorDataType dt, const float* dSc, int32_t dZr)
            : aShape(as, as + ac)
            , aType(at)
            , aScale(aSc ? *aSc : 1.0f)
            , aZero(aZr)
            , bShape(bs, bs + bc)
            , bType(bt)
            , bScale(bSc ? *bSc : 1.0f)
            , bZero(bZr)
            , actType(act)
            , dType(dt)
            , dScale(dSc ? *dSc : 1.0f)
            , dZero(dZr)
        {
            actParams[0] = ap ? ap[0] : 0.0f;
            actParams[1] = ap ? ap[1] : 0.0f;
        }

        bool Valid()
        {
            return
                (aType == SimdTensorData32f || aType == SimdTensorData8u) &&
                (bType == SimdTensorData32f || bType == SimdTensorData8u) &&
                (dType == SimdTensorData32f || dType == SimdTensorData8u);
        }
    };

    class SynetQuantizedAdd : public Deletable
    {
    public:
        SynetQuantizedAdd(const QuantizedAddParam& p);

        virtual void Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst) = 0;

    protected:
        QuantizedAddParam _param;
    };

    //------------------------------------------------------------------------------------------------

    namespace Base
    {
        //------------------------------------------------------------------------------------------------

        class SynetQuantizedAddUniform : public SynetQuantizedAdd
        {
        public:
            SynetQuantizedAddUniform(const QuantizedAddParam& p);

            static bool Preferable(const QuantizedAddParam& p);

            virtual void Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst);

            typedef void(*UniformPtr)(const uint8_t* a8, float aScale, int aZero, const uint8_t* b8, float bScale, int bZero, size_t size, const float* params, float dScale, int dZero, uint8_t* dst8);

        protected:
            size_t _size;
            UniformPtr _uniform;
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedAddInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero,
            SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetQuantizedAddUniform : public Base::SynetQuantizedAddUniform
        {
        public:
            SynetQuantizedAddUniform(const QuantizedAddParam& p);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedAddInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero,
            SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetQuantizedAddUniform : public Sse41::SynetQuantizedAddUniform
        {
        public:
            SynetQuantizedAddUniform(const QuantizedAddParam& p);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedAddInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero,
            SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetQuantizedAddUniform : public Avx2::SynetQuantizedAddUniform
        {
        public:
            SynetQuantizedAddUniform(const QuantizedAddParam& p);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedAddInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero,
            SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }
#endif
}

#endif

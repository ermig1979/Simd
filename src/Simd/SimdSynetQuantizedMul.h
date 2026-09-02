/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#ifndef __SimdSynetQuantizedMul_h__
#define __SimdSynetQuantizedMul_h__

#include "Simd/SimdShape.h"
#include "Simd/SimdMemory.h"

namespace Simd
{
    struct QuantizedMulParam
    {
        Shape aShape, bShape;
        SimdTensorDataType aType, bType, dType;
        float aScale, bScale, dScale;
        size_t aZero, bZero, dZero;

        QuantizedMulParam(const size_t* as, size_t ac, SimdTensorDataType at, const float* aSc, int32_t aZr, 
            const size_t* bs, size_t bc, SimdTensorDataType bt, const float* bSc, int32_t bZr, SimdTensorDataType dt, const float* dSc, int32_t dZr)
            : aShape(as, as + ac)
            , aType(at)
            , aScale(aSc ? *aSc : 1.0f)
            , aZero(aZr)
            , bShape(bs, bs + bc)
            , bType(bt)
            , bScale(bSc ? *bSc : 1.0f)
            , bZero(bZr)
            , dType(dt)
            , dScale(dSc ? *dSc : 1.0f)
            , dZero(dZr)
        {
        }

        bool Valid()
        {
            return
                (aType == SimdTensorData32f || aType == SimdTensorData8u) &&
                (bType == SimdTensorData32f || bType == SimdTensorData8u) &&
                (dType == SimdTensorData32f || dType == SimdTensorData8u) &&
                IsCompatible(aShape, bShape);
        }
    };

    //------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetQuantizedMul : public Deletable
        {
        public:
            SynetQuantizedMul(const QuantizedMulParam& p);

            virtual void Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst) = 0;

        protected:
            QuantizedMulParam _param;
        };

        class SynetQuantizedMulUniversal : public SynetQuantizedMul
        {
        public:
            SynetQuantizedMulUniversal(const QuantizedMulParam& p);

            static bool Preferable(const QuantizedMulParam& p);

            virtual void Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst);

            typedef void(*UniversalPtr)(const uint8_t* a8, const size_t* aSteps, float aScale, int aZero,
                const uint8_t* b8, const size_t* bSteps, float bScale, int bZero, uint8_t* dst8, const size_t* dstShape, float dScale, int dZero);

        protected:
            Shape _aSteps, _bSteps, _dShape;
            UniversalPtr _universal;
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetQuantizedMulUniversal : public Base::SynetQuantizedMulUniversal
        {
        public:
            SynetQuantizedMulUniversal(const QuantizedMulParam& p);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetQuantizedMulUniversal : public Sse41::SynetQuantizedMulUniversal
        {
        public:
            SynetQuantizedMulUniversal(const QuantizedMulParam& p);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetQuantizedMulUniversal : public Avx2::SynetQuantizedMulUniversal
        {
        public:
            SynetQuantizedMulUniversal(const QuantizedMulParam& p);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }
#endif

#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        class SynetQuantizedMulUniversal : public Base::SynetQuantizedMulUniversal
        {
        public:
            SynetQuantizedMulUniversal(const QuantizedMulParam& p);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }
#endif

#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        class SynetQuantizedMulUniversal : public Base::SynetQuantizedMulUniversal
        {
        public:
            SynetQuantizedMulUniversal(const QuantizedMulParam& p);
        };

        //------------------------------------------------------------------------------------------------

        void* SynetQuantizedMulInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const float* aScale, int32_t aZero,
            const size_t* bShape, size_t bCount, SimdTensorDataType bType, const float* bScale, int32_t bZero, SimdTensorDataType dstType, const float* dstScale, int32_t dstZero);
    }
#endif
}

#endif

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2024 Yermalayeu Ihar.
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
#ifndef __SimdSynetGatherElements_h__
#define __SimdSynetGatherElements_h__

#include "Simd/SimdArray.h"

namespace Simd
{
    struct GatherElementsParam
    {
        SimdTensorDataType dataType, indexType;
        SimdBool indexConst;
        size_t srcOuter, srcCount, srcInner, idxCount;

        GatherElementsParam(SimdTensorDataType dt, SimdTensorDataType it, SimdBool iC, size_t so, size_t sc, size_t si, size_t ic)
            : dataType(dt)
            , indexType(it)
            , indexConst(iC)
            , srcOuter(so)
            , srcCount(sc)
            , srcInner(si)
            , idxCount(ic)
        {
        }

        bool Valid() const
        {
            return true;
        }
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetGatherElements : public Simd::Deletable
        {
        public:
            SynetGatherElements(const GatherElementsParam& p);

            virtual size_t InternalBufferSize() const;
            virtual void SetIndex(const uint8_t* idx);
            virtual void Forward(const uint8_t* src, const uint8_t* idx, uint8_t* dst);

            typedef void(*GatherElementsPtr)(const uint8_t* src8, size_t srcOuter, size_t srcCount, size_t srcInner, const uint8_t* idx8, size_t idxCount, uint8_t* dst8);

        protected:
            GatherElementsParam _param;
            Array32i _idx32i;
            GatherElementsPtr _gatherElements;
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetGatherElementsInit(SimdTensorDataType dataType, SimdTensorDataType indexType, SimdBool indexConst, size_t srcOuter, size_t srcCount, size_t srcInner, size_t idxCount);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
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

#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))   
    namespace AmxBf16
    {
    }
#endif
}

#endif

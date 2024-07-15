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
#ifndef __SimdSynetAdd16b_h__
#define __SimdSynetAdd16b_h__

#include "Simd/SimdMemory.h"

#include <vector>

namespace Simd
{
    typedef std::vector<size_t> Shape;

    struct Add16bParam
    {
        Shape aShape, bShape;
        SimdTensorDataType aType, bType, dType;
        SimdTensorFormatType format;

        Add16bParam(const size_t* as, size_t ac, SimdTensorDataType at, const size_t* bs, size_t bc, SimdTensorDataType bt, SimdTensorDataType dt, SimdTensorFormatType f)
            : aShape(as, as + ac) 
            , aType(at) 
            , bShape(bs, bs + bc)
            , bType(bt)
            , dType(dt)
            , format(f)
        {
        }

        bool Valid()
        {
            return
                (format == SimdTensorFormatUnknown || format == SimdTensorFormatNhwc || format == SimdTensorFormatNchw) &&
                (aType == SimdTensorData32f || aType == SimdTensorData16b) &&
                (bType == SimdTensorData32f || bType == SimdTensorData16b) &&
                (dType == SimdTensorData32f || dType == SimdTensorData16b);
        }
    };

    //-------------------------------------------------------------------------------------------------

    class SynetAdd16b : public Deletable
    {
    public:
        SynetAdd16b(const Add16bParam& p);

        virtual void Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst) = 0;

    protected:
        Add16bParam _param;
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetAdd16bUniform : public SynetAdd16b
        {
        public:
            SynetAdd16bUniform(const Add16bParam& p);

            static bool Preferable(const Add16bParam& p);

            virtual void Forward(const uint8_t* a, const uint8_t* b, uint8_t* dst);

            typedef void(*UniformPtr)(const uint8_t* a, const uint8_t* b, size_t size, uint8_t* dst);

        protected:
            size_t _size;
            UniformPtr _uniform;
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetAdd16bInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetAdd16bUniform : public Base::SynetAdd16bUniform
        {
        public:
            SynetAdd16bUniform(const Add16bParam& p);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetAdd16bInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetAdd16bUniform : public Sse41::SynetAdd16bUniform
        {
        public:
            SynetAdd16bUniform(const Add16bParam& p);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetAdd16bInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format);
    }
#endif

//#ifdef SIMD_AVX512BW_ENABLE    
//    namespace Avx512bw
//    {
//        class SynetAdd16b : public Avx2::SynetAdd16b
//        {
//        public:
//            SynetAdd16b(const Add16bParam& p);
//        };
//
//        //-------------------------------------------------------------------------------------------------
//
//        void* SynetAdd16bInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format);
//
//    }
//#endif
//
//#if (defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)))   
//    namespace AmxBf16
//    {
//        class SynetAdd16b : public Avx512bw::SynetAdd16b
//        {
//        public:
//            SynetAdd16b(const Add16bParam& p);
//        };
//
//        //-------------------------------------------------------------------------------------------------
//
//        void* SynetAdd16bInit(const size_t* aShape, size_t aCount, SimdTensorDataType aType, const size_t* bShape, size_t bCount, SimdTensorDataType bType, SimdTensorDataType dstType, SimdTensorFormatType format);
//    }
//#endif
}

#endif

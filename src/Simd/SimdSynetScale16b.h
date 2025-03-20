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
#ifndef __SimdSynetSynet16b_h__
#define __SimdSynetSynet16b_h__

#include "Simd/SimdMemory.h"

namespace Simd
{
    struct Scale16bParam
    {
        size_t channels, spatial;
        SimdTensorDataType sType, dType;
        SimdTensorFormatType format;
        SimdBool norm, bias;

        Scale16bParam(size_t c, size_t s, SimdTensorDataType st, SimdTensorDataType dt, SimdTensorFormatType f, SimdBool n, SimdBool b)
            : channels(c)
            , spatial(s)
            , sType(st) 
            , dType(dt)
            , format(f)
            , norm(n)
            , bias(b)
        {
        }

        bool Valid()
        {
            return
                (channels > 0 && spatial > 0) && (norm || bias) &&
                (format == SimdTensorFormatNhwc || format == SimdTensorFormatNchw) &&
                (sType == SimdTensorData32f || sType == SimdTensorData16b) &&
                (dType == SimdTensorData32f || dType == SimdTensorData16b);
        }
    };

    //-------------------------------------------------------------------------------------------------

    class SynetScale16b : public Deletable
    {
    public:
        SynetScale16b(const Scale16bParam& p);

        virtual void Forward(const uint8_t* src, const float* norm, const float* bias, uint8_t* dst) = 0;

    protected:
        Scale16bParam _param;
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetScale16b : public Simd::SynetScale16b
        {
        public:
            SynetScale16b(const Scale16bParam& p);

            static bool Preferable(const Scale16bParam& p);

            virtual void Forward(const uint8_t* src, const float* norm, const float * bias, uint8_t* dst);

            typedef void (*WorkerPtr)(const uint8_t* src8, size_t channels, size_t spatial, SimdTensorFormatType format, const float* norm, const float* bias, uint8_t* dst8);

        protected:
            WorkerPtr _worker;
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetScale16bInit(size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdBool norm, SimdBool bias);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetScale16b : public Base::SynetScale16b
        {
        public:
            SynetScale16b(const Scale16bParam& p);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetScale16bInit(size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdBool norm, SimdBool bias);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetScale16b : public Sse41::SynetScale16b
        {
        public:
            SynetScale16b(const Scale16bParam& p);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetScale16bInit(size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdBool norm, SimdBool bias);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetScale16b : public Avx2::SynetScale16b
        {
        public:
            SynetScale16b(const Scale16bParam& p);
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetScale16bInit(size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdBool norm, SimdBool bias);

    }
#endif
}

#endif

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
#ifndef __SimdSynetScale8i_h__
#define __SimdSynetScale8i_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdSynetConvolution8i.h"

namespace Simd
{
    namespace Base
    {
        struct Scale8iParam
        {
            size_t batch, channels, spatial;
            SimdTensorDataType srcType, dstType;
            SimdTensorFormatType format;
            SimdSynetCompatibilityType compatibility;

            Scale8iParam(size_t ba, size_t ch, size_t sp, SimdTensorDataType st, SimdTensorDataType dt, SimdTensorFormatType f, SimdSynetCompatibilityType co)
                : batch(ba), channels(ch), spatial(sp), srcType(st), dstType(dt), format(f), compatibility(co) 
            { 
            }

            bool Valid() const
            {
                return true;
            }
        };

        class SynetScale8i : public Deletable
        {
        public:
            SynetScale8i(const Scale8iParam & param);

            size_t InternalBufferSize() const;

            void SetParams(const float* scale, const float* bias, const float* const* stats);

            void Forward(const uint8_t* src, uint8_t* dst);

        protected:
            virtual void Scale(const uint8_t* src, uint8_t* dst);
            virtual void Scale(const uint8_t* src, float* dst);
            virtual void Scale(const float* src, uint8_t* dst);
            virtual void Scale(const float* src, float* dst);

            CvtParam _srcCvt, _dstCvt;
            Array32f _scale, _shift;
            Scale8iParam _param;
        };

        void * SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        class SynetScale8i : public Base::SynetScale8i
        {
        public:
            SynetScale8i(const Base::Scale8iParam& param);

        protected:
            virtual void Scale(const uint8_t* src, uint8_t* dst);
            virtual void Scale(const uint8_t* src, float* dst);
            virtual void Scale(const float* src, uint8_t* dst);
            virtual void Scale(const float* src, float* dst);
        };

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        class SynetScale8i : public Sse41::SynetScale8i
        {
        public:
            SynetScale8i(const Base::Scale8iParam& param);

        protected:
            virtual void Scale(const uint8_t* src, uint8_t* dst);
            virtual void Scale(const uint8_t* src, float* dst);
            virtual void Scale(const float* src, uint8_t* dst);
            virtual void Scale(const float* src, float* dst);
        };

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        class SynetScale8i : public Avx2::SynetScale8i
        {
        public:
            SynetScale8i(const Base::Scale8iParam& param);

        protected:
            virtual void Scale(const uint8_t* src, uint8_t* dst);
            virtual void Scale(const uint8_t* src, float* dst);
            virtual void Scale(const float* src, uint8_t* dst);
            virtual void Scale(const float* src, float* dst);
        };

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);
    }
#endif

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        class SynetScale8i : public Base::SynetScale8i
        {
        public:
            SynetScale8i(const Base::Scale8iParam& param);

        protected:
            virtual void Scale(const uint8_t* src, uint8_t* dst);
            virtual void Scale(const uint8_t* src, float* dst);
            virtual void Scale(const float* src, uint8_t* dst);
            virtual void Scale(const float* src, float* dst);
        };

        void* SynetScale8iInit(size_t batch, size_t channels, size_t spatial, SimdTensorDataType srcType, SimdTensorDataType dstType, SimdTensorFormatType format, SimdSynetCompatibilityType compatibility);
    }
#endif
}

#endif//__SimdSynetConvolution8i_h__

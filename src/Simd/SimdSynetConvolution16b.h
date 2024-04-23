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
#ifndef __SimdSynetConvolution16b_h__
#define __SimdSynetConvolution16b_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdPerformance.h"
#include "Simd/SimdRuntime.h"
#include "Simd/SimdSynetConvParam.h"
#include "Simd/SimdGemm.h"

namespace Simd
{
    class SynetConvolution16b : public Deletable
    {
    public:
        SynetConvolution16b(const ConvParam& p);

        const ConvParam& Param() const
        {
            return _param;
        }

        virtual String Ext() const = 0;
        virtual String Desc() const = 0;

        virtual size_t ExternalBufferSize() const
        {
            return 1;
        }

        virtual size_t InternalBufferSize() const
        {
            return _buffer.RawSize() + _weight.RawSize() +
                _bias.RawSize() + _params.RawSize();
        }

        virtual void SetParams(const float* weight, const float* bias, const float* params);

        virtual void Forward(const uint8_t * src, uint8_t* buf, uint8_t* dst) = 0;

        uint8_t* Buffer(uint8_t* buffer)
        {
            if (buffer)
                return buffer;
            else
            {
                _buffer.Resize(ExternalBufferSize());
                return _buffer.data;
            }
        }

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* Perf(const char* func);
#endif

        const char* Info() const
        {
            _info = Desc();
            return _info.c_str();
        }

    protected:

        ConvParam _param;
        Array8u _buffer;
#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
        Base::PerformanceMeasurer* _perf;
#endif
        mutable String _info;
        Array16u _weight;
        Array32f _bias, _params;
        bool _src16b, _dst16b, _is1x1;
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        class SynetConvolution16bGemm : public SynetConvolution16b
        {
        public:
            SynetConvolution16bGemm(const ConvParam& p);
            virtual String Ext() const { return "Base"; }
            virtual String Desc() const { return Ext() + "::Gemm"; }
            virtual size_t ExternalBufferSize() const;
            virtual void SetParams(const float* weight, const float* bias, const float* params);
            virtual void Forward(const uint8_t* src, uint8_t* buf, uint8_t* dst);

        protected:
            void ImgToCol(const uint16_t* src, uint16_t* dst);
            void ImgToRow(const uint16_t* src, uint16_t* dst);

            void GemmNN(size_t M, size_t N, size_t K, const uint16_t* A, size_t lda, const uint16_t* B, size_t ldb, float* C, size_t ldc);

            size_t _M, _N, _K, _ldW, _ldS, _ldD, _grW, _grS, _grD, _batch, _sizeS, _sizeB, _sizeD;
        };

        //-------------------------------------------------------------------------------------------------

        void* SynetConvolution16bInit(size_t batch, const SimdConvolutionParameters* conv, SimdSynetCompatibilityType compatibility);
    }
}

#endif

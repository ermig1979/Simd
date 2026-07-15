/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#include "Test/TestUtils.h"
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestTensor.h"
#include "Test/TestRandom.h"
#include "Test/TestString.h"
#include "Test/TestOptions.h"
#include "Test/TestSynetPoolingParam.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    typedef ParamPooling ParamP;

    namespace
    {
        struct FuncSqpa
        {
            typedef void (*FuncPtr)(const uint8_t* src, const float* srcScale, int srcZero, size_t batch, size_t srcC, size_t srcH, size_t srcW,
                size_t kernelY, size_t kernelX, size_t strideY, size_t strideX, size_t padY, size_t padX, SimdBool excludePad,
                uint8_t* dst, const float* dstScale, int dstZero, size_t dstH, size_t dstW, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncSqpa(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const ParamP& p)
            {
                std::stringstream ss;
                ss << desc;
                ss << "[" << p.batch << "x" << p.srcC << "x" << p.srcH << "x" << p.srcW;
                ss << "-" << p.kernelY << "x" << p.kernelX;
                ss << "-" << p.strideX << "-" << Simd::Max(p.padX, p.padY) << "-" << p.excludePad;
                ss << "-" << (p.format == SimdTensorFormatNhwc ? "1" : "0") << "]";
                desc = ss.str();
            }

            void Call(const Tensor8u& src, float srcScale, int srcZero, const ParamP& p, 
                Tensor8u& dst, float dstScale, int dstZero) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), &srcScale, srcZero, p.batch, p.srcC, p.srcH, p.srcW, p.kernelY, p.kernelX, p.strideY, p.strideX,
                    p.padY, p.padX, p.excludePad, dst.Data(), &dstScale, dstZero, p.dstH, p.dstW, p.format);
            }
        };
    }

#define FUNC_SQPA(function) FuncSqpa(function, #function)

    bool SynetQuantizedPoolingAverageAutoTest(const ParamP& p, FuncSqpa f1, FuncSqpa f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " .");

        Tensor8u src(ToShape(p.batch, p.srcC, p.srcH, p.srcW, p.format));
        Tensor8u dst0(ToShape(p.batch, p.srcC, p.dstH, p.dstW, p.format));
        Tensor8u dst1(ToShape(p.batch, p.srcC, p.dstH, p.dstW, p.format));

        srand(0);
        FillRandom(src);

        float srcScale = 50.0f, dstScale = 70.0f;
        int32_t srcZero = 47, dstZero = 30;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcScale, srcZero, p, dst0, dstScale, dstZero));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcScale, srcZero, p, dst1, dstScale, dstZero));

#if defined(SIMD_X64_ENABLE) || defined(SIMD_X86_ENABLE)
        int differenceMax = 0;
#else
        int differenceMax = 1;
#endif
        result = result && Compare(dst0, dst1, differenceMax, true, 64);

        return result;
    }

    bool SynetQuantizedPoolingAverageAutoTest(::SimdTensorFormatType f, ::SimdBool c, ::SimdBool e, const FuncSqpa& f1, const FuncSqpa& f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);

#ifdef NDEBUG
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(4, 127, 55, 95, f), f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(1, 128, 54, 96, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(1, 128, 27, 48, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(1, 128, 13, 24, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(1, 65, 13, 24, _3, _2, _0, _1, f, c, e), f1, f2);
#else
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(2, 31, 25, 45, f), f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(1, 7, 54, 40, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(1, 17, 27, 24, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(1, 16, 33, 33, _3, _1, _1, _1, f, c, e), f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(ParamP(1, 16, 22, 22, _3, _2, _0, _1, f, c, e), f1, f2);
#endif

        return result;
    }

    bool SynetQuantizedPoolingAverageAutoTest(const FuncSqpa& f1, const FuncSqpa& f2)
    {
        bool result = true;

        result = result && SynetQuantizedPoolingAverageAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdFalse, f1, f2);
        result = result && SynetQuantizedPoolingAverageAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdFalse, f1, f2);

        return result;
    }

    bool SynetQuantizedPoolingAverageAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetQuantizedPoolingAverageAutoTest(FUNC_SQPA(Simd::Base::SynetQuantizedPoolingAverage), FUNC_SQPA(SimdSynetQuantizedPoolingAverage));

//#ifdef SIMD_SSE41_ENABLE
//        if (Simd::Sse41::Enable && TestSse41(options))
//            result = result && SynetQuantizedPoolingAverageAutoTest(FUNC_SQPA(Simd::Sse41::SynetQuantizedPoolingAverage), FUNC_SQPA(SimdSynetQuantizedPoolingAverage));
//#endif 

        return result;
    }
#endif
}

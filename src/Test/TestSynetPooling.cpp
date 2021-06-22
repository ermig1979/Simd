/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Test/TestPerformance.h"
#include "Test/TestData.h"
#include "Test/TestTensor.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct ParamP
        {
            size_t srcC, srcH, srcW, kernelY, kernelX, strideY, strideX, padY, padX, dstH, dstW;
            SimdTensorFormatType format;
            SimdBool ceil, excludePad;

            ParamP(size_t sC, size_t sH, size_t sW, Size k, Size s, Size b, Size e, ::SimdTensorFormatType f, ::SimdBool c, SimdBool ep)
                : srcC(sC), srcH(sH), srcW(sW), kernelY(k.y), kernelX(k.x), strideY(s.y), strideX(s.x)
                , padY(b.y), padX(b.x), format(f), ceil(c), excludePad(ep)
            {
                if (ceil)
                {
                    dstH = (size_t)(::ceil((float)(srcH + b.y + e.y - kernelY) / strideY)) + 1;
                    dstW = (size_t)(::ceil((float)(srcW + b.x + e.x - kernelX) / strideX)) + 1;
                }
                else
                {
                    dstH = (size_t)(::floor((float)(srcH + b.y + e.y - kernelY) / strideY)) + 1;
                    dstW = (size_t)(::floor((float)(srcW + b.x + e.x - kernelX) / strideX)) + 1;
                }
            }
        };
    }

    //---------------------------------------------------------------------

    namespace
    {
        struct FuncPA
        {
            typedef void(*FuncPtr)(const float* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
                size_t strideY, size_t strideX, size_t padY, size_t padX, float* dst, size_t dstH, size_t dstW, SimdBool excludePad, SimdTensorFormatType format);

            FuncPtr func;
            String desc;

            FuncPA(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const ParamP& p)
            {
                std::stringstream ss;
                ss << desc;
                ss << "[" << p.srcC << "x" << p.srcH << "x" << p.srcW;
                ss << "-" << p.kernelY << "x" << p.kernelX;
                ss << "-" << p.strideX << "-" << Simd::Max(p.padX, p.padY) << "-" << p.excludePad << "-" << p.format;
                ss << "]";
                desc = ss.str();
            }

            void Call(const ParamP& p, const Tensor32f& src, Tensor32f& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), p.srcC, p.srcH, p.srcW, p.kernelY, p.kernelX, p.strideY, p.strideX,
                    p.padY, p.padX, dst.Data(), p.dstH, p.dstW, p.excludePad, p.format);
            }
        };
    }

#define FUNC_PA(function) FuncPA(function, #function)

    bool SynetPoolingForwardAverageAutoTest(const ParamP& p, FuncPA f1, FuncPA f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << "].");

        Tensor32f src(ToShape(p.srcC, p.srcH, p.srcW, p.format));
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f dst1(ToShape(p.srcC, p.dstH, p.dstW, p.format));
        Tensor32f dst2(ToShape(p.srcC, p.dstH, p.dstW, p.format));

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool SynetPoolingForwardAverageAutoTest(::SimdTensorFormatType f, ::SimdBool c, ::SimdBool e, const FuncPA& f1, const FuncPA& f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);

        result = result && SynetPoolingForwardAverageAutoTest(ParamP(10, 238, 132, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetPoolingForwardAverageAutoTest(ParamP(32, 99, 99, _3, _1, _1, _1, f, c, e), f1, f2);
        result = result && SynetPoolingForwardAverageAutoTest(ParamP(32, 46, 46, _3, _2, _0, _1, f, c, e), f1, f2);

        return result;
    }

    bool SynetPoolingForwardAverageAutoTest(const FuncPA& f1, const FuncPA& f2)
    {
        bool result = true;

        result = result && SynetPoolingForwardAverageAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetPoolingForwardAverageAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetPoolingForwardAverageAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdFalse, f1, f2);
        result = result && SynetPoolingForwardAverageAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdFalse, f1, f2);

        return result;
    }

    bool SynetPoolingForwardAverageAutoTest()
    {
        bool result = true;

        result = result && SynetPoolingForwardAverageAutoTest(FUNC_PA(Simd::Base::SynetPoolingForwardAverage), FUNC_PA(SimdSynetPoolingForwardAverage));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetPoolingForwardAverageAutoTest(FUNC_PA(Simd::Sse2::SynetPoolingForwardAverage), FUNC_PA(SimdSynetPoolingForwardAverage));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetPoolingForwardAverageAutoTest(FUNC_PA(Simd::Avx::SynetPoolingForwardAverage), FUNC_PA(SimdSynetPoolingForwardAverage));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetPoolingForwardAverageAutoTest(FUNC_PA(Simd::Avx512f::SynetPoolingForwardAverage), FUNC_PA(SimdSynetPoolingForwardAverage));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetPoolingForwardAverageAutoTest(FUNC_PA(Simd::Neon::SynetPoolingForwardAverage), FUNC_PA(SimdSynetPoolingForwardAverage));
#endif 

        return result;
    }

    //---------------------------------------------------------------------

    template<class T> struct FuncPM
    {
        typedef void(*FuncPtr)(const T * src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, T * dst, size_t dstH, size_t dstW, SimdTensorFormatType format);

        FuncPtr func;
        String desc;

        FuncPM(const FuncPtr & f, const String & d) : func(f), desc(d) {}

        void Update(const ParamP & p)
        {
            std::stringstream ss;
            ss << desc;
            ss << "[" << p.srcC << "x" << p.srcH << "x" << p.srcW;
            ss << "-" << p.kernelY << "x" << p.kernelX;
            ss << "-" << p.strideX << "-" << Simd::Max(p.padX, p.padY) << "-" << p.format;
            ss << "]";
            desc = ss.str();
        }

        void Call(const ParamP & p, const Tensor<T> & src, Tensor<T>& dst) const
        {
            TEST_PERFORMANCE_TEST(desc);
            func(src.Data(), p.srcC, p.srcH, p.srcW, p.kernelY, p.kernelX, p.strideY, p.strideX, p.padY, p.padX, dst.Data(), p.dstH, p.dstW, p.format);
        }
    };

    typedef FuncPM<float> FuncPM32f;

#define FUNC_PM32F(function) FuncPM32f(function, #function)

    bool SynetPoolingForwardMax32fAutoTest(const ParamP & p, FuncPM32f f1, FuncPM32f f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << "].");

        Tensor32f src(ToShape(p.srcC, p.srcH, p.srcW, p.format));
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f dst1(ToShape(p.srcC, p.dstH, p.dstW, p.format));
        Tensor32f dst2(ToShape(p.srcC, p.dstH, p.dstW, p.format));

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool SynetPoolingForwardMax32fAutoTest(::SimdTensorFormatType f, ::SimdBool c, ::SimdBool e, const FuncPM32f & f1, const FuncPM32f & f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);

        result = result && SynetPoolingForwardMax32fAutoTest(ParamP(10, 238, 133, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetPoolingForwardMax32fAutoTest(ParamP(28, 99, 99, _3, _1, _1, _1, f, c, e), f1, f2);
        result = result && SynetPoolingForwardMax32fAutoTest(ParamP(32, 46, 46, _3, _2, _0, _1, f, c, e), f1, f2);
        result = result && SynetPoolingForwardMax32fAutoTest(ParamP(64, 21, 21, _3, _2, _1, _1, f, c, e), f1, f2);

        return result;
    }

    bool SynetPoolingForwardMax32fAutoTest(const FuncPM32f& f1, const FuncPM32f& f2)
    {
        bool result = true;

        result = result && SynetPoolingForwardMax32fAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetPoolingForwardMax32fAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdTrue, f1, f2);

        return result;
    }

    bool SynetPoolingForwardMax32fAutoTest()
    {
        bool result = true;

        result = result && SynetPoolingForwardMax32fAutoTest(FUNC_PM32F(Simd::Base::SynetPoolingForwardMax32f), FUNC_PM32F(SimdSynetPoolingForwardMax32f));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetPoolingForwardMax32fAutoTest(FUNC_PM32F(Simd::Sse2::SynetPoolingForwardMax32f), FUNC_PM32F(SimdSynetPoolingForwardMax32f));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetPoolingForwardMax32fAutoTest(FUNC_PM32F(Simd::Avx::SynetPoolingForwardMax32f), FUNC_PM32F(SimdSynetPoolingForwardMax32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetPoolingForwardMax32fAutoTest(FUNC_PM32F(Simd::Avx2::SynetPoolingForwardMax32f), FUNC_PM32F(SimdSynetPoolingForwardMax32f));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetPoolingForwardMax32fAutoTest(FUNC_PM32F(Simd::Avx512f::SynetPoolingForwardMax32f), FUNC_PM32F(SimdSynetPoolingForwardMax32f));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetPoolingForwardMax32fAutoTest(FUNC_PM32F(Simd::Neon::SynetPoolingForwardMax32f), FUNC_PM32F(SimdSynetPoolingForwardMax32f));
#endif 

        return result;
    }

    //---------------------------------------------------------------------

    typedef FuncPM<uint8_t> FuncPM8u;

#define FUNC_PM8U(function) FuncPM8u(function, #function)

    bool SynetPoolingForwardMax8uAutoTest(const ParamP& p, FuncPM8u f1, FuncPM8u f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << "].");

        Tensor8u src(ToShape(p.srcC, p.srcH, p.srcW, p.format));
        FillRandom(src.Data(), src.Size(), 0, 255);

        Tensor8u dst1(ToShape(p.srcC, p.dstH, p.dstW, p.format));
        Tensor8u dst2(ToShape(p.srcC, p.dstH, p.dstW, p.format));

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, src, dst2));

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool SynetPoolingForwardMax8uAutoTest(::SimdTensorFormatType f, ::SimdBool c, ::SimdBool e, const FuncPM8u& f1, const FuncPM8u& f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);

        result = result && SynetPoolingForwardMax8uAutoTest(ParamP(10, 238, 133, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetPoolingForwardMax8uAutoTest(ParamP(28, 99, 99, _3, _1, _1, _1, f, c, e), f1, f2);
        result = result && SynetPoolingForwardMax8uAutoTest(ParamP(32, 46, 46, _3, _2, _0, _1, f, c, e), f1, f2);
        result = result && SynetPoolingForwardMax8uAutoTest(ParamP(64, 21, 21, _3, _2, _1, _1, f, c, e), f1, f2);

        return result;
    }

    bool SynetPoolingForwardMax8uAutoTest(const FuncPM8u& f1, const FuncPM8u& f2)
    {
        bool result = true;

        result = result && SynetPoolingForwardMax8uAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetPoolingForwardMax8uAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdTrue, f1, f2);

        return result;
    }

    bool SynetPoolingForwardMax8uAutoTest()
    {
        bool result = true;

        result = result && SynetPoolingForwardMax8uAutoTest(FUNC_PM8U(Simd::Base::SynetPoolingForwardMax8u), FUNC_PM8U(SimdSynetPoolingForwardMax8u));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetPoolingForwardMax8uAutoTest(FUNC_PM8U(Simd::Sse41::SynetPoolingForwardMax8u), FUNC_PM8U(SimdSynetPoolingForwardMax8u));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetPoolingForwardMax8uAutoTest(FUNC_PM8U(Simd::Avx2::SynetPoolingForwardMax8u), FUNC_PM8U(SimdSynetPoolingForwardMax8u));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetPoolingForwardMax8uAutoTest(FUNC_PM8U(Simd::Avx512bw::SynetPoolingForwardMax8u), FUNC_PM8U(SimdSynetPoolingForwardMax8u));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
           result = result && SynetPoolingForwardMax8uAutoTest(FUNC_PM8U(Simd::Neon::SynetPoolingForwardMax8u), FUNC_PM8U(SimdSynetPoolingForwardMax8u));
#endif 

        return result;
    }
#endif
}

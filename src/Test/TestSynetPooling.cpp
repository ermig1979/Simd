/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
            size_t srcC, srcH, srcW, kernelC, kernelY, kernelX, strideC, strideY, strideX, padC, padY, padX, dstC, dstH, dstW;
            SimdTensorFormatType format;
            SimdBool ceil, excludePad;

            ParamP(size_t sC, size_t sH, size_t sW, Size k, Size s, Size b, Size e, ::SimdTensorFormatType f, ::SimdBool c, SimdBool ep)
                : srcC(sC), srcH(sH), srcW(sW), kernelC(1), kernelY(k.y), kernelX(k.x), strideC(1), strideY(s.y), strideX(s.x)
                , padC(0), padY(b.y), padX(b.x), format(f), ceil(c), excludePad(ep)
            {
                SetDst(0, e.y, e.x);
            }

            ParamP(size_t sC, size_t sH, size_t sW, const Shape& k, const Shape& s, const Shape& b, const Shape& e, ::SimdTensorFormatType f, ::SimdBool c, SimdBool ep)
                : srcC(sC), srcH(sH), srcW(sW), kernelC(k[0]), kernelY(k[1]), kernelX(k[2]), strideC(s[0]), strideY(s[1]), strideX(s[2])
                , padC(b[0]), padY(b[1]), padX(b[2]), format(f), ceil(c), excludePad(ep)
            {
                SetDst(e[0], e[1], e[2]);
            }

        protected:
            SIMD_INLINE void SetDst(size_t padD, size_t padH, size_t padW)
            {
                if (ceil)
                {
                    dstC = (size_t)(::ceil((float)(srcC + padC + padD - kernelC) / strideC)) + 1;
                    dstH = (size_t)(::ceil((float)(srcH + padY + padH - kernelY) / strideY)) + 1;
                    dstW = (size_t)(::ceil((float)(srcW + padX + padW - kernelX) / strideX)) + 1;
                }
                else
                {
                    dstC = (size_t)(::floor((float)(srcC + padC + padD - kernelC) / strideC)) + 1;
                    dstH = (size_t)(::floor((float)(srcH + padY + padH - kernelY) / strideY)) + 1;
                    dstW = (size_t)(::floor((float)(srcW + padX + padW - kernelX) / strideX)) + 1;
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
                ss << "-" << p.strideX << "-" << Simd::Max(p.padX, p.padY) << "-" << p.excludePad;
                ss << "-" << (p.format == SimdTensorFormatNhwc ? "1" : "0");
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

    bool SynetPoolingAverageAutoTest(const ParamP& p, FuncPA f1, FuncPA f2)
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

    bool SynetPoolingAverageAutoTest(::SimdTensorFormatType f, ::SimdBool c, ::SimdBool e, const FuncPA& f1, const FuncPA& f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);

        result = result && SynetPoolingAverageAutoTest(ParamP(10, 238, 132, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetPoolingAverageAutoTest(ParamP(32, 99, 99, _3, _1, _1, _1, f, c, e), f1, f2);
        result = result && SynetPoolingAverageAutoTest(ParamP(32, 46, 46, _3, _2, _0, _1, f, c, e), f1, f2);

        return result;
    }

    bool SynetPoolingAverageAutoTest(const FuncPA& f1, const FuncPA& f2)
    {
        bool result = true;

        result = result && SynetPoolingAverageAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetPoolingAverageAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetPoolingAverageAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdFalse, f1, f2);
        result = result && SynetPoolingAverageAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdFalse, f1, f2);

        return result;
    }

    bool SynetPoolingAverageAutoTest()
    {
        bool result = true;

        result = result && SynetPoolingAverageAutoTest(FUNC_PA(Simd::Base::SynetPoolingAverage), FUNC_PA(SimdSynetPoolingAverage));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetPoolingAverageAutoTest(FUNC_PA(Simd::Sse41::SynetPoolingAverage), FUNC_PA(SimdSynetPoolingAverage));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetPoolingAverageAutoTest(FUNC_PA(Simd::Avx::SynetPoolingAverage), FUNC_PA(SimdSynetPoolingAverage));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetPoolingAverageAutoTest(FUNC_PA(Simd::Avx512bw::SynetPoolingAverage), FUNC_PA(SimdSynetPoolingAverage));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetPoolingAverageAutoTest(FUNC_PA(Simd::Neon::SynetPoolingAverage), FUNC_PA(SimdSynetPoolingAverage));
#endif 

        return result;
    }

    //---------------------------------------------------------------------

    struct FuncPM32f
    {
        typedef void(*FuncPtr)(const float* src, size_t srcC, size_t srcH, size_t srcW,
            size_t kernelC, size_t kernelY, size_t kernelX, size_t strideC, size_t strideY, size_t strideX,
            size_t padC, size_t padY, size_t padX, float* dst, size_t dstC, size_t dstH, size_t dstW, SimdTensorFormatType format);

        FuncPtr func;
        String desc;

        FuncPM32f(const FuncPtr & f, const String & d) : func(f), desc(d) {}

        void Update(const ParamP & p)
        {
            std::stringstream ss;
            ss << desc;
            ss << "[" << p.srcC << "x" << p.srcH << "x" << p.srcW;
            ss << "-" << p.kernelC << "x" << p.kernelY << "x" << p.kernelX;
            ss << "-" << p.strideC << "x" << p.strideY << "x" << p.strideX;
            ss << "-" << Simd::Max(p.padC, Simd::Max(p.padX, p.padY));
            ss << "-" << (p.format == SimdTensorFormatNhwc ? "1" : "0");
            ss << "]";
            desc = ss.str();
        }

        void Call(const ParamP & p, const Tensor32f& src, Tensor32f& dst) const
        {
            TEST_PERFORMANCE_TEST(desc);
            func(src.Data(), p.srcC, p.srcH, p.srcW, p.kernelC, p.kernelY, p.kernelX, p.strideC, p.strideY, p.strideX, 
                p.padC, p.padY, p.padX, dst.Data(), p.dstC, p.dstH, p.dstW, p.format);
        }
    };

#define FUNC_PM32F(function) FuncPM32f(function, #function)

    bool SynetPoolingMax32fAutoTest(const ParamP & p, FuncPM32f f1, FuncPM32f f2)
    {
        bool result = true;

        f1.Update(p);
        f2.Update(p);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << "].");

        Tensor32f src(ToShape(p.srcC, p.srcH, p.srcW, p.format));
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f dst1(ToShape(p.dstC, p.dstH, p.dstW, p.format));
        Tensor32f dst2(ToShape(p.dstC, p.dstH, p.dstW, p.format));

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(p, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(p, src, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool SynetPoolingMax32fAutoTest(::SimdTensorFormatType f, const FuncPM32f & f1, const FuncPM32f & f2)
    {
        bool result = true;

        SimdBool c = SimdTrue, e = SimdTrue;
        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);

#if 0
        result = result && SynetPoolingMax32fAutoTest(ParamP(10, 238, 133, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetPoolingMax32fAutoTest(ParamP(28, 99, 99, _3, _1, _1, _1, f, c, e), f1, f2);
        result = result && SynetPoolingMax32fAutoTest(ParamP(32, 46, 46, _3, _2, _0, _1, f, c, e), f1, f2);
        result = result && SynetPoolingMax32fAutoTest(ParamP(64, 21, 21, _3, _2, _1, _1, f, c, e), f1, f2);
#endif
#if 0        
        result = result && SynetPoolingMax32fAutoTest(ParamP(101, 59, 99, Shp(2, 3, 1), Shp(3, 2, 1), Shp(0, 1, 1), Shp(0, 1, 0), f, c, e), f1, f2);
#endif
#if 1
        result = result && SynetPoolingMax32fAutoTest(ParamP(128, 19, 90, Shp(1, 3, 3), Shp(1, 2, 1), Shp(0, 0, 0), Shp(0, 0, 0), f, c, e), f1, f2);
        result = result && SynetPoolingMax32fAutoTest(ParamP(256, 9, 88, Shp(1, 3, 3), Shp(1, 2, 1), Shp(0, 0, 0), Shp(0, 0, 0), f, c, e), f1, f2);
        result = result && SynetPoolingMax32fAutoTest(ParamP(64, 21, 92, Shp(1, 3, 3), Shp(1, 1, 1), Shp(0, 0, 0), Shp(0, 0, 0), f, c, e), f1, f2);
        result = result && SynetPoolingMax32fAutoTest(ParamP(128, 19, 90, Shp(2, 3, 3), Shp(2, 2, 1), Shp(0, 0, 0), Shp(0, 0, 0), f, c, e), f1, f2);
        result = result && SynetPoolingMax32fAutoTest(ParamP(256, 9, 88, Shp(4, 3, 3), Shp(4, 2, 1), Shp(0, 0, 0), Shp(0, 0, 0), f, c, e), f1, f2);
#endif
            
        return result;
    }

    bool SynetPoolingMax32fAutoTest(const FuncPM32f& f1, const FuncPM32f& f2)
    {
        bool result = true;

        //result = result && SynetPoolingMax32fAutoTest(::SimdTensorFormatNchw, f1, f2);
        result = result && SynetPoolingMax32fAutoTest(::SimdTensorFormatNhwc, f1, f2);

        return result;
    }

    bool SynetPoolingMax32fAutoTest()
    {
        bool result = true;

        result = result && SynetPoolingMax32fAutoTest(FUNC_PM32F(Simd::Base::SynetPoolingMax32f), FUNC_PM32F(SimdSynetPoolingMax32f));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetPoolingMax32fAutoTest(FUNC_PM32F(Simd::Sse41::SynetPoolingMax32f), FUNC_PM32F(SimdSynetPoolingMax32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetPoolingMax32fAutoTest(FUNC_PM32F(Simd::Avx2::SynetPoolingMax32f), FUNC_PM32F(SimdSynetPoolingMax32f));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetPoolingMax32fAutoTest(FUNC_PM32F(Simd::Avx512bw::SynetPoolingMax32f), FUNC_PM32F(SimdSynetPoolingMax32f));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetPoolingMax32fAutoTest(FUNC_PM32F(Simd::Neon::SynetPoolingMax32f), FUNC_PM32F(SimdSynetPoolingMax32f));
#endif 

        return result;
    }

    //---------------------------------------------------------------------

    struct FuncPM8u
    {
        typedef void(*FuncPtr)(const uint8_t* src, size_t srcC, size_t srcH, size_t srcW, size_t kernelY, size_t kernelX,
            size_t strideY, size_t strideX, size_t padY, size_t padX, uint8_t* dst, size_t dstH, size_t dstW, SimdTensorFormatType format);

        FuncPtr func;
        String desc;

        FuncPM8u(const FuncPtr& f, const String& d) : func(f), desc(d) {}

        void Update(const ParamP& p)
        {
            std::stringstream ss;
            ss << desc;
            ss << "[" << p.srcC << "x" << p.srcH << "x" << p.srcW;
            ss << "-" << p.kernelY << "x" << p.kernelX;
            ss << "-" << p.strideY << "x" << p.strideX;
            ss << "-" << Simd::Max(p.padX, p.padY);
            ss << "-" << (p.format == SimdTensorFormatNhwc ? "1" : "0");
            ss << "]";
            desc = ss.str();
        }

        void Call(const ParamP& p, const Tensor8u& src, Tensor8u& dst) const
        {
            TEST_PERFORMANCE_TEST(desc);
            func(src.Data(), p.srcC, p.srcH, p.srcW, p.kernelY, p.kernelX, p.strideY, p.strideX, p.padY, p.padX, dst.Data(), p.dstH, p.dstW, p.format);
        }
    };

#define FUNC_PM8U(function) FuncPM8u(function, #function)

    bool SynetPoolingMax8uAutoTest(const ParamP& p, FuncPM8u f1, FuncPM8u f2)
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

    bool SynetPoolingMax8uAutoTest(::SimdTensorFormatType f, ::SimdBool c, ::SimdBool e, const FuncPM8u& f1, const FuncPM8u& f2)
    {
        bool result = true;

        Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3);

        result = result && SynetPoolingMax8uAutoTest(ParamP(10, 238, 133, _2, _2, _0, _0, f, c, e), f1, f2);
        result = result && SynetPoolingMax8uAutoTest(ParamP(28, 99, 99, _3, _1, _1, _1, f, c, e), f1, f2);
        result = result && SynetPoolingMax8uAutoTest(ParamP(32, 46, 46, _3, _2, _0, _1, f, c, e), f1, f2);
        result = result && SynetPoolingMax8uAutoTest(ParamP(64, 21, 21, _3, _2, _1, _1, f, c, e), f1, f2);

        return result;
    }

    bool SynetPoolingMax8uAutoTest(const FuncPM8u& f1, const FuncPM8u& f2)
    {
        bool result = true;

        result = result && SynetPoolingMax8uAutoTest(::SimdTensorFormatNchw, ::SimdTrue, ::SimdTrue, f1, f2);
        result = result && SynetPoolingMax8uAutoTest(::SimdTensorFormatNhwc, ::SimdTrue, ::SimdTrue, f1, f2);

        return result;
    }

    bool SynetPoolingMax8uAutoTest()
    {
        bool result = true;

        result = result && SynetPoolingMax8uAutoTest(FUNC_PM8U(Simd::Base::SynetPoolingMax8u), FUNC_PM8U(SimdSynetPoolingMax8u));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetPoolingMax8uAutoTest(FUNC_PM8U(Simd::Sse41::SynetPoolingMax8u), FUNC_PM8U(SimdSynetPoolingMax8u));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetPoolingMax8uAutoTest(FUNC_PM8U(Simd::Avx2::SynetPoolingMax8u), FUNC_PM8U(SimdSynetPoolingMax8u));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetPoolingMax8uAutoTest(FUNC_PM8U(Simd::Avx512bw::SynetPoolingMax8u), FUNC_PM8U(SimdSynetPoolingMax8u));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
           result = result && SynetPoolingMax8uAutoTest(FUNC_PM8U(Simd::Neon::SynetPoolingMax8u), FUNC_PM8U(SimdSynetPoolingMax8u));
#endif 

        return result;
    }
#endif
}

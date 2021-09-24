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
#include "Test/TestString.h"
#include "Simd/SimdSynet.h"

#include "Simd/SimdSynetInnerProduct32f.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncIP32F
        {
            typedef void* (*FuncPtr)(size_t batch, size_t input, size_t output, SimdBool transpose, SimdConvolutionActivationType activation);

            FuncPtr func;
            String desc;

            FuncIP32F(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t b, size_t i, size_t o, SimdBool t, SimdConvolutionActivationType a)
            {
                desc = desc + "[" + ToString(b) + "-" + ToString(i) + "-" + ToString(o) + "-" + ToString((int)t) + "]";
            }

            void Call(void* context, const Tensor32f& src, Tensor32f& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ::SimdSynetInnerProduct32fForward(context, src.Data(), dst.Data());
            }
        };
    }

#define FUNC_IP32F(function) \
    FuncIP32F(function, std::string(#function))

    bool SynetInnerProduct32fForwardAutoTest(float eps, size_t b, size_t i, size_t o, SimdBool t, SimdConvolutionActivationType a, FuncIP32F f1, FuncIP32F f2)
    {
        bool result = true;

        f1.Update(b, i, o, t, a);
        f2.Update(b, i, o, t, a);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        Tensor32f src({ b, i });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);

        Tensor32f weight({ t ? o : i, t ? i : o });
        FillRandom(weight.Data(), weight.Size(), -1.0, 1.0f);

        Tensor32f bias({ o });
        FillRandom(bias.Data(), bias.Size(), -1.0, 1.0f);

        Tensor32f params({ std::max<size_t>(o, 2) });
        FillRandom(params.Data(), params.Size(), 0.0f, 2.0f);

        if (a == ::SimdConvolutionActivationHswish)
        {
            params.Data()[0] = 3.0f;
            params.Data()[1] = 1.0f / 6.0f;
        }
        else if (a == ::SimdConvolutionActivationMish)
            params.Data()[0] = 20.0f;
        else
        {
            params.Data()[0] = 0.1f;
            params.Data()[1] = 1.1f;
        }

        Tensor32f buf;

        Tensor32f dst1({ b, o });
        Tensor32f dst2({ b, o });

        ::SimdFill32f(dst1.Data(), dst1.Size(), params.Data() + 0);
        ::SimdFill32f(dst2.Data(), dst2.Size(), params.Data() + 1);

        void* context1 = f1.func(b, i, o, t, a);
        void* context2 = f2.func(b, i, o, t, a);

        ::SimdSynetInnerProduct32fSetParams(context1, weight.Data(), NULL, bias.Data(), params.Data());
        ::SimdSynetInnerProduct32fSetParams(context2, weight.Data(), NULL, bias.Data(), params.Data());

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, src, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);

        result = result && Compare(dst1, dst2, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetInnerProduct32fForwardAutoTest(float eps, const FuncIP32F& f1, const FuncIP32F& f2)
    {
        bool result = true;

        SimdBool t = SimdTrue, f = SimdFalse;
        SimdConvolutionActivationType a = SimdConvolutionActivationIdentity;

#ifdef NDEBUG
#if 0
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 1, 192, 96, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 1, 192, 192, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 1, 288, 96, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 1, 288, 192, f, a, f1, f2);
#endif
#if 0
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 192, 96, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 192, 192, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 288, 96, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 288, 192, f, a, f1, f2);
#endif
#if 1        
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 1, 192, 96, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 1, 192, 192, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 1, 288, 96, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 1, 288, 192, t, a, f1, f2);
#endif
#if 0
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 192, 96, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 192, 192, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 288, 96, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 288, 192, t, a, f1, f2);
#endif
#if 1
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 1024, 4096, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 256, 1024, f, a, f1, f2);       
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 4096, 254, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 1024, 4096, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 256, 1024, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 10, 4096, 254, t, a, f1, f2);
        //result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 1024, 4096, f, a, f1, f2);
        //result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 256, 1024, f, a, f1, f2);
        //result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 4096, 254, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 1024, 4096, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 4096, 1024, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 1024, 4096, f, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 4096, 1024, f, a, f1, f2);
#endif
#else
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 1024, 4096, t, a, f1, f2);
        result = result && SynetInnerProduct32fForwardAutoTest(eps, 100, 4096, 1024, t, a, f1, f2);
#endif

        return result;
    }

    bool SynetInnerProduct32fForwardAutoTest()
    {
        const float EPS = 0.001f;
        bool result = true;

        result = result && SynetInnerProduct32fForwardAutoTest(EPS, FUNC_IP32F(Simd::Base::SynetInnerProduct32fInit), FUNC_IP32F(SimdSynetInnerProduct32fInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetInnerProduct32fForwardAutoTest(EPS, FUNC_IP32F(Simd::Sse41::SynetInnerProduct32fInit), FUNC_IP32F(SimdSynetInnerProduct32fInit));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetInnerProduct32fForwardAutoTest(EPS, FUNC_IP32F(Simd::Avx::SynetInnerProduct32fInit), FUNC_IP32F(SimdSynetInnerProduct32fInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetInnerProduct32fForwardAutoTest(EPS, FUNC_IP32F(Simd::Avx2::SynetInnerProduct32fInit), FUNC_IP32F(SimdSynetInnerProduct32fInit));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetInnerProduct32fForwardAutoTest(EPS, FUNC_IP32F(Simd::Avx512f::SynetInnerProduct32fInit), FUNC_IP32F(SimdSynetInnerProduct32fInit));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetInnerProduct32fForwardAutoTest(EPS, FUNC_IP32F(Simd::Neon::SynetInnerProduct32fInit), FUNC_IP32F(SimdSynetInnerProduct32fInit));
#endif

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncIPLF
        {
            typedef void(*FuncPtr)(const float * src, const float * weight, const float * bias, size_t count, size_t size, float * dst);

            FuncPtr func;
            String desc;

            FuncIPLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}
            FuncIPLF(const FuncIPLF & f, bool bias) : func(f.func), desc(f.desc + (bias ? "[1]" : "[0]")) {}

            void Call(const View & src, const View & weight, const View & bias, size_t count, size_t size, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)weight.data, (float*)bias.data, count, size, (float*)dst.data);
            }
        };
    }

#define FUNC_IPLF(function) FuncIPLF(function, #function)
#define ARGS_IPLF(bias, f1, f2) bias, FuncIPLF(f1, bias), FuncIPLF(f2, bias)

    bool SynetInnerProductLayerForwardAutoTest(size_t count, size_t size, bool hasBias, const FuncIPLF & f1, const FuncIPLF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View weight(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias;
        View dst1(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -1.0, 1.0);
        FillRandom32f(weight, -1.0, 1.0);
        if (hasBias)
        {
            bias.Recreate(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
            FillRandom32f(bias, -1.0, 1.0);
        }

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, weight, bias, count, size, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, weight, bias, count, size, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetInnerProductLayerForwardAutoTest(const FuncIPLF & f1, const FuncIPLF & f2)
    {
        bool result = true;

        result = result && SynetInnerProductLayerForwardAutoTest(H, W, ARGS_IPLF(true, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H - O, W + O, ARGS_IPLF(true, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H, W, ARGS_IPLF(false, f1, f2));
        result = result && SynetInnerProductLayerForwardAutoTest(H - O, W + O, ARGS_IPLF(false, f1, f2));

        return result;
    }

    bool SynetInnerProductLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Base::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Sse2::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx2::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Avx512f::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetInnerProductLayerForwardAutoTest(FUNC_IPLF(Simd::Neon::SynetInnerProductLayerForward), FUNC_IPLF(SimdSynetInnerProductLayerForward));
#endif

        return result;
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncIP8I
        {
            typedef void(*FuncPtr)(size_t M, size_t N, size_t K, const uint8_t* src, const int8_t* weight, int32_t* dst, SimdSynetCompatibilityType compatibility);

            FuncPtr func;
            String desc;

            FuncIP8I(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(size_t M, size_t N, size_t K, SimdSynetCompatibilityType c)
            {
                desc = desc + "[" + ToString(M) + "x" + ToString(N) + "x" + ToString(K) + "-"
                    + (Simd::Base::Narrowed(c) ? "n" : Simd::Base::Overflow(c) ? "o" : "p") + "]";
            }

            void Call(const Tensor8u & src, const Tensor8i & weight, Tensor32i& dst, SimdSynetCompatibilityType c) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Axis(0), weight.Axis(0), weight.Axis(1), src.Data(), weight.Data(), dst.Data(), c);
            }
        };
    }

#define FUNC_IP8I(function) FuncIP8I(function, #function)

    bool SynetInnerProduct8iAutoTest(size_t M, size_t N, size_t K, SimdSynetCompatibilityType c, FuncIP8I f1, FuncIP8I f2)
    {
        bool result = true;

        f1.Update(M, N, K, c);
        f2.Update(M, N, K, c);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        int uMin = Simd::Base::Narrowed(c) ? Simd::Base::U8_NARROWED_MIN : Simd::Base::U8_PRECISE_MIN;
        int uMax = Simd::Base::Narrowed(c) ? Simd::Base::U8_NARROWED_MAX : Simd::Base::U8_PRECISE_MAX;
        int iMin = Simd::Base::Narrowed(c) ? Simd::Base::I8_NARROWED_MIN : Simd::Base::I8_PRECISE_MIN;
        int iMax = Simd::Base::Narrowed(c) ? Simd::Base::I8_NARROWED_MAX : Simd::Base::I8_PRECISE_MAX;

        Tensor8u src(Shp(M, K));
        FillRandom(src, uMin, uMax);

        Tensor8i weight(Shp(N, K));
        FillRandom(weight, iMin, iMax);

        Tensor32i dst1(Shp(M, N)), dst2(Shp(M, N));
        FillRandom(dst1, 1, 1);
        FillRandom(dst1, 2, 2);

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, weight, dst1, c));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, weight, dst2, c));

        result = result && Compare(dst1, dst2, 0, true, 64);

        return result;
    }

    bool SynetInnerProduct8iAutoTest(SimdSynetCompatibilityType c, const FuncIP8I& f1, const FuncIP8I& f2)
    {
        bool result = true;

        result = result && SynetInnerProduct8iAutoTest(1, 256, 6912, c, f1, f2);
        result = result && SynetInnerProduct8iAutoTest(10, 256, 6912, c, f1, f2);
        result = result && SynetInnerProduct8iAutoTest(15, 65, 255, c, f1, f2);

        return result;
    }

    bool SynetInnerProduct8iAutoTest(const FuncIP8I& f1, const FuncIP8I& f2)
    {
        bool result = true;

        result = result && SynetInnerProduct8iAutoTest(SimdSynetCompatibility8iPrecise, f1, f2);
        result = result && SynetInnerProduct8iAutoTest(SimdSynetCompatibility8iOverflow, f1, f2);
        result = result && SynetInnerProduct8iAutoTest(SimdSynetCompatibility8iNarrowed, f1, f2);

        return result;
    }

    bool SynetInnerProduct8iAutoTest()
    {
        bool result = true;

        result = result && SynetInnerProduct8iAutoTest(FUNC_IP8I(Simd::Base::SynetInnerProduct8i), FUNC_IP8I(SimdSynetInnerProduct8i));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetInnerProduct8iAutoTest(FUNC_IP8I(Simd::Sse41::SynetInnerProduct8i), FUNC_IP8I(SimdSynetInnerProduct8i));
#endif

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetInnerProduct8iAutoTest(FUNC_IP8I(Simd::Avx2::SynetInnerProduct8i), FUNC_IP8I(SimdSynetInnerProduct8i));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetInnerProduct8iAutoTest(FUNC_IP8I(Simd::Avx512bw::SynetInnerProduct8i), FUNC_IP8I(SimdSynetInnerProduct8i));
#endif

        return result;
    }
#endif
}

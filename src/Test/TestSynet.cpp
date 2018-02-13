/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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

namespace Test
{
    namespace
    {
        struct FuncAB
        {
            typedef void(*FuncPtr)(const float * bias, size_t count, size_t size, float * dst);

            FuncPtr func;
            String desc;

            FuncAB(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Call(const View & bias, size_t count, size_t size, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(desc);
                func((float*)bias.data, count, size, (float*)dstDst.data);
            }
        };
    }

#define FUNC_AB(function) FuncAB(function, #function)

    bool SynetAddBiasAutoTest(size_t count, size_t size, const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstSrc(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(bias, -10.0, 10.0);
        FillRandom32f(dstSrc, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(bias, count, size, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(bias, count, size, dstSrc, dstDst2));

        result = result && Compare(dstDst1, dstDst2, EPS, true, 32, false);

        return result;
    }

    bool SynetAddBiasAutoTest(const FuncAB & f1, const FuncAB & f2)
    {
        bool result = true;

        result = result && SynetAddBiasAutoTest(H, W, f1, f2);
        result = result && SynetAddBiasAutoTest(H - O, W + O, f1, f2);

        return result;
    }

    bool SynetAddBiasAutoTest()
    {
        bool result = true;

        result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Base::SynetAddBias), FUNC_AB(SimdSynetAddBias));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Sse::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetAddBiasAutoTest(FUNC_AB(Simd::Avx512f::SynetAddBias), FUNC_AB(SimdSynetAddBias));
#endif 

        return result;
    }

    namespace
    {
        struct FuncSLF
        {
            typedef void(*FuncPtr)(const float * src, const float * scale, const float * bias, size_t count, size_t size, float * dst);

            FuncPtr func;
            String desc;

            FuncSLF(const FuncPtr & f, const String & d) : func(f), desc(d) {}
            FuncSLF(const FuncSLF & f, bool bias) : func(f.func), desc(f.desc + (bias ? "[1]" : "[0]")) {}

            void Call(const View & src, const View & scale, const View & bias, size_t count, size_t size, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func((float*)src.data, (float*)scale.data, (float*)bias.data, count, size, (float*)dst.data);
            }
        };
    }

#define FUNC_SLF(function) FuncSLF(function, #function)
#define ARGS_SLF(bias, f1, f2) bias, FuncSLF(f1, bias), FuncSLF(f2, bias)

    bool SynetScaleLayerForwardAutoTest(size_t count, size_t size, bool hasBias, const FuncSLF & f1, const FuncSLF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View scale(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias;
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);
        FillRandom32f(scale, -10.0, 10.0);
        if (hasBias)
        {
            bias.Recreate(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
            FillRandom32f(bias, -10.0, 10.0);
        }

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, scale, bias, count, size, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, scale, bias, count, size, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 32, false);

        return result;
    }

    bool SynetScaleLayerForwardAutoTest(const FuncSLF & f1, const FuncSLF & f2)
    {
        bool result = true;

        result = result && SynetScaleLayerForwardAutoTest(H, W, ARGS_SLF(true, f1, f2));
        result = result && SynetScaleLayerForwardAutoTest(H - O, W + O, ARGS_SLF(true, f1, f2));
        result = result && SynetScaleLayerForwardAutoTest(H, W, ARGS_SLF(false, f1, f2));
        result = result && SynetScaleLayerForwardAutoTest(H - O, W + O, ARGS_SLF(false, f1, f2));

        return result;
    }

    bool SynetScaleLayerForwardAutoTest()
    {
        bool result = true;

        result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Base::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Sse::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Avx::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx::Enable)
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Avx2::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && SynetScaleLayerForwardAutoTest(FUNC_SLF(Simd::Avx512f::SynetScaleLayerForward), FUNC_SLF(SimdSynetScaleLayerForward));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool SynetAddBiasDataTest(bool create, size_t count, size_t size, const FuncAB & f)
    {
        bool result = true;

        Data data(f.desc);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.desc << " [" << count << ", " << size << "].");

        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstSrc(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dstDst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(bias, -10.0, 10.0);
            FillRandom32f(dstSrc, -10.0, 10.0);

            TEST_SAVE(bias);
            TEST_SAVE(dstSrc);

            f.Call(bias, count, size, dstSrc, dstDst1);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(bias);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(bias, count, size, dstSrc, dstDst2);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, EPS, true, 32, false);
        }

        return result;
    }

    bool SynetAddBiasDataTest(bool create)
    {
        return SynetAddBiasDataTest(create, DH, DW, FUNC_AB(SimdSynetAddBias));
    }

    bool SynetScaleLayerForwardDataTest(bool create, size_t count, size_t size, const FuncSLF & f)
    {
        bool result = true;

        Data data(f.desc);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.desc << " [" << count << ", " << size << "].");

        View src(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View scale(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View bias(count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(count*size, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(src, -10.0, 10.0);
            FillRandom32f(scale, -10.0, 10.0);
            FillRandom32f(bias, -10.0, 10.0);

            TEST_SAVE(src);
            TEST_SAVE(scale);
            TEST_SAVE(bias);

            f.Call(src, scale, bias, count, size, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(scale);
            TEST_LOAD(bias);

            TEST_LOAD(dst1);

            f.Call(src, scale, bias, count, size, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 32, false);
        }

        return result;
    }

    bool SynetScaleLayerForwardDataTest(bool create)
    {
        return SynetScaleLayerForwardDataTest(create, DH, DW, FUNC_SLF(SimdSynetScaleLayerForward));
    }
}

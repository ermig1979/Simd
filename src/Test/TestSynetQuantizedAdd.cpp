/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
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
#include "Test/TestUtils.h"
#include "Test/TestCompare.h"
#include "Test/TestPerformance.h"
#include "Test/TestTensor.h"
#include "Test/TestRandom.h"
#include "Test/TestString.h"
#include "Test/TestOptions.h"

#include "Simd/SimdSynetQuantizedAdd.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    namespace
    {
        struct FuncQa
        {
            typedef void* (*FuncPtr)(const size_t* aShape, size_t aCount, SimdTensorDataType aType, int32_t aBias, const float* aNorm,
                const size_t* bShape, size_t bCount, SimdTensorDataType bType, int32_t bBias, const float* bNorm,
                SimdConvolutionActivationType actType, const float* actParams, SimdTensorDataType dstType, const float* dstNorm, int32_t dstZero);

            FuncPtr func;
            String desc;

            FuncQa(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(const Shape& as, SimdTensorDataType at, const Shape& bs, SimdTensorDataType bt, SimdConvolutionActivationType act, SimdTensorDataType dt)
            {
                const char* afs[] = { "-id", "-re", "-lr", "-rr", "-pr", "-el", "-hs", "-mi", "-hi", "-sw", "-ge" };
                desc = desc + "[" + ToString(as) + "+" + ToString(bs) + "-" +
                    ToChar(at) + ToChar(bt) + ToChar(dt) + afs[act] + "]";
            }

            void Call(void* context, const uint8_t* a, const uint8_t* b, uint8_t* dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                SimdSynetQuantizedAddForward(context, a, b, dst);
            }
        };
    }

#define FUNC_QA(function) FuncQa(function, #function)

    bool SynetQuantizedAddForwardAutoTest(const Shape& aShape, SimdTensorDataType aType, const Shape& bShape, SimdTensorDataType bType, SimdConvolutionActivationType actType, SimdTensorDataType dstType, FuncQa f1, FuncQa f2)
    {
        bool result = true;

        f1.Update(aShape, aType, bShape, bType, actType, dstType);
        f2.Update(aShape, aType, bShape, bType, actType, dstType);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc);

        Shape dShape = OutputShape(aShape, bShape);
        Tensor32f Af(aShape), Bf(bShape), dst1f(dShape), dst2f(dShape);
        Tensor8u Au(aShape), Bu(bShape), dst1u(dShape), dst2u(dShape);

        FillRandom(Af, -1.1, 1.2f);
        FillRandom(Bf, -1.2, 1.1f);

        FillRandom(Au, 2, 233);
        FillRandom(Bu, 10, 250);

        Fill(dst1f, 1.0f);
        Fill(dst1f, 2.0f);

        Fill(dst1u, uint8_t(1));
        Fill(dst2u, uint8_t(2));

        const uint8_t* A = aType == SimdTensorData32f ? (uint8_t*)Af.Data() : (uint8_t*)Au.Data();
        const uint8_t* B = bType == SimdTensorData32f ? (uint8_t*)Bf.Data() : (uint8_t*)Bu.Data();
        uint8_t* dst1 = dstType == SimdTensorData32f ? (uint8_t*)dst1f.Data() : (uint8_t*)dst1u.Data();
        uint8_t* dst2 = dstType == SimdTensorData32f ? (uint8_t*)dst2f.Data() : (uint8_t*)dst2u.Data();

        int32_t aBias = 47, bBias = 30, dZero = 3;
        float aNorm = 0.01f, bNorm = 0.02f, dNorm = 100.0f, actParams[2] = { 0.0f, 6.0f };

        void* context1 = f1.func(aShape.data(), aShape.size(), aType, aBias, &aNorm, bShape.data(), bShape.size(), bType, bBias, &bNorm, actType, actParams, dstType, &dNorm, dZero);
        void* context2 = f2.func(aShape.data(), aShape.size(), aType, aBias, &aNorm, bShape.data(), bShape.size(), bType, bBias, &bNorm, actType, actParams, dstType, &dNorm, dZero);

        if (context1 == NULL)
            return true;

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(context1, A, B, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(context2, A, B, dst2));

        ::SimdRelease(context1);
        ::SimdRelease(context2);


        if (dstType == SimdTensorData32f)
            result = result && Compare(dst1f, dst2f, EPS, true, 64, DifferenceBoth);
        else
            result = result && Compare(dst1u, dst2u, 0, true, 64);

        return result;
    }

    bool SynetQuantizedAddForwardAutoTest(FuncQa f1, FuncQa f2)
    {
        bool result = true;

        const SimdTensorDataType f32 = SimdTensorData32f, u8 = SimdTensorData8u;
        const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
            aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
            aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
            aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;

        result = result && SynetQuantizedAddForwardAutoTest(Shp(1, 127, 17, 17), u8, Shp(1, 127, 17, 17), u8, aId, u8, f1, f2);
        result = result && SynetQuantizedAddForwardAutoTest(Shp(1, 127, 17, 17), u8, Shp(1, 127, 17, 17), u8, aRe, u8, f1, f2);
        result = result && SynetQuantizedAddForwardAutoTest(Shp(1, 127, 17, 17), u8, Shp(1, 127, 17, 17), u8, aId, f32, f1, f2);

        return result;
    }

    bool SynetQuantizedAddForwardAutoTest(const Options & options)
    {
        bool result = true;

        if (TestBase(options))
            result = result && SynetQuantizedAddForwardAutoTest(FUNC_QA(Simd::Base::SynetQuantizedAddInit), FUNC_QA(SimdSynetQuantizedAddInit));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && TestSse41(options))
            result = result && SynetQuantizedAddForwardAutoTest(FUNC_QA(Simd::Sse41::SynetQuantizedAddInit), FUNC_QA(SimdSynetQuantizedAddInit));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && TestAvx2(options))
            result = result && SynetQuantizedAddForwardAutoTest(FUNC_QA(Simd::Avx2::SynetQuantizedAddInit), FUNC_QA(SimdSynetQuantizedAddInit));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && TestAvx512bw(options))
            result = result && SynetQuantizedAddForwardAutoTest(FUNC_QA(Simd::Avx512bw::SynetQuantizedAddInit), FUNC_QA(SimdSynetQuantizedAddInit));
#endif 

        return result;
    }
#endif
}

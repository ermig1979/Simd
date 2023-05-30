/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Test/TestString.h"
#include "Test/TestRandom.h"

#include "Simd/SimdSynet.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
    SIMD_INLINE String ToString(SimdSynetUnaryOperation32fType type)
    {
        switch (type)
        {
        case SimdSynetUnaryOperation32fAbs:
            return "Abs";
        case SimdSynetUnaryOperation32fErf:
            return "Erf";
        case SimdSynetUnaryOperation32fExp:
            return "Exp";
        case SimdSynetUnaryOperation32fLog:
            return "Log";
        case SimdSynetUnaryOperation32fNeg:
            return "Neg";
        case SimdSynetUnaryOperation32fNot:
            return "Not";
        case SimdSynetUnaryOperation32fRcp:
            return "Rcp";
        case SimdSynetUnaryOperation32fRsqrt:
            return "Rsqrt";
        case SimdSynetUnaryOperation32fSqrt:
            return "Sqrt";
        case SimdSynetUnaryOperation32fTanh:
            return "Tanh";
        case SimdSynetUnaryOperation32fZero:
            return "Zero";
        }
        assert(0);
        return "???";
    }

    namespace
    {
        struct FuncUO
        {
            typedef void(*FuncPtr)(const float* src, size_t size, SimdSynetUnaryOperation32fType type, float* dst);

            FuncPtr func;
            String desc;

            FuncUO(const FuncPtr & f, const String& d) : func(f), desc(d) {}

            void Update(SimdSynetUnaryOperation32fType type)
            {
                desc = desc + "[" + ToString(type) + "]";
            }

            void Call(const Tensor32f& src, SimdSynetUnaryOperation32fType type, Tensor32f& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.Data(), src.Size(), type, dst.Data());
            }
        };
    }

#define FUNC_UO(function) FuncUO(function, #function)

    bool SynetUnaryOperation32fAutoTest(size_t size, float eps, SimdSynetUnaryOperation32fType type, FuncUO f1, FuncUO f2)
    {
        bool result = true;

        f1.Update(type);
        f2.Update(type);

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << ".");

        Tensor32f src({ size });
        float lo = -200.0, hi = 10.0f;
        if (type == SimdSynetUnaryOperation32fLog || type == SimdSynetUnaryOperation32fRcp || type == SimdSynetUnaryOperation32fRsqrt || type == SimdSynetUnaryOperation32fSqrt)
            lo = 0.000001f;
        FillRandom(src.Data(), src.Size(), lo, hi);
        if (type == SimdSynetUnaryOperation32fNot)
        {
            float* ptr = src.Data();
            for (size_t i = 0, n = src.Size(); i < n; ++i)
                ptr[i] = Simd::Max(0.0f, ptr[i]);
        }

        Tensor32f dst1({ size });
        Tensor32f dst2({ size });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, type, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, type, dst2));

        if (type == SimdSynetUnaryOperation32fNot)
        {
#if defined(_MSC_VER) && defined(NDEBUG)
#else
            result = result && Compare(dst1, dst2, eps, true, 64, DifferenceLogical);
#endif
        }
        else
            result = result && Compare(dst1, dst2, eps, true, 64, DifferenceBoth);

        return result;
    }

    bool SynetUnaryOperation32fAutoTest(const FuncUO& f1, const FuncUO& f2)
    {
        bool result = true;

#if 0
        result = SynetUnaryOperation32fAutoTest(H * W, 0.000001f, SimdSynetUnaryOperation32fErf, f1, f2);
#endif
        for (int type = (int)SimdSynetUnaryOperation32fAbs; type <= (int)SimdSynetUnaryOperation32fZero; type++)
        {
            result = result && SynetUnaryOperation32fAutoTest(H*W, EPS, (SimdSynetUnaryOperation32fType)type, f1, f2);
            result = result && SynetUnaryOperation32fAutoTest(H*W + O, EPS, (SimdSynetUnaryOperation32fType)type, f1, f2);
        }

        return result;
    }

    bool SynetUnaryOperation32fAutoTest()
    {
        bool result = true;

        result = result && SynetUnaryOperation32fAutoTest(FUNC_UO(Simd::Base::SynetUnaryOperation32f), FUNC_UO(SimdSynetUnaryOperation32f));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && SynetUnaryOperation32fAutoTest(FUNC_UO(Simd::Sse41::SynetUnaryOperation32f), FUNC_UO(SimdSynetUnaryOperation32f));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && SynetUnaryOperation32fAutoTest(FUNC_UO(Simd::Avx2::SynetUnaryOperation32f), FUNC_UO(SimdSynetUnaryOperation32f));
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && SynetUnaryOperation32fAutoTest(FUNC_UO(Simd::Avx512bw::SynetUnaryOperation32f), FUNC_UO(SimdSynetUnaryOperation32f));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && SynetUnaryOperation32fAutoTest(FUNC_UO(Simd::Neon::SynetUnaryOperation32f), FUNC_UO(SimdSynetUnaryOperation32f));
#endif 

        return result;
    }
#endif
}

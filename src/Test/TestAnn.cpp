/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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
        struct FuncPS
        {
            typedef void (*FuncPtr)(const float * a, const float * b, size_t size, float * sum);

            FuncPtr func;
            std::string description;

            FuncPS(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & a, const View & b, float * sum) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)a.data, (float*)b.data, a.width, sum);
            }
        };
	}
#define FUNC_PS(function) FuncPS(function, #function)

    bool AnnProductSumAutoTest(int size, float eps, const FuncPS & f1, const FuncPS & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View a(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(a);

        View b(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(b);

        float s1, s2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, &s1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, &s2));

        result = Compare(s1, s2, eps, true);

        return result;
    }

    bool AnnProductSumAutoTest(float eps, const FuncPS & f1, const FuncPS & f2)
    {
        bool result = true;

        result = result && AnnProductSumAutoTest(W*H, eps, f1, f2);
        result = result && AnnProductSumAutoTest(W*H + O, eps, f1, f2);
        result = result && AnnProductSumAutoTest(W*H - O, eps, f1, f2);

        return result;
    }

    bool AnnProductSumAutoTest()
    {
        bool result = true;

        result = result && AnnProductSumAutoTest(EPS, FUNC_PS(Simd::Base::AnnProductSum), FUNC_PS(SimdAnnProductSum));

#ifdef SIMD_SSE_ENABLE
		if (Simd::Sse::Enable)
			result = result && AnnProductSumAutoTest(EPS, FUNC_PS(Simd::Sse::AnnProductSum), FUNC_PS(SimdAnnProductSum));
#endif 

#ifdef SIMD_AVX_ENABLE
		if (Simd::Avx::Enable)
			result = result && AnnProductSumAutoTest(EPS, FUNC_PS(Simd::Avx::AnnProductSum), FUNC_PS(SimdAnnProductSum));
#endif

#ifdef SIMD_VSX_ENABLE
		if (Simd::Vsx::Enable)
			result = result && AnnProductSumAutoTest(EPS, FUNC_PS(Simd::Vsx::AnnProductSum), FUNC_PS(SimdAnnProductSum));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool AnnProductSumDataTest(bool create, int size, float eps, const FuncPS & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View a(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View b(size, 1, View::Float, NULL, TEST_ALIGN(size));

        float s1, s2;

        if(create)
        {
            FillRandom32f(a);
            FillRandom32f(b);

            TEST_SAVE(a);
            TEST_SAVE(b);

            f.Call(a, b, &s1);

            TEST_SAVE(s1);
        }
        else
        {
            TEST_LOAD(a);
            TEST_LOAD(b);

            TEST_LOAD(s1);

            f.Call(a, b, &s2);

            TEST_SAVE(s2);

            result = result && Compare(s1, s2, eps, true);
        }

        return result;
    }

    bool AnnProductSumDataTest(bool create)
    {
        bool result = true;

        result = result && AnnProductSumDataTest(create, DH, EPS, FUNC_PS(SimdAnnProductSum));

        return result;
    }
}

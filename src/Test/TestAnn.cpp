/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2016 Yermalayeu Ihar.
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
	const float ROUGH_SIGMOID_ERROR = 0.0025f;

	namespace
	{
		struct FuncC
		{
			typedef void(*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion);

			FuncPtr func;
			std::string description;
			bool inversion;

			FuncC(const FuncPtr & f, const std::string & d, bool i) : func(f), description(d + (i ? "[1]" : "[0]")), inversion(i) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, (float*)dst.data, inversion ? 1 : 0);
			}
		};
	}
#define FUNC_C(function, inversion) FuncC(function, #function, inversion)

	bool AnnConvertAutoTest(int width, int height, float eps, const FuncC & f1, const FuncC & f2)
	{
		bool result = true;

		TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

		View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(src);

		View dst1(width*height, 1, View::Float, NULL, TEST_ALIGN(width));
		View dst2(width*height, 1, View::Float, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

		result = Compare(dst1, dst2, eps, true, 32);

		return result;
	}

	bool AnnConvertAutoTest(float eps, const FuncC & f1, const FuncC & f2)
	{
		bool result = true;

		result = result && AnnConvertAutoTest(W, H, eps, f1, f2);
		result = result && AnnConvertAutoTest(W - O, H + O, eps, f1, f2);
		result = result && AnnConvertAutoTest(W + O, H - O, eps, f1, f2);

		return result;
	}

	bool AnnConvertAutoTest()
	{
		bool result = true;

		result = result && AnnConvertAutoTest(EPS, FUNC_C(Simd::Base::AnnConvert, true), FUNC_C(SimdAnnConvert, true));
		result = result && AnnConvertAutoTest(EPS, FUNC_C(Simd::Base::AnnConvert, false), FUNC_C(SimdAnnConvert, false));

#ifdef SIMD_SSE2_ENABLE
		if (Simd::Sse2::Enable)
		{
			result = result && AnnConvertAutoTest(EPS, FUNC_C(Simd::Sse2::AnnConvert, true), FUNC_C(SimdAnnConvert, true));
			result = result && AnnConvertAutoTest(EPS, FUNC_C(Simd::Sse2::AnnConvert, false), FUNC_C(SimdAnnConvert, false));
		}
#endif 

#ifdef SIMD_AVX2_ENABLE
		if (Simd::Avx2::Enable)
		{
			result = result && AnnConvertAutoTest(EPS, FUNC_C(Simd::Avx2::AnnConvert, true), FUNC_C(SimdAnnConvert, true));
			result = result && AnnConvertAutoTest(EPS, FUNC_C(Simd::Avx2::AnnConvert, false), FUNC_C(SimdAnnConvert, false));
		}
#endif

#ifdef SIMD_VSX_ENABLE
		if (Simd::Vsx::Enable)
		{
			result = result && AnnConvertAutoTest(EPS, FUNC_C(Simd::Vsx::AnnConvert, true), FUNC_C(SimdAnnConvert, true));
			result = result && AnnConvertAutoTest(EPS, FUNC_C(Simd::Vsx::AnnConvert, false), FUNC_C(SimdAnnConvert, false));
		}
#endif 

		return result;
	}

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

	namespace
	{
		struct FuncS
		{
			typedef void(*FuncPtr)(const float * src, size_t size, const float * slope, float * dst);

			FuncPtr func;
			std::string description;

			FuncS(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, float slope, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func((float*)src.data, src.width, &slope, (float*)dst.data);
			}
		};
	}
#define FUNC_S(function) FuncS(function, #function)

	bool AnnSigmoidAutoTest(int size, float error, bool relative, const FuncS & f1, const FuncS & f2)
	{
		bool result = true;

		TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

		View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
		FillRandom32f(src, -10.0f, 10.0f);

		float slope = 3.0;

		View dst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
		View dst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, slope, dst1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, slope, dst2));

		result = Compare(dst1, dst2, error, true, 32, relative);

		return result;
	}

	bool AnnSigmoidAutoTest(float error, bool relative, const FuncS & f1, const FuncS & f2)
	{
		bool result = true;

		result = result && AnnSigmoidAutoTest(W*H, error, relative, f1, f2);
		result = result && AnnSigmoidAutoTest(W*H + O, error, relative, f1, f2);
		result = result && AnnSigmoidAutoTest(W*H - O, error, relative, f1, f2);

		return result;
	}

	bool AnnSigmoidAutoTest()
	{
		bool result = true;

		result = result && AnnSigmoidAutoTest(EPS, true, FUNC_S(Simd::Base::AnnSigmoid), FUNC_S(SimdAnnSigmoid));

		return result;
	}

	bool AnnRoughSigmoidAutoTest()
	{
		bool result = true;

		result = result && AnnSigmoidAutoTest(EPS, true, FUNC_S(Simd::Base::AnnRoughSigmoid), FUNC_S(SimdAnnRoughSigmoid));

#ifdef SIMD_SSE_ENABLE
		if (Simd::Sse::Enable)
			result = result && AnnSigmoidAutoTest(EPS, true, FUNC_S(Simd::Sse::AnnRoughSigmoid), FUNC_S(SimdAnnRoughSigmoid));
#endif 

#ifdef SIMD_AVX_ENABLE
		if (Simd::Avx::Enable)
			result = result && AnnSigmoidAutoTest(EPS, true, FUNC_S(Simd::Avx::AnnRoughSigmoid), FUNC_S(SimdAnnRoughSigmoid));
#endif

#ifdef SIMD_VSX_ENABLE
		if (Simd::Vsx::Enable)
			result = result && AnnSigmoidAutoTest(EPS, true, FUNC_S(Simd::Vsx::AnnRoughSigmoid), FUNC_S(SimdAnnRoughSigmoid));
#endif

		return result;
	}

    //-----------------------------------------------------------------------

	bool AnnConvertDataTest(bool create, int width, int height, float eps, const FuncC & f)
	{
		bool result = true;

		Data data(f.description);

		TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

		View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(src);

		View dst1(width*height, 1, View::Float, NULL, TEST_ALIGN(width));
		View dst2(width*height, 1, View::Float, NULL, TEST_ALIGN(width));

		if (create)
		{
			FillRandom(src);

			TEST_SAVE(src);

			f.Call(src, dst1);

			TEST_SAVE(dst1);
		}
		else
		{
			TEST_LOAD(src);

			TEST_LOAD(dst1);

			f.Call(src, dst2);

			TEST_SAVE(dst2);

			result = Compare(dst1, dst2, eps, true);
		}

		return result;
	}

	bool AnnConvertDataTest(bool create)
	{
		bool result = true;

		result = result && AnnConvertDataTest(create, DW, DH, EPS, FUNC_C(SimdAnnConvert, true));

		return result;
	}

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


	bool AnnSigmoidDataTest(bool create, int size, float error, bool relative, const FuncS & f)
	{
		bool result = true;

		Data data(f.description);

		TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

		View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
		View dst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
		View dst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

		float slope = 3.0;

		if (create)
		{
			FillRandom32f(src, -10.0f, 10.0f);

			TEST_SAVE(src);

			f.Call(src, slope, dst1);

			TEST_SAVE(dst1);
		}
		else
		{
			TEST_LOAD(src);

			TEST_LOAD(dst1);

			f.Call(src, slope, dst2);

			TEST_SAVE(dst2);

			result = Compare(dst1, dst2, error, true, 0, relative);
		}

		return result;
	}

	bool AnnSigmoidDataTest(bool create)
	{
		bool result = true;

		result = result && AnnSigmoidDataTest(create, DH, EPS, true, FUNC_S(SimdAnnSigmoid));

		return result;
	}

	bool AnnRoughSigmoidDataTest(bool create)
	{
		bool result = true;

		result = result && AnnSigmoidDataTest(create, DH, EPS, true, FUNC_S(SimdAnnRoughSigmoid));

		return result;
	}
}

/*
* Simd Library Tests.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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

namespace Test
{
	namespace
	{
		struct Func
		{
			typedef void (*FuncPtr)(const uchar * src, size_t srcStride, size_t width, size_t height, 
				uchar value, uchar positive, uchar negative, uchar * dst, size_t dstStride, Simd::CompareType type);

			FuncPtr func;
			std::string description;

			Func(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & src, uchar value, uchar positive, uchar negative, View & dst, Simd::CompareType type) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, value, positive, negative, dst.data, dst.stride, type);
			}
		};
	}

    SIMD_INLINE std::string CompareTypeDescription(Simd::CompareType type)
    {
        switch(type)
        {
        case Simd::CompareGreaterThen:
            return "(>)";
        case Simd::CompareLesserThen:
            return "(<)";
        case Simd::CompareEqualTo:
            return "(=)";
        }
        assert(0);
        return "(Unknown)";
    }

#define ARGS1(width, height, type, function1, function2) \
    width, height, type, \
    Func(function1.func, function1.description + CompareTypeDescription(type)), \
    Func(function2.func, function2.description + CompareTypeDescription(type))

#define ARGS2(function1, function2) \
    Func(function1, std::string(#function1)), Func(function2, std::string(#function2))


	bool BinarizationTest(int width, int height, Simd::CompareType type, const Func & f1, const Func & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View src(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		FillRandom(src);

		uchar value = Random(256);
		uchar positive = Random(256);
		uchar negative = Random(256);

		View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
		View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, positive, negative, d1, type));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, positive, negative, d2, type));

		result = result && Compare(d1, d2, 0, true, 10);

		return result;
	}

    bool BinarizationTest(const Func & f1, const Func & f2)
    {
        bool result = true;

        for(Simd::CompareType type = Simd::CompareGreaterThen; type <= Simd::CompareEqualTo && result; type = Simd::CompareType(type + 1))
        {
            result = result && BinarizationTest(ARGS1(W, H, type, f1, f2));
            result = result && BinarizationTest(ARGS1(W + 1, H - 1, type, f1, f2));
        }

        return result;
    }

	bool BinarizationTest()
	{
		bool result = true;

		result = result && BinarizationTest(ARGS2(Simd::Base::Binarization, Simd::Binarization));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && BinarizationTest(ARGS2(Simd::Avx2::Binarization, Simd::Sse2::Binarization));
#endif 

		return result;
	}
}

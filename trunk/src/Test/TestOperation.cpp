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
#include "Test/Test.h"

namespace Test
{
	namespace
	{
		struct FuncO
		{
			typedef void (*FuncPtr)(const uchar * a, size_t aStride, const uchar * b, size_t bStride,
				size_t width, size_t height, size_t channelCount, uchar * dst, size_t dstStride, Simd::OperationType type);

			FuncPtr func;
			std::string description;

			FuncO(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, View & dst, Simd::OperationType type) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(a.data, a.stride, b.data, b.stride, a.width, a.height, View::PixelSize(a.format), dst.data, dst.stride, type);
			}
		};

        struct FuncVP
        {
            typedef void (*FuncPtr)(const uchar * vertical, const uchar * horizontal, uchar * dst, size_t stride, size_t width, size_t height);

            FuncPtr func;
            std::string description;

            FuncVP(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & v, const View & h, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(v.data, h.data, dst.data, dst.stride, dst.width, dst.height);
            }
        };
	}

    SIMD_INLINE std::string OperationTypeDescription(Simd::OperationType type)
    {
        switch(type)
        {
        case Simd::OperationAverage:
            return "<Avg>";
        case Simd::OperationAnd:
            return "<And>";
        case Simd::OperationMaximum:
            return "<Max>";
        case Simd::OperationSaturatedSubtraction:
            return "<Subs>";
        }
		assert(0);
		return "<Unknown";
    }

#define ARGS_O1(format, width, height, type, function1, function2) \
	format, width, height, type, \
	FuncO(function1.func, function1.description + OperationTypeDescription(type) + ColorDescription(format)), \
	FuncO(function2.func, function2.description + OperationTypeDescription(type) + ColorDescription(format))

#define ARGS_O2(function1, function2) \
	FuncO(function1, std::string(#function1)), FuncO(function2, std::string(#function2))

#define ARGS_VP(function) FuncVP(function, std::string(#function))

	bool OperationTest(View::Format format, int width, int height, Simd::OperationType type, const FuncO & f1, const FuncO & f2)
	{
		bool result = true;

		std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

		View a(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(a);

		View b(width, height, format, NULL, TEST_ALIGN(width));
		FillRandom(b);

		View d1(width, height, format, NULL, TEST_ALIGN(width));
		View d2(width, height, format, NULL, TEST_ALIGN(width));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, d1, type));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, d2, type));

		result = result && Compare(d1, d2, 0, true, 64);

		return result;
	}

	bool OperationTest(const FuncO & f1, const FuncO & f2)
	{
		bool result = true;

        for(Simd::OperationType type = Simd::OperationAverage; type <= Simd::OperationSaturatedSubtraction && result; type = Simd::OperationType(type + 1))
        {
            for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
            {
                result = result && OperationTest(ARGS_O1(format, W, H, type, f1, f2));
                result = result && OperationTest(ARGS_O1(format, W + 1, H - 1, type, f1, f2));
                result = result && OperationTest(ARGS_O1(format, W - 1, H + 1, type, f1, f2));
            }
        }

		return result;
	}

	bool OperationTest()
	{
		bool result = true;

		result = result && OperationTest(ARGS_O2(Simd::Base::Operation, Simd::Operation));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
		if(Simd::Sse2::Enable && Simd::Avx2::Enable)
			result = result && OperationTest(ARGS_O2(Simd::Avx2::Operation, Simd::Sse2::Operation));
#endif 

		return result;
	}

    bool VectorProductTest(int width, int height, const FuncVP & f1, const FuncVP & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View v(height, 1, View::Gray8, NULL, TEST_ALIGN(height));
        FillRandom(v);
        View h(width, 1, View::Gray8, NULL, TEST_ALIGN(width));
        FillRandom(h);

        View d1(width, height, View::Gray8, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Gray8, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(v, h, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(v, h, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool VectorProductTest()
    {
        bool result = true;

        result = result && VectorProductTest(W, H, ARGS_VP(Simd::Base::VectorProduct), ARGS_VP(Simd::VectorProduct));
        result = result && VectorProductTest(W - 1, H + 1, ARGS_VP(Simd::Base::VectorProduct), ARGS_VP(Simd::VectorProduct));
        result = result && VectorProductTest(W + 1, H - 1, ARGS_VP(Simd::Base::VectorProduct), ARGS_VP(Simd::VectorProduct));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
        {
            result = result && VectorProductTest(W, H, ARGS_VP(Simd::Avx2::VectorProduct), ARGS_VP(Simd::Sse2::VectorProduct));
            result = result && VectorProductTest(W - 1, H + 1, ARGS_VP(Simd::Avx2::VectorProduct), ARGS_VP(Simd::Sse2::VectorProduct));
            result = result && VectorProductTest(W + 1, H - 1, ARGS_VP(Simd::Avx2::VectorProduct), ARGS_VP(Simd::Sse2::VectorProduct));
        }
#endif 

        return result;
    }
}

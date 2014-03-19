/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
		struct FuncOB8U
		{
			typedef void (*FuncPtr)(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
				size_t width, size_t height, size_t channelCount, uint8_t * dst, size_t dstStride, SimdOperationBinary8uType type);

			FuncPtr func;
			std::string description;

			FuncOB8U(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

			void Call(const View & a, const View & b, View & dst, SimdOperationBinary8uType type) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(a.data, a.stride, b.data, b.stride, a.width, a.height, View::PixelSize(a.format), dst.data, dst.stride, type);
			}
		};

        struct FuncOB16I
        {
            typedef void (*FuncPtr)(const uint8_t * a, size_t aStride, const uint8_t * b, size_t bStride,
                size_t width, size_t height, uint8_t * dst, size_t dstStride, SimdOperationBinary16iType type);

            FuncPtr func;
            std::string description;

            FuncOB16I(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const View & a, const View & b, View & dst, SimdOperationBinary16iType type) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(a.data, a.stride, b.data, b.stride, a.width, a.height, dst.data, dst.stride, type);
            }
        };

        struct FuncVP
        {
            typedef void (*FuncPtr)(const uint8_t * vertical, const uint8_t * horizontal, uint8_t * dst, size_t stride, size_t width, size_t height);

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

    SIMD_INLINE std::string OperationBinary8uTypeDescription(SimdOperationBinary8uType type)
    {
        switch(type)
        {
        case SimdOperationBinary8uAverage:
            return "<Avg>";
        case SimdOperationBinary8uAnd:
            return "<And>";
        case SimdOperationBinary8uMaximum:
            return "<Max>";
        case SimdOperationBinary8uSaturatedSubtraction:
            return "<Subs>";
        }
		assert(0);
		return "<Unknown";
    }

    SIMD_INLINE std::string OperationBinary16iTypeDescription(SimdOperationBinary16iType type)
    {
        switch(type)
        {
        case SimdOperationBinary16iAddition:
            return "<Add>";
        }
        assert(0);
        return "<Unknown";
    }

#define ARGS_OB8U1(format, width, height, type, function1, function2) \
	format, width, height, type, \
	FuncOB8U(function1.func, function1.description + OperationBinary8uTypeDescription(type) + ColorDescription(format)), \
	FuncOB8U(function2.func, function2.description + OperationBinary8uTypeDescription(type) + ColorDescription(format))

#define ARGS_OB8U2(function1, function2) \
	FuncOB8U(function1, std::string(#function1)), FuncOB8U(function2, std::string(#function2))

#define ARGS_OB16I1(width, height, type, function1, function2) \
    width, height, type, \
    FuncOB16I(function1.func, function1.description + OperationBinary16iTypeDescription(type)), \
    FuncOB16I(function2.func, function2.description + OperationBinary16iTypeDescription(type))

#define ARGS_OB16I2(function1, function2) \
    FuncOB16I(function1, std::string(#function1)), FuncOB16I(function2, std::string(#function2))

#define ARGS_VP(function) FuncVP(function, std::string(#function))

	bool OperationBinary8uTest(View::Format format, int width, int height, SimdOperationBinary8uType type, const FuncOB8U & f1, const FuncOB8U & f2)
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

	bool OperationBinary8uTest(const FuncOB8U & f1, const FuncOB8U & f2)
	{
		bool result = true;

        for(SimdOperationBinary8uType type = SimdOperationBinary8uAverage; type <= SimdOperationBinary8uSaturatedSubtraction && result; type = SimdOperationBinary8uType(type + 1))
        {
            for(View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
            {
                result = result && OperationBinary8uTest(ARGS_OB8U1(format, W, H, type, f1, f2));
                result = result && OperationBinary8uTest(ARGS_OB8U1(format, W + 1, H - 1, type, f1, f2));
                result = result && OperationBinary8uTest(ARGS_OB8U1(format, W - 1, H + 1, type, f1, f2));
            }
        }

		return result;
	}

	bool OperationBinary8uTest()
	{
		bool result = true;

		result = result && OperationBinary8uTest(ARGS_OB8U2(Simd::Base::OperationBinary8u, SimdOperationBinary8u));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
		if(Simd::Sse2::Enable && Simd::Avx2::Enable)
			result = result && OperationBinary8uTest(ARGS_OB8U2(Simd::Avx2::OperationBinary8u, Simd::Sse2::OperationBinary8u));
#endif 

		return result;
	}

    bool OperationBinary16iTest(int width, int height, SimdOperationBinary16iType type, const FuncOB16I & f1, const FuncOB16I & f2)
    {
        bool result = true;

        std::cout << "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "]." << std::endl;

        View a(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(a);

        View b(width, height, View::Int16, NULL, TEST_ALIGN(width));
        FillRandom(b);

        View d1(width, height, View::Int16, NULL, TEST_ALIGN(width));
        View d2(width, height, View::Int16, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(a, b, d1, type));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(a, b, d2, type));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool OperationBinary16iTest(const FuncOB16I & f1, const FuncOB16I & f2)
    {
        bool result = true;

        for(SimdOperationBinary16iType type = SimdOperationBinary16iAddition; type <= SimdOperationBinary16iAddition && result; type = SimdOperationBinary16iType(type + 1))
        {
            result = result && OperationBinary16iTest(ARGS_OB16I1(W, H, type, f1, f2));
            result = result && OperationBinary16iTest(ARGS_OB16I1(W + 1, H - 1, type, f1, f2));
            result = result && OperationBinary16iTest(ARGS_OB16I1(W - 1, H + 1, type, f1, f2));
        }

        return result;
    }

    bool OperationBinary16iTest()
    {
        bool result = true;

        result = result && OperationBinary16iTest(ARGS_OB16I2(Simd::Base::OperationBinary16i, SimdOperationBinary16i));

#if defined(SIMD_SSE2_ENABLE) && defined(SIMD_AVX2_ENABLE)
        if(Simd::Sse2::Enable && Simd::Avx2::Enable)
            result = result && OperationBinary16iTest(ARGS_OB16I2(Simd::Avx2::OperationBinary16i, Simd::Sse2::OperationBinary16i));
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

        result = result && VectorProductTest(W, H, ARGS_VP(Simd::Base::VectorProduct), ARGS_VP(SimdVectorProduct));
        result = result && VectorProductTest(W - 1, H + 1, ARGS_VP(Simd::Base::VectorProduct), ARGS_VP(SimdVectorProduct));
        result = result && VectorProductTest(W + 1, H - 1, ARGS_VP(Simd::Base::VectorProduct), ARGS_VP(SimdVectorProduct));

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

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
		struct FuncC1
		{
			typedef void(*FuncPtr)(const uint8_t * src, size_t stride, size_t width, size_t height, float * dst, int inversion);

			FuncPtr func;
			String description;
			bool inversion;

			FuncC1(const FuncPtr & f, const String & d, bool i) : func(f), description(d + (i ? "[1]" : "[0]")), inversion(i) {}

			void Call(const View & src, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func(src.data, src.stride, src.width, src.height, (float*)dst.data, inversion ? 1 : 0);
			}
		};
	}
#define FUNC_C1(function, inversion) FuncC1(function, #function, inversion)

	bool AnnConvertAutoTest(int width, int height, float eps, const FuncC1 & f1, const FuncC1 & f2)
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

	bool AnnConvertAutoTest(float eps, const FuncC1 & f1, const FuncC1 & f2)
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

		result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Base::AnnConvert, true), FUNC_C1(SimdAnnConvert, true));
		result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Base::AnnConvert, false), FUNC_C1(SimdAnnConvert, false));

#ifdef SIMD_SSE2_ENABLE
		if (Simd::Sse2::Enable)
		{
			result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Sse2::AnnConvert, true), FUNC_C1(SimdAnnConvert, true));
			result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Sse2::AnnConvert, false), FUNC_C1(SimdAnnConvert, false));
		}
#endif 

#ifdef SIMD_AVX2_ENABLE
		if (Simd::Avx2::Enable)
		{
			result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Avx2::AnnConvert, true), FUNC_C1(SimdAnnConvert, true));
			result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Avx2::AnnConvert, false), FUNC_C1(SimdAnnConvert, false));
		}
#endif

#ifdef SIMD_VSX_ENABLE
		if (Simd::Vsx::Enable)
		{
			result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Vsx::AnnConvert, true), FUNC_C1(SimdAnnConvert, true));
			result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Vsx::AnnConvert, false), FUNC_C1(SimdAnnConvert, false));
		}
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
        {
            result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Neon::AnnConvert, true), FUNC_C1(SimdAnnConvert, true));
            result = result && AnnConvertAutoTest(EPS, FUNC_C1(Simd::Neon::AnnConvert, false), FUNC_C1(SimdAnnConvert, false));
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
            String description;

            FuncPS(const FuncPtr & f, const String & d) : func(f), description(d) {}

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

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && AnnProductSumAutoTest(EPS, FUNC_PS(Simd::Neon::AnnProductSum), FUNC_PS(SimdAnnProductSum));
#endif

        return result;
    }

    namespace
    {
        struct FuncAVMV
        {
            typedef void(*FuncPtr)(const float * src, size_t size, const float * value, float * dst);

            FuncPtr func;
            String description;

            FuncAVMV(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, float value, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.width, &value, (float*)dstDst.data);
            }
        };
    }
#define FUNC_AVMV(function) FuncAVMV(function, #function)

    bool AnnAddVectorMultipliedByValueAutoTest(int size, float eps, const FuncAVMV & f1, const FuncAVMV & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(src);

        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(dstSrc);

        const float value = 0.3f;

        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, value, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, value, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, eps, true);

        return result;
    }

    bool AnnAddVectorMultipliedByValueAutoTest(float eps, const FuncAVMV & f1, const FuncAVMV & f2)
    {
        bool result = true;

        result = result && AnnAddVectorMultipliedByValueAutoTest(W*H, eps, f1, f2);
        result = result && AnnAddVectorMultipliedByValueAutoTest(W*H + O, eps, f1, f2);
        result = result && AnnAddVectorMultipliedByValueAutoTest(W*H - O, eps, f1, f2);

        return result;
    }

    bool AnnAddVectorMultipliedByValueAutoTest()
    {
        bool result = true;

        result = result && AnnAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Base::AnnAddVectorMultipliedByValue), FUNC_AVMV(SimdAnnAddVectorMultipliedByValue));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Sse::AnnAddVectorMultipliedByValue), FUNC_AVMV(SimdAnnAddVectorMultipliedByValue));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnAddVectorMultipliedByValueAutoTest(EPS, FUNC_AVMV(Simd::Avx::AnnAddVectorMultipliedByValue), FUNC_AVMV(SimdAnnAddVectorMultipliedByValue));
#endif

        return result;
    }

	namespace
	{
		struct FuncA
		{
			typedef void(*FuncPtr)(const float * src, size_t size, const float * slope, float * dst);

			FuncPtr func;
			String description;

			FuncA(const FuncPtr & f, const String & d) : func(f), description(d) {}

			void Call(const View & src, float slope, View & dst) const
			{
				TEST_PERFORMANCE_TEST(description);
				func((float*)src.data, src.width, &slope, (float*)dst.data);
			}
		};
	}
#define FUNC_A(function) FuncA(function, #function)

	bool AnnActivateFunctionAutoTest(int size, float error, bool relative, float slope, const FuncA & f1, const FuncA & f2)
	{
		bool result = true;

		TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

		View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
		FillRandom32f(src, -10.0f, 10.0f);

		View dst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
		View dst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, slope, dst1));

		TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, slope, dst2));

		result = Compare(dst1, dst2, error, true, 32, relative);

		return result;
	}

	bool AnnActivateFunctionAutoTest(float error, bool relative, float slope, const FuncA & f1, const FuncA & f2)
	{
		bool result = true;

		result = result && AnnActivateFunctionAutoTest(W*H, error, relative, slope, f1, f2);
		result = result && AnnActivateFunctionAutoTest(W*H + O, error, relative, slope, f1, f2);
		result = result && AnnActivateFunctionAutoTest(W*H - O, error, relative, slope, f1, f2);

		return result;
	}

	bool AnnSigmoidAutoTest()
	{
		bool result = true;

		result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Base::AnnSigmoid), FUNC_A(SimdAnnSigmoid));

		return result;
	}

	bool AnnRoughSigmoidAutoTest()
	{
		bool result = true;

		result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Base::AnnRoughSigmoid), FUNC_A(SimdAnnRoughSigmoid));

#ifdef SIMD_SSE_ENABLE
		if (Simd::Sse::Enable)
			result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Sse::AnnRoughSigmoid), FUNC_A(SimdAnnRoughSigmoid));
#endif 

#ifdef SIMD_AVX_ENABLE
		if (Simd::Avx::Enable)
			result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Avx::AnnRoughSigmoid), FUNC_A(SimdAnnRoughSigmoid));
#endif

#ifdef SIMD_VSX_ENABLE
		if (Simd::Vsx::Enable)
			result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Vsx::AnnRoughSigmoid), FUNC_A(SimdAnnRoughSigmoid));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Neon::AnnRoughSigmoid), FUNC_A(SimdAnnRoughSigmoid));
#endif

		return result;
	}

    bool AnnDerivativeSigmoidAutoTest()
    {
        bool result = true;

        result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Base::AnnDerivativeSigmoid), FUNC_A(SimdAnnDerivativeSigmoid));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Sse::AnnDerivativeSigmoid), FUNC_A(SimdAnnDerivativeSigmoid));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Avx::AnnDerivativeSigmoid), FUNC_A(SimdAnnDerivativeSigmoid));
#endif

        return result;
    }


    bool AnnTanhAutoTest()
    {
        bool result = true;

        result = result && AnnActivateFunctionAutoTest(EPS, false, 3.0f, FUNC_A(Simd::Base::AnnTanh), FUNC_A(SimdAnnTanh));

        return result;
    }

    bool AnnRoughTanhAutoTest()
    {
        bool result = true;

        result = result && AnnActivateFunctionAutoTest(EPS, false, 3.0f, FUNC_A(Simd::Base::AnnRoughTanh), FUNC_A(SimdAnnRoughTanh));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, false, 3.0f, FUNC_A(Simd::Sse::AnnRoughTanh), FUNC_A(SimdAnnRoughTanh));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, false, 3.0f, FUNC_A(Simd::Avx::AnnRoughTanh), FUNC_A(SimdAnnRoughTanh));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, false, 3.0f, FUNC_A(Simd::Neon::AnnRoughTanh), FUNC_A(SimdAnnRoughTanh));
#endif

        return result;
    }

    bool AnnDerivativeTanhAutoTest()
    {
        bool result = true;

        result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Base::AnnDerivativeTanh), FUNC_A(SimdAnnDerivativeTanh));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Sse::AnnDerivativeTanh), FUNC_A(SimdAnnDerivativeTanh));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, true, 3.0f, FUNC_A(Simd::Avx::AnnDerivativeTanh), FUNC_A(SimdAnnDerivativeTanh));
#endif

        return result;
    }

    bool AnnReluAutoTest()
    {
        bool result = true;

        result = result && AnnActivateFunctionAutoTest(EPS, false, 0.5f, FUNC_A(Simd::Base::AnnRelu), FUNC_A(SimdAnnRelu));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, false, 0.5f, FUNC_A(Simd::Sse::AnnRelu), FUNC_A(SimdAnnRelu));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, false, 0.5f, FUNC_A(Simd::Avx::AnnRelu), FUNC_A(SimdAnnRelu));
#endif

        return result;
    }

    bool AnnDerivativeReluAutoTest()
    {
        bool result = true;

        result = result && AnnActivateFunctionAutoTest(EPS, true, 0.5f, FUNC_A(Simd::Base::AnnDerivativeRelu), FUNC_A(SimdAnnDerivativeRelu));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, true, 0.5f, FUNC_A(Simd::Sse::AnnDerivativeRelu), FUNC_A(SimdAnnDerivativeRelu));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnActivateFunctionAutoTest(EPS, true, 0.5f, FUNC_A(Simd::Avx::AnnDerivativeRelu), FUNC_A(SimdAnnDerivativeRelu));
#endif

        return result;
    }

    namespace
    {
        struct FuncUW
        {
            typedef void(*FuncPtr)(const float * x, size_t size, const float * a, const float * b, float * d, float * w);

            FuncPtr func;
            String description;

            FuncUW(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & x, float a, float b, const View & d, const View & w, View & dDst, View & wDst) const
            {
                Simd::Copy(d, dDst);
                Simd::Copy(w, wDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)x.data, x.width, &a, &b, (float*)dDst.data, (float*)wDst.data);
            }
        };
    }
#define FUNC_UW(function) FuncUW(function, #function)

    bool AnnUpdateWeightsAutoTest(int size, float error, bool relative, const FuncUW & f1, const FuncUW & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View x(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View d(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View w(size, 1, View::Float, NULL, TEST_ALIGN(size));
        FillRandom32f(x, -10.0f, 10.0f);
        FillRandom32f(d, -10.0f, 10.0f);
        FillRandom32f(w, -10.0f, 10.0f);

        float a = 2.0, b = 3.0;

        View dDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(x, a, b, d, w, dDst1, wDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(x, a, b, d, w, dDst2, wDst2));

        result = result && Compare(dDst1, dDst2, error, true, 32, relative, "d");
        result = result && Compare(wDst1, wDst2, error, true, 32, relative, "w");

        return result;
    }

    bool AnnUpdateWeightsAutoTest(float error, bool relative, const FuncUW & f1, const FuncUW & f2)
    {
        bool result = true;

        result = result && AnnUpdateWeightsAutoTest(W*H, error, relative, f1, f2);
        result = result && AnnUpdateWeightsAutoTest(W*H + O, error, relative, f1, f2);
        result = result && AnnUpdateWeightsAutoTest(W*H - O, error, relative, f1, f2);

        return result;
    }

    bool AnnUpdateWeightsAutoTest()
    {
        bool result = true;

        result = result && AnnUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Base::AnnUpdateWeights), FUNC_UW(SimdAnnUpdateWeights));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Sse::AnnUpdateWeights), FUNC_UW(SimdAnnUpdateWeights));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnUpdateWeightsAutoTest(EPS, false, FUNC_UW(Simd::Avx::AnnUpdateWeights), FUNC_UW(SimdAnnUpdateWeights));
#endif

        return result;
    }

    namespace
    {
        struct FuncC2
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, const float * weights, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncC2(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, const Size& size, const float * weights, const View & dstSrc, View & dstDst) const
            {
                Simd::Copy(dstSrc, dstDst);
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride/sizeof(float), size.x, size.y, weights, (float*)dstDst.data, dstDst.stride/sizeof(float));
            }
        };
    }
#define FUNC_C2(function) FuncC2(function, #function)

    bool AnnAddConvolutionAutoTest(int width, int height, float eps, int half, bool forward, const FuncC2 & f1, const FuncC2 & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        Size size(width, height), border(half, half), s(size), d(size);
        if (forward)
            s += 2*border;
        else
            d += 2*border;

        View src(s.x, s.y, View::Float, NULL, TEST_ALIGN(width));
        FillRandom32f(src, 0, 1);

        View weights(Simd::Square(1 + 2 * half), 1, View::Float, NULL, TEST_ALIGN(width));
        FillRandom32f(weights, -1, 1);

        View dstSrc(d.x, d.y, View::Float, NULL, TEST_ALIGN(width));
        FillRandom32f(dstSrc, -1000, 1000);

        View dstDst1(d.x, d.y, View::Float, NULL, TEST_ALIGN(width));
        View dstDst2(d.x, d.y, View::Float, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, size, (float*)weights.data, dstSrc, dstDst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, size, (float*)weights.data, dstSrc, dstDst2));

        result = Compare(dstDst1, dstDst2, eps, true, 32);

        return result;
    }

    bool AnnAddConvolutionAutoTest(float eps, int half, bool forward, const FuncC2 & f1, const FuncC2 & f2)
    {
        bool result = true;

        result = result && AnnAddConvolutionAutoTest(W, H, eps, half, forward, f1, f2);
        result = result && AnnAddConvolutionAutoTest(W - O, H + O, eps, half, forward, f1, f2);
        result = result && AnnAddConvolutionAutoTest(W + O, H - O, eps, half, forward, f1, f2);

        return result;
    }

    bool AnnAddConvolution3x3AutoTest()
    {
        bool result = true;

        result = result && AnnAddConvolutionAutoTest(EPS, 1, true, FUNC_C2(Simd::Base::AnnAddConvolution3x3), FUNC_C2(SimdAnnAddConvolution3x3));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnAddConvolutionAutoTest(EPS, 1, true, FUNC_C2(Simd::Sse::AnnAddConvolution3x3), FUNC_C2(SimdAnnAddConvolution3x3));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnAddConvolutionAutoTest(EPS, 1, true, FUNC_C2(Simd::Avx::AnnAddConvolution3x3), FUNC_C2(SimdAnnAddConvolution3x3));
#endif

        return result;
    }

    bool AnnAddConvolution5x5AutoTest()
    {
        bool result = true;

        result = result && AnnAddConvolutionAutoTest(EPS, 2, true, FUNC_C2(Simd::Base::AnnAddConvolution5x5), FUNC_C2(SimdAnnAddConvolution5x5));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnAddConvolutionAutoTest(EPS, 2, true, FUNC_C2(Simd::Sse::AnnAddConvolution5x5), FUNC_C2(SimdAnnAddConvolution5x5));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnAddConvolutionAutoTest(EPS, 2, true, FUNC_C2(Simd::Avx::AnnAddConvolution5x5), FUNC_C2(SimdAnnAddConvolution5x5));
#endif

        return result;
    }

    bool AnnAddConvolution3x3BackAutoTest()
    {
        bool result = true;

        result = result && AnnAddConvolutionAutoTest(EPS, 1, false, FUNC_C2(Simd::Base::AnnAddConvolution3x3Back), FUNC_C2(SimdAnnAddConvolution3x3Back));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnAddConvolutionAutoTest(EPS, 1, false, FUNC_C2(Simd::Sse::AnnAddConvolution3x3Back), FUNC_C2(SimdAnnAddConvolution3x3Back));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnAddConvolutionAutoTest(EPS, 1, false, FUNC_C2(Simd::Avx::AnnAddConvolution3x3Back), FUNC_C2(SimdAnnAddConvolution3x3Back));
#endif

        return result;
    }

    bool AnnAddConvolution5x5BackAutoTest()
    {
        bool result = true;

        result = result && AnnAddConvolutionAutoTest(EPS, 2, false, FUNC_C2(Simd::Base::AnnAddConvolution5x5Back), FUNC_C2(SimdAnnAddConvolution5x5Back));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnAddConvolutionAutoTest(EPS, 2, false, FUNC_C2(Simd::Sse::AnnAddConvolution5x5Back), FUNC_C2(SimdAnnAddConvolution5x5Back));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnAddConvolutionAutoTest(EPS, 2, false, FUNC_C2(Simd::Avx::AnnAddConvolution5x5Back), FUNC_C2(SimdAnnAddConvolution5x5Back));
#endif

        return result;
    }

    namespace
    {
        struct FuncM
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, size_t width, size_t height, float * dst, size_t dstStride);

            FuncPtr func;
            String description;

            FuncM(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, src.stride/sizeof(float), src.width, src.height, (float*)dst.data, dst.stride/sizeof(float));
            }
        };
    }
#define FUNC_M(function) FuncM(function, #function)

    bool AnnMax2x2AutoTest(int width, int height, float eps, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Float, NULL, TEST_ALIGN(width));
        FillRandom32f(src, -1, 1);

        View dst1(width / 2, height / 2, View::Float, NULL, TEST_ALIGN(width));
        View dst2(width / 2, height / 2, View::Float, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2));

        result = Compare(dst1, dst2, eps, true, 32);

        return result;
    }

    bool AnnMax2x2AutoTest(float eps, const FuncM & f1, const FuncM & f2)
    {
        bool result = true;

        result = result && AnnMax2x2AutoTest(W, H, eps, f1, f2);
        result = result && AnnMax2x2AutoTest(W - E, H + E, eps, f1, f2);
        result = result && AnnMax2x2AutoTest(W + E, H - E, eps, f1, f2);

        return result;
    }

    bool AnnMax2x2AutoTest()
    {
        bool result = true;

        result = result && AnnMax2x2AutoTest(EPS, FUNC_M(Simd::Base::AnnMax2x2), FUNC_M(SimdAnnMax2x2));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && AnnMax2x2AutoTest(EPS, FUNC_M(Simd::Sse::AnnMax2x2), FUNC_M(SimdAnnMax2x2));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && AnnMax2x2AutoTest(EPS, FUNC_M(Simd::Avx::AnnMax2x2), FUNC_M(SimdAnnMax2x2));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

	bool AnnConvertDataTest(bool create, int width, int height, float eps, const FuncC1 & f)
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

		result = result && AnnConvertDataTest(create, DW, DH, EPS, FUNC_C1(SimdAnnConvert, true));

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

    bool AnnAddVectorMultipliedByValueDataTest(bool create, int size, float eps, const FuncAVMV & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        const float value = 0.3f;

        View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstSrc(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dstDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(src);
            FillRandom32f(dstSrc);

            TEST_SAVE(src);
            TEST_SAVE(dstSrc);

            f.Call(src, value, dstSrc, dstDst1);

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            f.Call(src, value, dstSrc, dstDst2);

            TEST_SAVE(dstDst2);

            result = result && Compare(dstDst1, dstDst2, eps, true);
        }

        return result;
    }

    bool AnnAddVectorMultipliedByValueDataTest(bool create)
    {
        bool result = true;

        result = result && AnnAddVectorMultipliedByValueDataTest(create, DH, EPS, FUNC_AVMV(SimdAnnAddVectorMultipliedByValue));

        return result;
    }

	bool AnnActivateFunctionDataTest(bool create, int size, float error, bool relative, float slope, const FuncA & f)
	{
		bool result = true;

		Data data(f.description);

		TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

		View src(size, 1, View::Float, NULL, TEST_ALIGN(size));
		View dst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
		View dst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

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

		result = result && AnnActivateFunctionDataTest(create, DH, EPS, true, 3.0f, FUNC_A(SimdAnnSigmoid));

		return result;
	}

	bool AnnRoughSigmoidDataTest(bool create)
	{
		bool result = true;

		result = result && AnnActivateFunctionDataTest(create, DH, EPS, true, 3.0f, FUNC_A(SimdAnnRoughSigmoid));

		return result;
	}

    bool AnnDerivativeSigmoidDataTest(bool create)
    {
        bool result = true;

        result = result && AnnActivateFunctionDataTest(create, DH, EPS, true, 3.0f, FUNC_A(SimdAnnDerivativeSigmoid));

        return result;
    }

    bool AnnTanhDataTest(bool create)
    {
        bool result = true;

        result = result && AnnActivateFunctionDataTest(create, DH, EPS, false, 3.0f, FUNC_A(SimdAnnTanh));

        return result;
    }

    bool AnnRoughTanhDataTest(bool create)
    {
        bool result = true;

        result = result && AnnActivateFunctionDataTest(create, DH, EPS, false, 3.0f, FUNC_A(SimdAnnRoughTanh));

        return result;
    }

    bool AnnDerivativeTanhDataTest(bool create)
    {
        bool result = true;

        result = result && AnnActivateFunctionDataTest(create, DH, EPS, true, 3.0f, FUNC_A(SimdAnnDerivativeTanh));

        return result;
    }

    bool AnnReluDataTest(bool create)
    {
        bool result = true;

        result = result && AnnActivateFunctionDataTest(create, DH, EPS, false, 0.5f, FUNC_A(SimdAnnRelu));

        return result;
    }

    bool AnnDerivativeReluDataTest(bool create)
    {
        bool result = true;

        result = result && AnnActivateFunctionDataTest(create, DH, EPS, true, 0.5f, FUNC_A(SimdAnnDerivativeRelu));

        return result;
    }

    bool AnnUpdateWeightsDataTest(bool create, int size, float error, bool relative, const FuncUW & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View x(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View d(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View w(size, 1, View::Float, NULL, TEST_ALIGN(size));

        float a = 3.0, b = 5.0;

        View dDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst1(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View dDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));
        View wDst2(size, 1, View::Float, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom32f(x, -10.0f, 10.0f);
            FillRandom32f(d, -10.0f, 10.0f);
            FillRandom32f(w, -10.0f, 10.0f);

            TEST_SAVE(x);
            TEST_SAVE(d);
            TEST_SAVE(w);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(x, a, b, d, w, dDst1, wDst1));

            TEST_SAVE(dDst1);
            TEST_SAVE(wDst1);
        }
        else
        {
            TEST_LOAD(x);
            TEST_LOAD(d);
            TEST_LOAD(w);

            TEST_LOAD(dDst1);
            TEST_LOAD(wDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(x, a, b, d, w, dDst2, wDst2));

            TEST_SAVE(dDst2);
            TEST_SAVE(wDst2);

            result = Compare(dDst1, dDst2, error, true, 32, relative);
            result = Compare(wDst1, wDst2, error, true, 32, relative);
        }

        return result;
    }

    bool AnnUpdateWeightsDataTest(bool create)
    {
        bool result = true;

        result = result && AnnUpdateWeightsDataTest(create, DH, EPS, true, FUNC_UW(SimdAnnUpdateWeights));

        return result;
    }

    bool AnnAddConvolutionDataTest(bool create, int width, int height, float eps, int half, bool forward, const FuncC2 & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        Size size(width, height), border(half, half), s(size), d(size);
        if (forward)
            s += 2 * border;
        else
            d += 2 * border;

        View src(s.x, s.y, View::Float, NULL, TEST_ALIGN(width));
        View weights(Simd::Square(1 + 2*half), 1, View::Float, NULL, TEST_ALIGN(width));
        View dstSrc(d.x, d.y, View::Float, NULL, TEST_ALIGN(width));
        View dstDst1(d.x, d.y, View::Float, NULL, TEST_ALIGN(width));
        View dstDst2(d.x, d.y, View::Float, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom32f(src, 0, 1);
            FillRandom32f(weights, -1, 1);
            FillRandom32f(dstSrc, -1000, 1000);

            TEST_SAVE(src);
            TEST_SAVE(weights);
            TEST_SAVE(dstSrc);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, size, (float*)weights.data, dstSrc, dstDst1));

            TEST_SAVE(dstDst1);
        }
        else
        {
            TEST_LOAD(src);
            TEST_LOAD(weights);
            TEST_LOAD(dstSrc);

            TEST_LOAD(dstDst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, size, (float*)weights.data, dstSrc, dstDst2));

            TEST_SAVE(dstDst2);

            result = Compare(dstDst1, dstDst2, eps, true, 32, false);
        }

        return result;
    }

    bool AnnAddConvolution3x3DataTest(bool create)
    {
        bool result = true;

        result = result && AnnAddConvolutionDataTest(create, DW, DH, EPS, 1, true, FUNC_C2(SimdAnnAddConvolution3x3));

        return result;
    }

    bool AnnAddConvolution5x5DataTest(bool create)
    {
        bool result = true;

        result = result && AnnAddConvolutionDataTest(create, DW, DH, EPS, 2, true, FUNC_C2(SimdAnnAddConvolution5x5));

        return result;
    }

    bool AnnAddConvolution3x3BackDataTest(bool create)
    {
        bool result = true;

        result = result && AnnAddConvolutionDataTest(create, DW, DH, EPS, 1, false, FUNC_C2(SimdAnnAddConvolution3x3Back));

        return result;
    }

    bool AnnAddConvolution5x5BackDataTest(bool create)
    {
        bool result = true;

        result = result && AnnAddConvolutionDataTest(create, DW, DH, EPS, 2, false, FUNC_C2(SimdAnnAddConvolution5x5Back));

        return result;
    }

    bool AnnMax2x2DataTest(bool create, int width, int height, float eps, const FuncM & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        View src(width, height, View::Float, NULL, TEST_ALIGN(width));
        View dst1(width / 2, height / 2, View::Float, NULL, TEST_ALIGN(width));
        View dst2(width / 2, height / 2, View::Float, NULL, TEST_ALIGN(width));

        if (create)
        {
            FillRandom32f(src, -1, 1);

            TEST_SAVE(src);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, dst1));

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            TEST_EXECUTE_AT_LEAST_MIN_TIME(f.Call(src, dst2));

            TEST_SAVE(dst2);

            result = Compare(dst1, dst2, eps, true, 32);
        }

        return result;
    }

    bool AnnMax2x2DataTest(bool create)
    {
        bool result = true;

        result = result && AnnMax2x2DataTest(create, DW, DH, EPS, FUNC_M(SimdAnnMax2x2));

        return result;
    }
}

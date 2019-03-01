/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Test/TestTensor.h"
#include "Test/TestData.h"

namespace Test
{
    namespace
    {
        struct FuncWF
        {
            typedef void(*FuncPtr)(const float * src, size_t size, float * dst, SimdBool trans);

            FuncPtr func;
            String description;

            FuncWF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(SimdBool trans)
            {
                description = description + (trans ? "[1]" : "[0]");
            }

            void Call(const Tensor32f & src, size_t size, Tensor32f & dst, SimdBool trans) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.Data(), size, dst.Data(), trans);
            }
        };
    }

#define FUNC_WF(function) FuncWF(function, #function)

    bool WinogradSetFilterAutoTest(size_t srcC, size_t dstC, size_t block, size_t core, SimdBool trans, FuncWF f1, FuncWF f2)
    {
        bool result = true;

        f1.Update(trans);
        f2.Update(trans);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcC << ", " << dstC << "].");

        size_t count = Simd::Square(block + core - 1);
        Tensor32f src({ trans ? core : dstC, trans ? core : srcC, trans ? srcC : core, trans ? dstC : core });
        FillRandom(src.Data(), src.Size(), -10.0, 10.0f);
        Tensor32f dst1({count,  trans ? srcC : dstC, trans ? dstC : srcC});
        Tensor32f dst2({ count,  trans ? srcC : dstC, trans ? dstC : srcC });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcC*dstC, dst1, trans));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcC*dstC, dst2, trans));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool WinogradSetFilterAutoTest(size_t block, size_t core, const FuncWF & f1, const FuncWF & f2)
    {
        bool result = true;

        result = result && WinogradSetFilterAutoTest(W / 3, W / 4, block, core, ::SimdFalse, f1, f2);
        result = result && WinogradSetFilterAutoTest(W / 3, W / 4, block, core, ::SimdTrue, f1, f2);

        return result;
    }

    bool Winograd2x3SetFilterAutoTest()
    {
        bool result = true;

        result = result && WinogradSetFilterAutoTest(2, 3, FUNC_WF(Simd::Base::Winograd2x3SetFilter), FUNC_WF(SimdWinograd2x3SetFilter));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetFilterAutoTest(2, 3, FUNC_WF(Simd::Sse::Winograd2x3SetFilter), FUNC_WF(SimdWinograd2x3SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(2, 3, FUNC_WF(Simd::Avx::Winograd2x3SetFilter), FUNC_WF(SimdWinograd2x3SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(2, 3, FUNC_WF(Simd::Avx512f::Winograd2x3SetFilter), FUNC_WF(SimdWinograd2x3SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(2, 3, FUNC_WF(Simd::Neon::Winograd2x3SetFilter), FUNC_WF(SimdWinograd2x3SetFilter));
#endif 

        return result;
    }

    bool Winograd4x3SetFilterAutoTest()
    {
        bool result = true;

        result = result && WinogradSetFilterAutoTest(4, 3, FUNC_WF(Simd::Base::Winograd4x3SetFilter), FUNC_WF(SimdWinograd4x3SetFilter));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetFilterAutoTest(4, 3, FUNC_WF(Simd::Sse::Winograd4x3SetFilter), FUNC_WF(SimdWinograd4x3SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(4, 3, FUNC_WF(Simd::Avx::Winograd4x3SetFilter), FUNC_WF(SimdWinograd4x3SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(4, 3, FUNC_WF(Simd::Avx512f::Winograd4x3SetFilter), FUNC_WF(SimdWinograd4x3SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(4, 3, FUNC_WF(Simd::Neon::Winograd4x3SetFilter), FUNC_WF(SimdWinograd4x3SetFilter));
#endif 

        return result;
    }

    namespace
    {
        struct FuncWI
        {
            typedef void(*FuncPtr)(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, SimdBool pad, SimdBool trans);

            FuncPtr func;
            String description;

            FuncWI(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(size_t c, size_t h, size_t w, int p, int t)
            {
                description = description + "[" + ToString(c) + "-" + ToString(h) + "-" + ToString(w) + "-" + ToString(p) + "-" + ToString(t) + "]";
            }

            void Call(const Tensor32f & src, size_t srcC, size_t srcH, size_t srcW, Tensor32f & dst, int pad, int trans) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.Data(), srcC, srcH, srcW, dst.Data(), (SimdBool)pad, (SimdBool)trans);
            }
        };
    }

#define FUNC_WI(function) FuncWI(function, #function)

    bool WinogradSetInputAutoTest(size_t srcC, size_t srcH, size_t srcW, size_t block, size_t core, int pad, int trans, FuncWI f1, FuncWI f2)
    {
        bool result = true;

        f1.Update(srcC, srcH, srcW, pad, trans);
        f2.Update(srcC, srcH, srcW, pad, trans);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " .");

        size_t dstW = pad ? srcW : srcW - core + 1;
        size_t dstH = pad ? srcH : srcH - core + 1;
        size_t count = Simd::Square(block + core - 1);
        size_t tileH = (dstH + block - 1) / block;
        size_t tileW = (dstW + block - 1) / block;

        Tensor32f src({ trans ? srcH : srcC, trans ? srcW : srcH, trans ? srcC : srcW});
        FillRandom(src.Data(), src.Size(), -10.0, 10.0f);
        Tensor32f dst1({ count, trans ? tileH : srcC, trans ? tileW : tileH, trans ? srcC : tileW });
        Tensor32f dst2({ count, trans ? tileH : srcC, trans ? tileW : tileH, trans ? srcC : tileW });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcC, srcH, srcW, dst1, pad, trans));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcC, srcH, srcW, dst2, pad, trans));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool WinogradSetInputAutoTest(size_t block, size_t core, int pad, int trans, const FuncWI & f1, const FuncWI & f2)
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(64, 56, 48, block, core, pad, trans, f1, f2);
        result = result && WinogradSetInputAutoTest(128, 28, 24, block, core, pad, trans, f1, f2);
        result = result && WinogradSetInputAutoTest(256, 14, 12, block, core, pad, trans, f1, f2);
        result = result && WinogradSetInputAutoTest(512, 7, 6, block, core, pad, trans, f1, f2);

        return result;
    }

    bool WinogradSetInputAutoTest(size_t block, size_t core, const FuncWI & f1, const FuncWI & f2)
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(block, core, 1, 0, f1, f2);
        result = result && WinogradSetInputAutoTest(block, core, 1, 1, f1, f2);

        return result;
    }

    bool Winograd2x3SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Base::Winograd2x3SetInput), FUNC_WI(SimdWinograd2x3SetInput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Sse::Winograd2x3SetInput), FUNC_WI(SimdWinograd2x3SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Avx::Winograd2x3SetInput), FUNC_WI(SimdWinograd2x3SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Avx512f::Winograd2x3SetInput), FUNC_WI(SimdWinograd2x3SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Neon::Winograd2x3SetInput), FUNC_WI(SimdWinograd2x3SetInput));
#endif

        return result;
    }

    bool Winograd4x3SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(4, 3, FUNC_WI(Simd::Base::Winograd4x3SetInput), FUNC_WI(SimdWinograd4x3SetInput));

        return result;
    }

    namespace
    {
        struct FuncWO
        {
            typedef void(*FuncPtr)(const float * src, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

            FuncPtr func;
            String description;

            FuncWO(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(size_t c, size_t h, size_t w, int t)
            {
                description = description + "[" + ToString(c) + "-" + ToString(h) + "-" + ToString(w) + "-" + ToString(t) + "]";
            }

            void Call(const Tensor32f & src, Tensor32f & dst, size_t dstC, size_t dstH, size_t dstW, int trans) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.Data(), dst.Data(), dstC, dstH, dstW, (SimdBool)trans);
            }
        };
    }

#define FUNC_WO(function) FuncWO(function, #function)

    bool WinogradSetOutputAutoTest(size_t dstC, size_t dstH, size_t dstW, size_t block, size_t core, int trans, FuncWO f1, FuncWO f2)
    {
        bool result = true;

        f1.Update(dstC, dstH, dstW, trans);
        f2.Update(dstC, dstH, dstW, trans);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " .");

        size_t count = Simd::Square(block + core - 1);
        size_t tileH = (dstH + block - 1) / block;
        size_t tileW = (dstW + block - 1) / block;

        Tensor32f src({count, trans ? tileH : dstC, trans ? tileW : tileH, trans ? dstC : tileW });
        FillRandom(src.Data(), src.Size(), -10.0, 10.0f);
        Tensor32f dst1({ trans ? dstH : dstC, trans ? dstW : dstH, trans ? dstC : dstW });
        Tensor32f dst2({ trans ? dstH : dstC, trans ? dstW : dstH, trans ? dstC : dstW });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1, dstC, dstH, dstW, trans));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2, dstC, dstH, dstW, trans));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool WinogradSetOutputAutoTest(size_t block, size_t core, int trans, const FuncWO & f1, const FuncWO & f2)
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(64, 56, 48, block, core, trans, f1, f2);
        result = result && WinogradSetOutputAutoTest(128, 28, 24, block, core, trans, f1, f2);
        result = result && WinogradSetOutputAutoTest(256, 14, 12, block, core, trans, f1, f2);
        result = result && WinogradSetOutputAutoTest(512, 7, 6, block, core, trans, f1, f2);

        return result;
    }

    bool WinogradSetOutputAutoTest(size_t block, size_t core, const FuncWO & f1, const FuncWO & f2)
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(block, core, 0, f1, f2);
        result = result && WinogradSetOutputAutoTest(block, core, 1, f1, f2);

        return result;
    }

    bool Winograd2x3SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Base::Winograd2x3SetOutput), FUNC_WO(SimdWinograd2x3SetOutput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Sse::Winograd2x3SetOutput), FUNC_WO(SimdWinograd2x3SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Avx::Winograd2x3SetOutput), FUNC_WO(SimdWinograd2x3SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Avx512f::Winograd2x3SetOutput), FUNC_WO(SimdWinograd2x3SetOutput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Neon::Winograd2x3SetOutput), FUNC_WO(SimdWinograd2x3SetOutput));
#endif

        return result;
    }

    bool Winograd4x3SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(4, 3, FUNC_WO(Simd::Base::Winograd4x3SetOutput), FUNC_WO(SimdWinograd4x3SetOutput));

        return result;
    }
}

/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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

        return result;
    }

    namespace
    {
        struct FuncWI
        {
            typedef void(*FuncPtr)(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, int pad);

            FuncPtr func;
            String description;

            FuncWI(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(size_t c, size_t h, size_t w, int p)
            {
                description = description + "[" + ToString(w) + "-" + ToString(h) + "-" + ToString(c) + "-" + ToString(p) + "]";
            }

            void Call(const View & src, size_t srcChannels, size_t srcHeight, size_t srcWidth, View & dst, int pad) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, srcChannels, srcHeight, srcWidth, (float*)dst.data, pad);
            }
        };
    }

#define FUNC_WI(function) FuncWI(function, #function)

    bool WinogradSetInputAutoTest(size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t block, size_t core, FuncWI f1, FuncWI f2, int pad)
    {
        bool result = true;

        f1.Update(srcChannels, srcHeight, srcWidth, pad);
        f2.Update(srcChannels, srcHeight, srcWidth, pad);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcChannels << ", " << srcHeight << ", " << srcWidth << "].");

        size_t dstWidth = pad ? srcWidth : srcWidth - core + 1;
        size_t dstHeight = pad ? srcHeight : srcHeight - core + 1;
        size_t count = Simd::Square(block + core - 1);
        size_t tileH = (dstHeight + block - 1) / block;
        size_t tileW = (dstWidth + block - 1) / block;
        size_t strideS = srcChannels * tileH * tileW;

        View src(srcChannels*srcHeight*srcWidth, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(strideS*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(strideS*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcChannels, srcHeight, srcWidth, dst1, pad));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcChannels, srcHeight, srcWidth, dst2, pad));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool WinogradSetInputAutoTest(size_t block, size_t core, const FuncWI & f1, const FuncWI & f2)
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(3, 320, 320, block, core, f1, f2, 1);
        result = result && WinogradSetInputAutoTest(16, 160, 160, block, core, f1, f2, 1);
        result = result && WinogradSetInputAutoTest(32, 80, 80, block, core, f1, f2, 1);
        result = result && WinogradSetInputAutoTest(64, 40, 40, block, core, f1, f2, 1);
        result = result && WinogradSetInputAutoTest(128, 20, 20, block, core, f1, f2, 1);
        result = result && WinogradSetInputAutoTest(256, 10, 10, block, core, f1, f2, 1);
        result = result && WinogradSetInputAutoTest(320, 20, 20, block, core, f1, f2, 1);

        return result;
    }

    bool Winograd2x3iSetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Base::Winograd2x3iSetInput), FUNC_WI(SimdWinograd2x3iSetInput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Sse::Winograd2x3iSetInput), FUNC_WI(SimdWinograd2x3iSetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Avx::Winograd2x3iSetInput), FUNC_WI(SimdWinograd2x3iSetInput));
#endif 

        return result;
    }

    bool Winograd2x3pSetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Base::Winograd2x3pSetInput), FUNC_WI(SimdWinograd2x3pSetInput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Sse::Winograd2x3pSetInput), FUNC_WI(SimdWinograd2x3pSetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Avx::Winograd2x3pSetInput), FUNC_WI(SimdWinograd2x3pSetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetInputAutoTest(2, 3, FUNC_WI(Simd::Avx512f::Winograd2x3pSetInput), FUNC_WI(SimdWinograd2x3pSetInput));
#endif 

        return result;
    }

    bool Winograd4x3pSetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(4, 3, FUNC_WI(Simd::Base::Winograd4x3pSetInput), FUNC_WI(SimdWinograd4x3pSetInput));

        return result;
    }

    namespace
    {
        struct FuncWO
        {
            typedef void(*FuncPtr)(const float * src, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth);

            FuncPtr func;
            String description;

            FuncWO(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(size_t c, size_t h, size_t w)
            {
                description = description + "[" + ToString(w) + "-" + ToString(h) + "-" + ToString(c) + "]";
            }

            void Call(const View & src, View & dst, size_t dstChannels, size_t dstHeight, size_t dstWidth) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, (float*)dst.data, dstChannels, dstHeight, dstWidth);
            }
        };
    }

#define FUNC_WO(function) FuncWO(function, #function)

    bool WinogradSetOutputAutoTest(size_t dstChannels, size_t dstHeight, size_t dstWidth, size_t block, size_t core, FuncWO f1, FuncWO f2)
    {
        bool result = true;

        f1.Update(dstChannels, dstHeight, dstWidth);
        f2.Update(dstChannels, dstHeight, dstWidth);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << dstChannels << ", " << dstHeight << ", " << dstWidth << "].");

        size_t count = Simd::Square(block + core - 1);
        size_t tileH = (dstHeight + block - 1) / block;
        size_t tileW = (dstWidth + block - 1) / block;
        size_t strideD = dstChannels * tileH * tileW;

        View src(strideD*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(dstChannels*dstHeight*dstWidth, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(dstChannels*dstHeight*dstWidth, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, dst1, dstChannels, dstHeight, dstWidth));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, dst2, dstChannels, dstHeight, dstWidth));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool WinogradSetOutputAutoTest(size_t block, size_t core, const FuncWO & f1, const FuncWO & f2)
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(16, 320, 320, block, core, f1, f2);
        result = result && WinogradSetOutputAutoTest(32, 160, 160, block, core, f1, f2);
        result = result && WinogradSetOutputAutoTest(64, 80, 80, block, core, f1, f2);
        result = result && WinogradSetOutputAutoTest(128, 40, 40, block, core, f1, f2);
        result = result && WinogradSetOutputAutoTest(256, 20, 20, block, core, f1, f2);
        result = result && WinogradSetOutputAutoTest(256, 10, 10, block, core, f1, f2);

        return result;
    }

    bool Winograd2x3iSetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Base::Winograd2x3iSetOutput), FUNC_WO(SimdWinograd2x3iSetOutput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Sse::Winograd2x3iSetOutput), FUNC_WO(SimdWinograd2x3iSetOutput));
#endif

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Avx::Winograd2x3iSetOutput), FUNC_WO(SimdWinograd2x3iSetOutput));
#endif 

        return result;
    }

    bool Winograd2x3pSetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Base::Winograd2x3pSetOutput), FUNC_WO(SimdWinograd2x3pSetOutput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Sse::Winograd2x3pSetOutput), FUNC_WO(SimdWinograd2x3pSetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Avx::Winograd2x3pSetOutput), FUNC_WO(SimdWinograd2x3pSetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetOutputAutoTest(2, 3, FUNC_WO(Simd::Avx512f::Winograd2x3pSetOutput), FUNC_WO(SimdWinograd2x3pSetOutput));
#endif 

        return result;
    }

    bool Winograd4x3pSetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(4, 3, FUNC_WO(Simd::Base::Winograd4x3pSetOutput), FUNC_WO(SimdWinograd4x3pSetOutput));

        return result;
    }

    //-----------------------------------------------------------------------

    bool WinogradSetInputDataTest(bool create, size_t srcChannels, size_t srcHeight, size_t srcWidth, size_t block, size_t core, const FuncWI & f)
    {
        bool result = true;

        int pad = 1;
        size_t dstWidth = pad ? srcWidth : srcWidth - core + 1;
        size_t dstHeight = pad ? srcHeight : srcHeight - core + 1;
        size_t count = Simd::Square(block + core - 1);
        size_t tileH = (dstHeight + block - 1) / block;
        size_t tileW = (dstWidth + block - 1) / block;
        size_t strideS = srcChannels * tileH * tileW;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << srcChannels << ", " << srcHeight << ", " << srcWidth << "].");

        View src(srcChannels*srcHeight*srcWidth, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(strideS*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(strideS*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(src, -10.0, 10.0);

            TEST_SAVE(src);

            f.Call(src, srcChannels, srcHeight, srcWidth, dst1, pad);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, srcChannels, srcHeight, srcWidth, dst2, pad);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        return result;
    }

    bool Winograd2x3iSetInputDataTest(bool create)
    {
        return WinogradSetInputDataTest(create, DW / 7, DW / 7, DW / 7, 2, 3, FUNC_WI(SimdWinograd2x3iSetInput));
    }

    bool Winograd2x3pSetInputDataTest(bool create)
    {
        return WinogradSetInputDataTest(create, DW / 7, DW / 7, DW / 7, 2, 3, FUNC_WI(SimdWinograd2x3pSetInput));
    }

    bool Winograd4x3pSetInputDataTest(bool create)
    {
        return WinogradSetInputDataTest(create, DW / 7, DW / 7, DW / 7, 4, 3, FUNC_WI(SimdWinograd4x3pSetInput));
    }

    bool WinogradSetOutputDataTest(bool create, size_t dstChannels, size_t dstHeight, size_t dstWidth, size_t block, size_t core, const FuncWO & f)
    {
        bool result = true;

        size_t count = Simd::Square(block + core - 1);
        size_t tileH = (dstHeight + block - 1) / block;
        size_t tileW = (dstWidth + block - 1) / block;
        size_t strideD = dstChannels * tileH * tileW;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << dstChannels << ", " << dstHeight << ", " << dstWidth << "].");

        View src(strideD*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(dstChannels*dstHeight*dstWidth, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(dstChannels*dstHeight*dstWidth, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(src, -10.0, 10.0);

            TEST_SAVE(src);

            f.Call(src, dst1, dstChannels, dstHeight, dstWidth);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, dst2, dstChannels, dstHeight, dstWidth);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        return result;
    }

    bool Winograd2x3iSetOutputDataTest(bool create)
    {
        return WinogradSetOutputDataTest(create, DW / 7, DW / 7, DW / 7, 2, 3, FUNC_WO(SimdWinograd2x3iSetOutput));
    }

    bool Winograd2x3pSetOutputDataTest(bool create)
    {
        return WinogradSetOutputDataTest(create, DW / 7, DW / 7, DW / 7, 2, 3, FUNC_WO(SimdWinograd2x3pSetOutput));
    }

    bool Winograd4x3pSetOutputDataTest(bool create)
    {
        return WinogradSetOutputDataTest(create, DW / 7, DW / 7, DW / 7, 4, 3, FUNC_WO(SimdWinograd4x3pSetOutput));
    }
}

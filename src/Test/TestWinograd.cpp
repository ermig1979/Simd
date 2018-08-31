/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
        struct FuncWF
        {
            typedef void(*FuncPtr)(const float * src, size_t size, float * dst);

            FuncPtr func;
            String description;

            FuncWF(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, size_t size, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func((float*)src.data, size, (float*)dst.data);
            }
        };
    }

#define FUNC_WF(function) FuncWF(function, #function)

    bool WinogradSetFilterAutoTest(size_t srcChannel, size_t dstChannel, size_t block, size_t core, const FuncWF & f1, const FuncWF & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcChannel << ", " << dstChannel << "].");

        size_t count = Simd::Square(block + core - 1);
        View src(core*core*srcChannel*dstChannel, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(srcChannel*dstChannel*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(srcChannel*dstChannel*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        FillRandom32f(src, -10.0, 10.0);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcChannel*dstChannel, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcChannel*dstChannel, dst2));

        result = result && Compare(dst1, dst2, EPS, true, 64);

        return result;
    }

    bool WinogradSetFilterAutoTest(size_t block, size_t core, const FuncWF & f1, const FuncWF & f2)
    {
        bool result = true;

        result = result && WinogradSetFilterAutoTest(W / 3, W / 4, block, core, f1, f2);

        return result;
    }

    bool Winograd2x3pSetFilterAutoTest()
    {
        bool result = true;

        result = result && WinogradSetFilterAutoTest(2, 3, FUNC_WF(Simd::Base::Winograd2x3pSetFilter), FUNC_WF(SimdWinograd2x3pSetFilter));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetFilterAutoTest(2, 3, FUNC_WF(Simd::Sse::Winograd2x3pSetFilter), FUNC_WF(SimdWinograd2x3pSetFilter));
#endif 

        return result;
    }

    bool Winograd4x3pSetFilterAutoTest()
    {
        bool result = true;

        result = result && WinogradSetFilterAutoTest(4, 3, FUNC_WF(Simd::Base::Winograd4x3pSetFilter), FUNC_WF(SimdWinograd4x3pSetFilter));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetFilterAutoTest(4, 3, FUNC_WF(Simd::Sse::Winograd4x3pSetFilter), FUNC_WF(SimdWinograd4x3pSetFilter));
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

            void Update(int pad)
            {
                description = description + (pad ? "[1]" : "[0]");
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

        f1.Update(pad);
        f2.Update(pad);

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

        result = result && WinogradSetInputAutoTest(W / 14, W / 15, W / 13, block, core, f1, f2, 0);
        result = result && WinogradSetInputAutoTest(W / 14, W / 15, W / 13, block, core, f1, f2, 1);

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

        result = result && WinogradSetOutputAutoTest(W / 14, W / 15, W / 13, block, core, f1, f2);

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

        return result;
    }

    bool Winograd4x3pSetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(4, 3, FUNC_WO(Simd::Base::Winograd4x3pSetOutput), FUNC_WO(SimdWinograd4x3pSetOutput));

        return result;
    }

    //-----------------------------------------------------------------------

    bool WinogradSetFilterDataTest(bool create, size_t srcChannel, size_t dstChannel, size_t block, size_t core, const FuncWF & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << srcChannel << ", " << dstChannel << "].");

        size_t count = Simd::Square(block + core - 1);
        View src(core*core*srcChannel*dstChannel, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst1(srcChannel*dstChannel*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));
        View dst2(srcChannel*dstChannel*count, 1, View::Float, NULL, TEST_ALIGN(SIMD_ALIGN));

        if (create)
        {
            FillRandom32f(src, -10.0, 10.0);

            TEST_SAVE(src);

            f.Call(src, srcChannel*dstChannel, dst1);

            TEST_SAVE(dst1);
        }
        else
        {
            TEST_LOAD(src);

            TEST_LOAD(dst1);

            f.Call(src, srcChannel*dstChannel, dst2);

            TEST_SAVE(dst2);

            result = result && Compare(dst1, dst2, EPS, true, 64);
        }

        return result;
    }

    bool Winograd2x3pSetFilterDataTest(bool create)
    {
        return WinogradSetFilterDataTest(create, DW, DH, 2, 3, FUNC_WF(SimdWinograd2x3pSetFilter));
    }

    bool Winograd4x3pSetFilterDataTest(bool create)
    {
        return WinogradSetFilterDataTest(create, DW, DH, 4, 3, FUNC_WF(SimdWinograd4x3pSetFilter));
    }

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

    bool Winograd2x3pSetOutputDataTest(bool create)
    {
        return WinogradSetOutputDataTest(create, DW / 7, DW / 7, DW / 7, 2, 3, FUNC_WO(SimdWinograd2x3pSetOutput));
    }

    bool Winograd4x3pSetOutputDataTest(bool create)
    {
        return WinogradSetOutputDataTest(create, DW / 7, DW / 7, DW / 7, 4, 3, FUNC_WO(SimdWinograd4x3pSetOutput));
    }
}

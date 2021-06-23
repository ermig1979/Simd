/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Test/TestString.h"

namespace Test
{
#if defined(SIMD_SYNET_ENABLE)
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

    bool WinogradSetFilterAutoTest(size_t srcC, size_t dstC, const Size& block, const Size& core, SimdBool trans, FuncWF f1, FuncWF f2)
    {
        bool result = true;

        f1.Update(trans);
        f2.Update(trans);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << srcC << ", " << dstC << "].");

        size_t count = (block.x + core.x - 1) * (block.y + core.y - 1);
        Tensor32f src({ trans ? core.y : dstC, trans ? core.x : srcC, trans ? srcC : core.y, trans ? dstC : core.x });
        FillRandom(src.Data(), src.Size(), -10.0, 10.0f);
        Tensor32f dst1({ count, trans ? srcC : dstC, trans ? dstC : srcC });
        Tensor32f dst2({ count, trans ? srcC : dstC, trans ? dstC : srcC });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcC*dstC, dst1, trans));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcC*dstC, dst2, trans));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool WinogradSetFilterAutoTest(const Size& block, const Size& core, const FuncWF & f1, const FuncWF & f2)
    {
        bool result = true;

        result = result && WinogradSetFilterAutoTest(W / 3, W / 4, block, core, ::SimdFalse, f1, f2);
        result = result && WinogradSetFilterAutoTest(W / 3, W / 4, block, core, ::SimdTrue, f1, f2);

        return result;
    }

    bool WinogradKernel1x3Block1x4SetFilterAutoTest()
    {
        bool result = true;

        Size _1x3(3, 1), _1x4(4, 1);

        result = result && WinogradSetFilterAutoTest(_1x4, _1x3, FUNC_WF(Simd::Base::WinogradKernel1x3Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x3Block1x4SetFilter));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradSetFilterAutoTest(_1x4, _1x3, FUNC_WF(Simd::Sse2::WinogradKernel1x3Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x3Block1x4SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(_1x4, _1x3, FUNC_WF(Simd::Avx::WinogradKernel1x3Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x3Block1x4SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(_1x4, _1x3, FUNC_WF(Simd::Avx512f::WinogradKernel1x3Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x3Block1x4SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(_1x4, _1x3, FUNC_WF(Simd::Neon::WinogradKernel1x3Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x3Block1x4SetFilter));
#endif

        return result;
    }

    bool WinogradKernel1x5Block1x4SetFilterAutoTest()
    {
        bool result = true;

        Size _1x5(5, 1), _1x4(4, 1);

        result = result && WinogradSetFilterAutoTest(_1x4, _1x5, FUNC_WF(Simd::Base::WinogradKernel1x5Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x5Block1x4SetFilter));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradSetFilterAutoTest(_1x4, _1x5, FUNC_WF(Simd::Sse2::WinogradKernel1x5Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x5Block1x4SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(_1x4, _1x5, FUNC_WF(Simd::Avx::WinogradKernel1x5Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x5Block1x4SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(_1x4, _1x5, FUNC_WF(Simd::Avx512f::WinogradKernel1x5Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x5Block1x4SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(_1x4, _1x5, FUNC_WF(Simd::Neon::WinogradKernel1x5Block1x4SetFilter), FUNC_WF(SimdWinogradKernel1x5Block1x4SetFilter));
#endif

        return result;
    }

    bool WinogradKernel2x2Block2x2SetFilterAutoTest()
    {
        bool result = true;

        Size _2x2(2, 2);

        result = result && WinogradSetFilterAutoTest(_2x2, _2x2, FUNC_WF(Simd::Base::WinogradKernel2x2Block2x2SetFilter), FUNC_WF(SimdWinogradKernel2x2Block2x2SetFilter));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradSetFilterAutoTest(_2x2, _2x2, FUNC_WF(Simd::Sse2::WinogradKernel2x2Block2x2SetFilter), FUNC_WF(SimdWinogradKernel2x2Block2x2SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(_2x2, _2x2, FUNC_WF(Simd::Avx::WinogradKernel2x2Block2x2SetFilter), FUNC_WF(SimdWinogradKernel2x2Block2x2SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(_2x2, _2x2, FUNC_WF(Simd::Avx512f::WinogradKernel2x2Block2x2SetFilter), FUNC_WF(SimdWinogradKernel2x2Block2x2SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(_2x2, _2x2, FUNC_WF(Simd::Neon::WinogradKernel2x2Block2x2SetFilter), FUNC_WF(SimdWinogradKernel2x2Block2x2SetFilter));
#endif

        return result;
    }

    bool WinogradKernel2x2Block4x4SetFilterAutoTest()
    {
        bool result = true;

        Size _2x2(2, 2), _4x4(4, 4);

        result = result && WinogradSetFilterAutoTest(_4x4, _2x2, FUNC_WF(Simd::Base::WinogradKernel2x2Block4x4SetFilter), FUNC_WF(SimdWinogradKernel2x2Block4x4SetFilter));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradSetFilterAutoTest(_4x4, _2x2, FUNC_WF(Simd::Sse2::WinogradKernel2x2Block4x4SetFilter), FUNC_WF(SimdWinogradKernel2x2Block4x4SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(_4x4, _2x2, FUNC_WF(Simd::Avx::WinogradKernel2x2Block4x4SetFilter), FUNC_WF(SimdWinogradKernel2x2Block4x4SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(_4x4, _2x2, FUNC_WF(Simd::Avx512f::WinogradKernel2x2Block4x4SetFilter), FUNC_WF(SimdWinogradKernel2x2Block4x4SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(_4x4, _2x2, FUNC_WF(Simd::Neon::WinogradKernel2x2Block4x4SetFilter), FUNC_WF(SimdWinogradKernel2x2Block4x4SetFilter));
#endif

        return result;
    }

    bool WinogradKernel3x3Block2x2SetFilterAutoTest()
    {
        bool result = true;

        Size _2x2(2, 2), _3x3(3, 3);

        result = result && WinogradSetFilterAutoTest(_2x2, _3x3, FUNC_WF(Simd::Base::WinogradKernel3x3Block2x2SetFilter), FUNC_WF(SimdWinogradKernel3x3Block2x2SetFilter));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradSetFilterAutoTest(_2x2, _3x3, FUNC_WF(Simd::Sse2::WinogradKernel3x3Block2x2SetFilter), FUNC_WF(SimdWinogradKernel3x3Block2x2SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(_2x2, _3x3, FUNC_WF(Simd::Avx::WinogradKernel3x3Block2x2SetFilter), FUNC_WF(SimdWinogradKernel3x3Block2x2SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(_2x2, _3x3, FUNC_WF(Simd::Avx512f::WinogradKernel3x3Block2x2SetFilter), FUNC_WF(SimdWinogradKernel3x3Block2x2SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(_2x2, _3x3, FUNC_WF(Simd::Neon::WinogradKernel3x3Block2x2SetFilter), FUNC_WF(SimdWinogradKernel3x3Block2x2SetFilter));
#endif 

        return result;
    }

    bool WinogradKernel3x3Block3x3SetFilterAutoTest()
    {
        bool result = true;

        Size _3x3(3, 3);

        result = result && WinogradSetFilterAutoTest(_3x3, _3x3, FUNC_WF(Simd::Base::WinogradKernel3x3Block3x3SetFilter), FUNC_WF(SimdWinogradKernel3x3Block3x3SetFilter));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradSetFilterAutoTest(_3x3, _3x3, FUNC_WF(Simd::Sse2::WinogradKernel3x3Block3x3SetFilter), FUNC_WF(SimdWinogradKernel3x3Block3x3SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(_3x3, _3x3, FUNC_WF(Simd::Avx::WinogradKernel3x3Block3x3SetFilter), FUNC_WF(SimdWinogradKernel3x3Block3x3SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(_3x3, _3x3, FUNC_WF(Simd::Avx512f::WinogradKernel3x3Block3x3SetFilter), FUNC_WF(SimdWinogradKernel3x3Block3x3SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(_3x3, _3x3, FUNC_WF(Simd::Neon::WinogradKernel3x3Block3x3SetFilter), FUNC_WF(SimdWinogradKernel3x3Block3x3SetFilter));
#endif 

        return result;
    }

    bool WinogradKernel3x3Block4x4SetFilterAutoTest()
    {
        bool result = true;

        Size _3x3(3, 3), _4x4(4, 4);

        result = result && WinogradSetFilterAutoTest(_4x4, _3x3, FUNC_WF(Simd::Base::WinogradKernel3x3Block4x4SetFilter), FUNC_WF(SimdWinogradKernel3x3Block4x4SetFilter));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradSetFilterAutoTest(_4x4, _3x3, FUNC_WF(Simd::Sse2::WinogradKernel3x3Block4x4SetFilter), FUNC_WF(SimdWinogradKernel3x3Block4x4SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(_4x4, _3x3, FUNC_WF(Simd::Avx::WinogradKernel3x3Block4x4SetFilter), FUNC_WF(SimdWinogradKernel3x3Block4x4SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(_4x4, _3x3, FUNC_WF(Simd::Avx512f::WinogradKernel3x3Block4x4SetFilter), FUNC_WF(SimdWinogradKernel3x3Block4x4SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(_4x4, _3x3, FUNC_WF(Simd::Neon::WinogradKernel3x3Block4x4SetFilter), FUNC_WF(SimdWinogradKernel3x3Block4x4SetFilter));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    namespace
    {
        struct FuncWI
        {
            typedef void(*FuncPtr)(const float* src, size_t srcChannels, size_t srcHeight, size_t srcWidth,
                size_t padY, size_t padX, size_t padH, size_t padW, float* dst, size_t dstStride, SimdBool trans);

            FuncPtr func;
            String description;

            FuncWI(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Update(size_t c, size_t h, size_t w, int p, int t)
            {
                description = description + "[" + ToString(c) + "-" + ToString(h) + "-" + ToString(w) + "-" + ToString(p) + "-" + ToString(t) + "]";
            }

            void Call(const Tensor32f & src, size_t srcC, size_t srcH, size_t srcW, const Size& padB, const Size& padE, Tensor32f & dst, int trans) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.Data(), srcC, srcH, srcW, padB.y, padB.x, padE.y, padE.x, dst.Data(), dst.Size(1), (SimdBool)trans);
            }
        };
    }

#define FUNC_WI(function) FuncWI(function, #function)

    bool WinogradSetInputAutoTest(size_t srcC, size_t srcH, size_t srcW, const Size & block, const Size & core, const Size & padB, const Size & padE, int trans, FuncWI f1, FuncWI f2)
    {
        bool result = true;

        int pad = padB.x || padB.y || padE.x || padE.y ? 1 : 0;

        f1.Update(srcC, srcH, srcW, pad, trans);
        f2.Update(srcC, srcH, srcW, pad, trans);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " .");

        size_t dstW = srcW - core.x + 1 + padB.x + padE.x;
        size_t dstH = srcH - core.y + 1 + padB.y + padE.y;
        size_t count = (block.x + core.x - 1)* (block.y + core.y - 1);
        size_t tileH = (dstH + block.y - 1) / block.y;
        size_t tileW = (dstW + block.x - 1) / block.x;

        Tensor32f src({ trans ? srcH : srcC, trans ? srcW : srcH, trans ? srcC : srcW});
        FillRandom(src.Data(), src.Size(), -10.0, 10.0f);
        Tensor32f dst1({ count, trans ? tileH : srcC, trans ? tileW : tileH, trans ? srcC : tileW });
        Tensor32f dst2({ count, trans ? tileH : srcC, trans ? tileW : tileH, trans ? srcC : tileW });

        TEST_ALIGN(SIMD_ALIGN);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(src, srcC, srcH, srcW, padB, padE, dst1, trans));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(src, srcC, srcH, srcW, padB, padE, dst2, trans));

        result = result && Compare(dst1, dst2, EPS, true, 64, DifferenceAbsolute);

        return result;
    }

    bool WinogradKernel1x3SetInputAutoTest(size_t block, int padB, int padE, int trans, const FuncWI& f1, const FuncWI& f2)
    {
        bool result = true;

        Size _core(3, 1), _block(block, 1), _padB(padB, 0), _padE(padE, 0);

        result = result && WinogradSetInputAutoTest(128, 7, 59, _block, _core, _padB, _padE, trans, f1, f2);

        return result;
    }

    bool WinogradKernel1x3SetInputAutoTest(size_t block, const FuncWI& f1, const FuncWI& f2)
    {
        bool result = true;

        result = result && WinogradKernel1x3SetInputAutoTest(block, 0, 0, 0, f1, f2);
        result = result && WinogradKernel1x3SetInputAutoTest(block, 0, 0, 1, f1, f2);
        result = result && WinogradKernel1x3SetInputAutoTest(block, 1, 1, 0, f1, f2);
        result = result && WinogradKernel1x3SetInputAutoTest(block, 1, 1, 1, f1, f2);

        return result;
    }

    bool WinogradKernel1x3Block1x4SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel1x3SetInputAutoTest(4, FUNC_WI(Simd::Base::WinogradKernel1x3Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x3Block1x4SetInput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel1x3SetInputAutoTest(4, FUNC_WI(Simd::Sse2::WinogradKernel1x3Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x3Block1x4SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel1x3SetInputAutoTest(4, FUNC_WI(Simd::Avx::WinogradKernel1x3Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x3Block1x4SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel1x3SetInputAutoTest(4, FUNC_WI(Simd::Avx512f::WinogradKernel1x3Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x3Block1x4SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel1x3SetInputAutoTest(4, FUNC_WI(Simd::Neon::WinogradKernel1x3Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x3Block1x4SetInput));
#endif

        return result;
    }

    bool WinogradKernel1x5SetInputAutoTest(size_t block, int padB, int padE, int trans, const FuncWI& f1, const FuncWI& f2)
    {
        bool result = true;

        Size _core(5, 1), _block(block, 1), _padB(padB, 0), _padE(padE, 0);

        result = result && WinogradSetInputAutoTest(128, 5, 55, _block, _core, _padB, _padE, trans, f1, f2);

        return result;
    }

    bool WinogradKernel1x5SetInputAutoTest(size_t block, const FuncWI& f1, const FuncWI& f2)
    {
        bool result = true;

        result = result && WinogradKernel1x5SetInputAutoTest(block, 0, 0, 0, f1, f2);
        result = result && WinogradKernel1x5SetInputAutoTest(block, 0, 0, 1, f1, f2);
        result = result && WinogradKernel1x5SetInputAutoTest(block, 2, 2, 0, f1, f2);
        result = result && WinogradKernel1x5SetInputAutoTest(block, 2, 2, 1, f1, f2);

        return result;
    }

    bool WinogradKernel1x5Block1x4SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel1x5SetInputAutoTest(4, FUNC_WI(Simd::Base::WinogradKernel1x5Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x5Block1x4SetInput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel1x5SetInputAutoTest(4, FUNC_WI(Simd::Sse2::WinogradKernel1x5Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x5Block1x4SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel1x5SetInputAutoTest(4, FUNC_WI(Simd::Avx::WinogradKernel1x5Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x5Block1x4SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel1x5SetInputAutoTest(4, FUNC_WI(Simd::Avx512f::WinogradKernel1x5Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x5Block1x4SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel1x5SetInputAutoTest(4, FUNC_WI(Simd::Neon::WinogradKernel1x5Block1x4SetInput), FUNC_WI(SimdWinogradKernel1x5Block1x4SetInput));
#endif

        return result;
    }

    bool WinogradKernel2x2SetInputAutoTest(size_t block, int padB, int padE, int trans, const FuncWI& f1, const FuncWI& f2)
    {
        bool result = true;

        Size _core(2, 2), _block(block, block), _padB(padB, padB), _padE(padE, padE);

        result = result && WinogradSetInputAutoTest(128, 8, 60, _block, _core, _padB, _padE, trans, f1, f2);

        return result;
    }

    bool WinogradKernel2x2SetInputAutoTest(size_t block, const FuncWI& f1, const FuncWI& f2)
    {
        bool result = true;

        result = result && WinogradKernel2x2SetInputAutoTest(block, 0, 0, 0, f1, f2);
        result = result && WinogradKernel2x2SetInputAutoTest(block, 0, 0, 1, f1, f2);
        result = result && WinogradKernel2x2SetInputAutoTest(block, 1, 0, 0, f1, f2);
        result = result && WinogradKernel2x2SetInputAutoTest(block, 1, 0, 1, f1, f2);
        result = result && WinogradKernel2x2SetInputAutoTest(block, 0, 1, 0, f1, f2);
        result = result && WinogradKernel2x2SetInputAutoTest(block, 0, 1, 1, f1, f2);

        return result;
    }

    bool WinogradKernel2x2Block2x2SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel2x2SetInputAutoTest(2, FUNC_WI(Simd::Base::WinogradKernel2x2Block2x2SetInput), FUNC_WI(SimdWinogradKernel2x2Block2x2SetInput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel2x2SetInputAutoTest(2, FUNC_WI(Simd::Sse2::WinogradKernel2x2Block2x2SetInput), FUNC_WI(SimdWinogradKernel2x2Block2x2SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel2x2SetInputAutoTest(2, FUNC_WI(Simd::Avx::WinogradKernel2x2Block2x2SetInput), FUNC_WI(SimdWinogradKernel2x2Block2x2SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel2x2SetInputAutoTest(2, FUNC_WI(Simd::Avx512f::WinogradKernel2x2Block2x2SetInput), FUNC_WI(SimdWinogradKernel2x2Block2x2SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel2x2SetInputAutoTest(2, FUNC_WI(Simd::Neon::WinogradKernel2x2Block2x2SetInput), FUNC_WI(SimdWinogradKernel2x2Block2x2SetInput));
#endif

        return result;
    }

    bool WinogradKernel2x2Block4x4SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel2x2SetInputAutoTest(4, FUNC_WI(Simd::Base::WinogradKernel2x2Block4x4SetInput), FUNC_WI(SimdWinogradKernel2x2Block4x4SetInput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel2x2SetInputAutoTest(4, FUNC_WI(Simd::Sse2::WinogradKernel2x2Block4x4SetInput), FUNC_WI(SimdWinogradKernel2x2Block4x4SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel2x2SetInputAutoTest(4, FUNC_WI(Simd::Avx::WinogradKernel2x2Block4x4SetInput), FUNC_WI(SimdWinogradKernel2x2Block4x4SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel2x2SetInputAutoTest(4, FUNC_WI(Simd::Avx512f::WinogradKernel2x2Block4x4SetInput), FUNC_WI(SimdWinogradKernel2x2Block4x4SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel2x2SetInputAutoTest(4, FUNC_WI(Simd::Neon::WinogradKernel2x2Block4x4SetInput), FUNC_WI(SimdWinogradKernel2x2Block4x4SetInput));
#endif

        return result;
    }

    bool WinogradKernel3x3SetInputAutoTest(size_t block, int pad, int trans, const FuncWI & f1, const FuncWI & f2)
    {
        bool result = true;

        Size _core(3, 3), _block(block, block), _pad(pad, pad);

        result = result && WinogradSetInputAutoTest(64, 56, 48, _block, _core, _pad, _pad, trans, f1, f2);
        result = result && WinogradSetInputAutoTest(128, 28, 24, _block, _core, _pad, _pad, trans, f1, f2);
        result = result && WinogradSetInputAutoTest(256, 14, 12, _block, _core, _pad, _pad, trans, f1, f2);
        result = result && WinogradSetInputAutoTest(512, 7, 6, _block, _core, _pad, _pad, trans, f1, f2);
        if(block == 4 && pad)
        {
            result = result && WinogradSetInputAutoTest(64, 40, 159, _block, _core, _pad, Size(1, 0), trans, f1, f2);
            result = result && WinogradSetInputAutoTest(64, 41, 159, _block, _core, Size(1, 0), _pad, trans, f1, f2);
        }

        return result;
    }

    bool WinogradKernel3x3SetInputAutoTest(size_t block, const FuncWI & f1, const FuncWI & f2)
    {
        bool result = true;

        result = result && WinogradKernel3x3SetInputAutoTest(block, 0, 0, f1, f2);
        result = result && WinogradKernel3x3SetInputAutoTest(block, 0, 1, f1, f2);
        result = result && WinogradKernel3x3SetInputAutoTest(block, 1, 0, f1, f2);
        result = result && WinogradKernel3x3SetInputAutoTest(block, 1, 1, f1, f2);

        return result;
    }

    bool WinogradKernel3x3Block2x2SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel3x3SetInputAutoTest(2, FUNC_WI(Simd::Base::WinogradKernel3x3Block2x2SetInput), FUNC_WI(SimdWinogradKernel3x3Block2x2SetInput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(2, FUNC_WI(Simd::Sse2::WinogradKernel3x3Block2x2SetInput), FUNC_WI(SimdWinogradKernel3x3Block2x2SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(2, FUNC_WI(Simd::Avx::WinogradKernel3x3Block2x2SetInput), FUNC_WI(SimdWinogradKernel3x3Block2x2SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(2, FUNC_WI(Simd::Avx512f::WinogradKernel3x3Block2x2SetInput), FUNC_WI(SimdWinogradKernel3x3Block2x2SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(2, FUNC_WI(Simd::Neon::WinogradKernel3x3Block2x2SetInput), FUNC_WI(SimdWinogradKernel3x3Block2x2SetInput));
#endif

        return result;
    }

    bool WinogradKernel3x3Block3x3SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel3x3SetInputAutoTest(3, FUNC_WI(Simd::Base::WinogradKernel3x3Block3x3SetInput), FUNC_WI(SimdWinogradKernel3x3Block3x3SetInput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(3, FUNC_WI(Simd::Sse2::WinogradKernel3x3Block3x3SetInput), FUNC_WI(SimdWinogradKernel3x3Block3x3SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(3, FUNC_WI(Simd::Avx::WinogradKernel3x3Block3x3SetInput), FUNC_WI(SimdWinogradKernel3x3Block3x3SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(3, FUNC_WI(Simd::Avx512f::WinogradKernel3x3Block3x3SetInput), FUNC_WI(SimdWinogradKernel3x3Block3x3SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(3, FUNC_WI(Simd::Neon::WinogradKernel3x3Block3x3SetInput), FUNC_WI(SimdWinogradKernel3x3Block3x3SetInput));
#endif

        return result;
    }

    bool WinogradKernel3x3Block4x4SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel3x3SetInputAutoTest(4, FUNC_WI(Simd::Base::WinogradKernel3x3Block4x4SetInput), FUNC_WI(SimdWinogradKernel3x3Block4x4SetInput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(4, FUNC_WI(Simd::Sse2::WinogradKernel3x3Block4x4SetInput), FUNC_WI(SimdWinogradKernel3x3Block4x4SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(4, FUNC_WI(Simd::Avx::WinogradKernel3x3Block4x4SetInput), FUNC_WI(SimdWinogradKernel3x3Block4x4SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(4, FUNC_WI(Simd::Avx512f::WinogradKernel3x3Block4x4SetInput), FUNC_WI(SimdWinogradKernel3x3Block4x4SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel3x3SetInputAutoTest(4, FUNC_WI(Simd::Neon::WinogradKernel3x3Block4x4SetInput), FUNC_WI(SimdWinogradKernel3x3Block4x4SetInput));
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    namespace
    {
        struct FuncWO
        {
            typedef void(*FuncPtr)(const float * src, size_t srcStride, float * dst, size_t dstChannels, size_t dstHeight, size_t dstWidth, SimdBool trans);

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
                func(src.Data(), src.Size(1), dst.Data(), dstC, dstH, dstW, (SimdBool)trans);
            }
        };
    }

#define FUNC_WO(function) FuncWO(function, #function)

    bool WinogradSetOutputAutoTest(size_t dstC, size_t dstH, size_t dstW, const Size& block, const Size& core, int trans, FuncWO f1, FuncWO f2)
    {
        bool result = true;

        f1.Update(dstC, dstH, dstW, trans);
        f2.Update(dstC, dstH, dstW, trans);

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " .");

        size_t count = (block.x + core.x - 1) * (block.y + core.y - 1);
        size_t tileH = (dstH + block.y - 1) / block.y;
        size_t tileW = (dstW + block.x - 1) / block.x;

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

    bool WinogradKernel1x3SetOutputAutoTest(const Size& block, const Size& core, int trans, const FuncWO& f1, const FuncWO& f2)
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(128, 7, 59, block, core, trans, f1, f2);

        return result;
    }

    bool WinogradKernel1x3SetOutputAutoTest(size_t block, const FuncWO& f1, const FuncWO& f2)
    {
        bool result = true;

        Size _core(3, 1), _block(block, 1);

        result = result && WinogradKernel1x3SetOutputAutoTest(_block, _core, 0, f1, f2);
        result = result && WinogradKernel1x3SetOutputAutoTest(_block, _core, 1, f1, f2);

        return result;
    }

    bool WinogradKernel1x3Block1x4SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel1x3SetOutputAutoTest(4, FUNC_WO(Simd::Base::WinogradKernel1x3Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x3Block1x4SetOutput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel1x3SetOutputAutoTest(4, FUNC_WO(Simd::Sse2::WinogradKernel1x3Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x3Block1x4SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel1x3SetOutputAutoTest(4, FUNC_WO(Simd::Avx::WinogradKernel1x3Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x3Block1x4SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel1x3SetOutputAutoTest(4, FUNC_WO(Simd::Avx512f::WinogradKernel1x3Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x3Block1x4SetOutput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel1x3SetOutputAutoTest(4, FUNC_WO(Simd::Neon::WinogradKernel1x3Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x3Block1x4SetOutput));
#endif

        return result;
    }

    bool WinogradKernel1x5SetOutputAutoTest(const Size& block, const Size& core, int trans, const FuncWO& f1, const FuncWO& f2)
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(128, 5, 55, block, core, trans, f1, f2);

        return result;
    }

    bool WinogradKernel1x5SetOutputAutoTest(size_t block, const FuncWO& f1, const FuncWO& f2)
    {
        bool result = true;

        Size _core(5, 1), _block(block, 1);

        result = result && WinogradKernel1x3SetOutputAutoTest(_block, _core, 0, f1, f2);
        result = result && WinogradKernel1x3SetOutputAutoTest(_block, _core, 1, f1, f2);

        return result;
    }

    bool WinogradKernel1x5Block1x4SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel1x5SetOutputAutoTest(4, FUNC_WO(Simd::Base::WinogradKernel1x5Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x5Block1x4SetOutput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel1x5SetOutputAutoTest(4, FUNC_WO(Simd::Sse2::WinogradKernel1x5Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x5Block1x4SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel1x5SetOutputAutoTest(4, FUNC_WO(Simd::Avx::WinogradKernel1x5Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x5Block1x4SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel1x5SetOutputAutoTest(4, FUNC_WO(Simd::Avx512f::WinogradKernel1x5Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x5Block1x4SetOutput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel1x5SetOutputAutoTest(4, FUNC_WO(Simd::Neon::WinogradKernel1x5Block1x4SetOutput), FUNC_WO(SimdWinogradKernel1x5Block1x4SetOutput));
#endif

        return result;
    }

    bool WinogradKernel2x2SetOutputAutoTest(const Size& block, const Size& core, int trans, const FuncWO& f1, const FuncWO& f2)
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(256, 8, 60, block, core, trans, f1, f2);
        result = result && WinogradSetOutputAutoTest(256, 7, 59, block, core, trans, f1, f2);

        return result;
    }

    bool WinogradKernel2x2SetOutputAutoTest(size_t block, const FuncWO& f1, const FuncWO& f2)
    {
        bool result = true;

        Size _core(2, 2), _block(block, block);

        result = result && WinogradKernel2x2SetOutputAutoTest(_block, _core, 0, f1, f2);
        result = result && WinogradKernel2x2SetOutputAutoTest(_block, _core, 1, f1, f2);

        return result;
    }

    bool WinogradKernel2x2Block2x2SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel2x2SetOutputAutoTest(2, FUNC_WO(Simd::Base::WinogradKernel2x2Block2x2SetOutput), FUNC_WO(SimdWinogradKernel2x2Block2x2SetOutput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel2x2SetOutputAutoTest(2, FUNC_WO(Simd::Sse2::WinogradKernel2x2Block2x2SetOutput), FUNC_WO(SimdWinogradKernel2x2Block2x2SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel2x2SetOutputAutoTest(2, FUNC_WO(Simd::Avx::WinogradKernel2x2Block2x2SetOutput), FUNC_WO(SimdWinogradKernel2x2Block2x2SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel2x2SetOutputAutoTest(2, FUNC_WO(Simd::Avx512f::WinogradKernel2x2Block2x2SetOutput), FUNC_WO(SimdWinogradKernel2x2Block2x2SetOutput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel2x2SetOutputAutoTest(2, FUNC_WO(Simd::Neon::WinogradKernel2x2Block2x2SetOutput), FUNC_WO(SimdWinogradKernel2x2Block2x2SetOutput));
#endif

        return result;
    }

    bool WinogradKernel2x2Block4x4SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel2x2SetOutputAutoTest(4, FUNC_WO(Simd::Base::WinogradKernel2x2Block4x4SetOutput), FUNC_WO(SimdWinogradKernel2x2Block4x4SetOutput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel2x2SetOutputAutoTest(4, FUNC_WO(Simd::Sse2::WinogradKernel2x2Block4x4SetOutput), FUNC_WO(SimdWinogradKernel2x2Block4x4SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel2x2SetOutputAutoTest(4, FUNC_WO(Simd::Avx::WinogradKernel2x2Block4x4SetOutput), FUNC_WO(SimdWinogradKernel2x2Block4x4SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel2x2SetOutputAutoTest(4, FUNC_WO(Simd::Avx512f::WinogradKernel2x2Block4x4SetOutput), FUNC_WO(SimdWinogradKernel2x2Block4x4SetOutput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel2x2SetOutputAutoTest(4, FUNC_WO(Simd::Neon::WinogradKernel2x2Block4x4SetOutput), FUNC_WO(SimdWinogradKernel2x2Block4x4SetOutput));
#endif

        return result;
    }

    bool WinogradKernel3x3SetOutputAutoTest(const Size& block, const Size& core, int trans, const FuncWO & f1, const FuncWO & f2)
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(64, 56, 48, block, core, trans, f1, f2);
        result = result && WinogradSetOutputAutoTest(128, 28, 24, block, core, trans, f1, f2);
        result = result && WinogradSetOutputAutoTest(256, 14, 12, block, core, trans, f1, f2);
        result = result && WinogradSetOutputAutoTest(512, 7, 6, block, core, trans, f1, f2);

        return result;
    }

    bool WinogradKernel3x3SetOutputAutoTest(size_t block, const FuncWO & f1, const FuncWO & f2)
    {
        bool result = true;

        Size _core(3, 3), _block(block, block);

        result = result && WinogradKernel3x3SetOutputAutoTest(_block, _core, 0, f1, f2);
        result = result && WinogradKernel3x3SetOutputAutoTest(_block, _core, 1, f1, f2);

        return result;
    }

    bool WinogradKernel3x3Block2x2SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel3x3SetOutputAutoTest(2, FUNC_WO(Simd::Base::WinogradKernel3x3Block2x2SetOutput), FUNC_WO(SimdWinogradKernel3x3Block2x2SetOutput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(2, FUNC_WO(Simd::Sse2::WinogradKernel3x3Block2x2SetOutput), FUNC_WO(SimdWinogradKernel3x3Block2x2SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(2, FUNC_WO(Simd::Avx::WinogradKernel3x3Block2x2SetOutput), FUNC_WO(SimdWinogradKernel3x3Block2x2SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(2, FUNC_WO(Simd::Avx512f::WinogradKernel3x3Block2x2SetOutput), FUNC_WO(SimdWinogradKernel3x3Block2x2SetOutput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(2, FUNC_WO(Simd::Neon::WinogradKernel3x3Block2x2SetOutput), FUNC_WO(SimdWinogradKernel3x3Block2x2SetOutput));
#endif

        return result;
    }

    bool WinogradKernel3x3Block3x3SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel3x3SetOutputAutoTest(3, FUNC_WO(Simd::Base::WinogradKernel3x3Block3x3SetOutput), FUNC_WO(SimdWinogradKernel3x3Block3x3SetOutput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(3, FUNC_WO(Simd::Sse2::WinogradKernel3x3Block3x3SetOutput), FUNC_WO(SimdWinogradKernel3x3Block3x3SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(3, FUNC_WO(Simd::Avx::WinogradKernel3x3Block3x3SetOutput), FUNC_WO(SimdWinogradKernel3x3Block3x3SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(3, FUNC_WO(Simd::Avx512f::WinogradKernel3x3Block3x3SetOutput), FUNC_WO(SimdWinogradKernel3x3Block3x3SetOutput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(3, FUNC_WO(Simd::Neon::WinogradKernel3x3Block3x3SetOutput), FUNC_WO(SimdWinogradKernel3x3Block3x3SetOutput));
#endif

        return result;
    }

    bool WinogradKernel3x3Block4x4SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradKernel3x3SetOutputAutoTest(4, FUNC_WO(Simd::Base::WinogradKernel3x3Block4x4SetOutput), FUNC_WO(SimdWinogradKernel3x3Block4x4SetOutput));

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(4, FUNC_WO(Simd::Sse2::WinogradKernel3x3Block4x4SetOutput), FUNC_WO(SimdWinogradKernel3x3Block4x4SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(4, FUNC_WO(Simd::Avx::WinogradKernel3x3Block4x4SetOutput), FUNC_WO(SimdWinogradKernel3x3Block4x4SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(4, FUNC_WO(Simd::Avx512f::WinogradKernel3x3Block4x4SetOutput), FUNC_WO(SimdWinogradKernel3x3Block4x4SetOutput));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradKernel3x3SetOutputAutoTest(4, FUNC_WO(Simd::Neon::WinogradKernel3x3Block4x4SetOutput), FUNC_WO(SimdWinogradKernel3x3Block4x4SetOutput));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    void ImgToCol(const float * src, size_t srcC, size_t srcH, size_t srcW, const Size& core, const Size& padB, const Size& padE, float * dst)
    {
        size_t dstH = srcH - core.y + 1 + padB.y + padE.y;
        size_t dstW = srcW - core.x + 1 + padB.x + padE.x;
        const ptrdiff_t bodySize = dstW - padB.x - padE.x;
        for (size_t c = 0; c < srcC; ++c)
        {
            for (size_t ky = 0; ky < (size_t)core.y; ++ky)
            {
                for (size_t kx = 0; kx < (size_t)core.x; ++kx)
                {
                    size_t sy = ky - padB.y;
                    for (size_t dy = 0; dy < dstH; ++dy, ++sy)
                    {
                        if (sy < srcH)
                        {
                            size_t sx = kx - padB.x, dx = 0;
                            const float * psrc = src + sy * srcW;
                            for (; dx < (size_t)padB.x; ++dx, ++sx)
                            {
                                if (sx < srcW)
                                    *(dst++) = psrc[sx];
                                else
                                    *(dst++) = 0;
                            }
                            if (bodySize > 0)
                            {
                                memcpy(dst, psrc + sx, bodySize * sizeof(float));
                                dst += bodySize;
                                dx += bodySize;
                                sx += bodySize;
                            }
                            for (; dx < dstW; ++dx, ++sx)
                            {
                                if (sx < srcW)
                                    *(dst++) = psrc[sx];
                                else
                                    *(dst++) = 0;
                            }
                        }
                        else
                        {
                            memset(dst, 0, dstW * sizeof(float));
                            dst += dstW;
                        }
                    }
                }
            }
            src += srcW * srcH;
        }
    }

    void ImgToRow(const float * src, size_t srcC, size_t srcH, size_t srcW, const Size& core, const Size& padB, const Size& padE, float * dst)
    {
        size_t dstH = srcH - core.y + 1 + padB.y + padE.y;
        size_t dstW = srcW - core.x + 1 + padB.x + padE.x;
        for (size_t dy = 0; dy < dstH; ++dy)
        {
            for (size_t dx = 0; dx < dstW; ++dx)
            {
                for (size_t ky = 0; ky < (size_t)core.y; ky++)
                {
                    size_t sy = dy + ky - padB.y;
                    if (sy < srcH)
                    {
                        for (size_t kx = 0; kx < (size_t)core.x; kx++)
                        {
                            size_t sx = dx + kx - padB.x;
                            if (sx < srcW)
                            {
                                memcpy(dst, src + (sy * srcW + sx)*srcC, srcC * sizeof(float));
                                dst += srcC;
                            }
                            else
                            {
                                memset(dst, 0, srcC * sizeof(float));
                                dst += srcC;
                            }
                        }
                    }
                    else
                    {
                        memset(dst, 0, core.x * srcC * sizeof(float));
                        dst += core.x * srcC;
                    }
                }
            }
        }
    }

    bool WinogradSpecialTest(float eps, size_t srcC, size_t srcH, size_t srcW, size_t dstC, const Size& block, 
        const Size& core, const Size& padB, const Size& padE, SimdBool trans, FuncWF ff, FuncWI fi, FuncWO fo)
    {
        bool result = true;

        size_t dstH = srcH - core.y + 1 + padB.y + padE.y;
        size_t dstW = srcW - core.x + 1 + padB.x + padE.x;
        size_t count = (block.y + core.y - 1)* (block.x + core.x - 1);
        size_t tileH = (dstH + block.y - 1) / block.y;
        size_t tileW = (dstW + block.x - 1) / block.x;

        int pad = padB.x || padB.y || padE.x || padE.y ? 1 : 0;

        ff.Update(trans);
        fi.Update(srcC, srcH, srcW, pad, trans);
        fo.Update(dstC, dstH, dstW, trans);

        TEST_LOG_SS(Info, "Test " << ff.description << ", " << fi.description << ", " << fo.description << "].");

        Tensor32f src({ trans ? srcH : srcC, trans ? srcW : srcH, trans ? srcC : srcW });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);
        Tensor32f filter({ trans ? core.y : dstC, trans ? core.x : srcC, trans ? srcC : core.y, trans ? dstC : core.x });
        FillRandom(filter.Data(), filter.Size(), -1.0, 1.0f);

        Tensor32f gemmInput({ trans ? dstH : srcC, trans ? dstW : core.y * core.x, trans ? core.y * core.x : dstH, trans ? srcC : dstW });
        Tensor32f gemmDst({ trans ? dstH : dstC, trans ? dstW : dstH, trans ? dstC : dstW });
        Tensor32f winogradFilter({ count,  trans ? srcC : dstC, trans ? dstC : srcC });
        Tensor32f winogradInput({ count, trans ? tileH : srcC, trans ? tileW : tileH, trans ? srcC : tileW });
        Tensor32f winogradOutput({ count, trans ? tileH : dstC, trans ? tileW : tileH, trans ? dstC : tileW });
        Tensor32f winogradDst({ trans ? dstH : dstC, trans ? dstW : dstH, trans ? dstC : dstW });

        float _0 = 0, _1 = 1;
        if (trans)
        {
            ImgToRow(src.Data(), srcC, srcH, srcW, core, padB, padE, gemmInput.Data());
            size_t M = dstH * dstW, N = dstC, K = srcC * core.y * core.x;
            SimdGemm32fNN(M, N, K, &_1, gemmInput.Data(), K, filter.Data(), N, &_0, gemmDst.Data(), N);
        }
        else
        {
            ImgToCol(src.Data(), srcC, srcH, srcW, core, padB, padE, gemmInput.Data());
            size_t M = dstC, N = dstH * dstW, K = srcC * core.y * core.x;
            SimdGemm32fNN(M, N, K, &_1, filter.Data(), K, gemmInput.Data(), N, &_0, gemmDst.Data(), N);
        }

        ff.func(filter.Data(), srcC*dstC, winogradFilter.Data(), trans);
        fi.func(src.Data(), srcC, srcH, srcW, padB.y, padB.x, padE.y, padE.x, winogradInput.Data(), winogradInput.Size(1), trans);
        if (trans)
        {
            size_t M = tileW * tileH, N = dstC, K = srcC;
            for (size_t i = 0; i < count; ++i)
                SimdGemm32fNN(M, N, K, &_1, winogradInput.Data() + i * M * K, K, winogradFilter.Data() + i * K * N, N, &_0, winogradOutput.Data() + i * M * N, N);
        }
        else
        {
            size_t M = dstC, N = tileW * tileH, K = srcC;
            for (size_t i = 0; i < count; ++i)
                SimdGemm32fNN(M, N, K, &_1, winogradFilter.Data() + i * M * K, K, winogradInput.Data() + i * K * N, N, &_0, winogradOutput.Data() + i * M * N, N);
        }
        fo.func(winogradOutput.Data(), dstC * tileH * tileW, winogradDst.Data(), dstC, dstH, dstW, trans);

        result = result && Compare(gemmDst, winogradDst, eps, true, 64, DifferenceAbsolute);

        return result;
    }

    bool WinogradKernel1x3SpecialTest(size_t block, size_t padB, size_t padE, SimdBool trans, const FuncWF& ff, const FuncWI& fi, const FuncWO& fo)
    {
        bool result = true;

        Size _core(3, 1), _block(block, 1), _padB(padB, 0), _padE(padE, 0);

        result = result && WinogradSpecialTest(EPS * 1, 128, 8, 60, 256, _block, _core, _padB, _padE, trans, ff, fi, fo);
        result = result && WinogradSpecialTest(EPS * 1, 128, 9, 61, 256, _block, _core, _padB, _padE, trans, ff, fi, fo);

        return result;
    }

    bool WinogradKernel1x3SpecialTest(size_t block, const FuncWF& ff, const FuncWI& fi, const FuncWO& fo)
    {
        bool result = true;

        result = result && WinogradKernel1x3SpecialTest(block, 0, 0, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel1x3SpecialTest(block, 0, 0, ::SimdTrue, ff, fi, fo);
        result = result && WinogradKernel1x3SpecialTest(block, 1, 1, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel1x3SpecialTest(block, 1, 1, ::SimdTrue, ff, fi, fo);

        return result;
    }

    bool WinogradKernel1x3Block1x4SpecialTest()
    {
        return WinogradKernel1x3SpecialTest(4, FUNC_WF(SimdWinogradKernel1x3Block1x4SetFilter), FUNC_WI(SimdWinogradKernel1x3Block1x4SetInput), FUNC_WO(SimdWinogradKernel1x3Block1x4SetOutput));
    }

    bool WinogradKernel1x5SpecialTest(size_t block, size_t padB, size_t padE, SimdBool trans, const FuncWF& ff, const FuncWI& fi, const FuncWO& fo)
    {
        bool result = true;

        Size _core(5, 1), _block(block, 1), _padB(padB, 0), _padE(padE, 0);

        result = result && WinogradSpecialTest(EPS * 1, 128, 8, 60, 256, _block, _core, _padB, _padE, trans, ff, fi, fo);
        result = result && WinogradSpecialTest(EPS * 1, 128, 9, 61, 256, _block, _core, _padB, _padE, trans, ff, fi, fo);

        return result;
    }

    bool WinogradKernel1x5SpecialTest(size_t block, const FuncWF& ff, const FuncWI& fi, const FuncWO& fo)
    {
        bool result = true;

        result = result && WinogradKernel1x5SpecialTest(block, 0, 0, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel1x5SpecialTest(block, 0, 0, ::SimdTrue, ff, fi, fo);
        result = result && WinogradKernel1x5SpecialTest(block, 2, 2, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel1x5SpecialTest(block, 2, 2, ::SimdTrue, ff, fi, fo);

        return result;
    }

    bool WinogradKernel1x5Block1x4SpecialTest()
    {
        return WinogradKernel1x5SpecialTest(4, FUNC_WF(SimdWinogradKernel1x5Block1x4SetFilter), FUNC_WI(SimdWinogradKernel1x5Block1x4SetInput), FUNC_WO(SimdWinogradKernel1x5Block1x4SetOutput));
    }

    bool WinogradKernel2x2SpecialTest(size_t block, size_t padB, size_t padE, SimdBool trans, const FuncWF& ff, const FuncWI& fi, const FuncWO& fo)
    {
        bool result = true;

        Size _core(2, 2), _block(block, block), _padB(padB, padB), _padE(padE, padE);

        result = result && WinogradSpecialTest(EPS * 1, 128, 8, 60, 256, _block, _core, _padB, _padE, trans, ff, fi, fo);
        result = result && WinogradSpecialTest(EPS * 1, 128, 9, 61, 256, _block, _core, _padB, _padE, trans, ff, fi, fo);

        return result;
    }

    bool WinogradKernel2x2SpecialTest(size_t block, const FuncWF& ff, const FuncWI& fi, const FuncWO& fo)
    {
        bool result = true;

        result = result && WinogradKernel2x2SpecialTest(block, 0, 0, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel2x2SpecialTest(block, 0, 0, ::SimdTrue, ff, fi, fo);
        result = result && WinogradKernel2x2SpecialTest(block, 1, 0, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel2x2SpecialTest(block, 1, 0, ::SimdTrue, ff, fi, fo);
        result = result && WinogradKernel2x2SpecialTest(block, 0, 1, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel2x2SpecialTest(block, 0, 1, ::SimdTrue, ff, fi, fo);

        return result;
    }

    bool WinogradKernel2x2Block2x2SpecialTest()
    {
        return WinogradKernel2x2SpecialTest(2, FUNC_WF(SimdWinogradKernel2x2Block2x2SetFilter), FUNC_WI(SimdWinogradKernel2x2Block2x2SetInput), FUNC_WO(SimdWinogradKernel2x2Block2x2SetOutput));
    }

    bool WinogradKernel2x2Block4x4SpecialTest()
    {
        return WinogradKernel2x2SpecialTest(4, FUNC_WF(SimdWinogradKernel2x2Block4x4SetFilter), FUNC_WI(SimdWinogradKernel2x2Block4x4SetInput), FUNC_WO(SimdWinogradKernel2x2Block4x4SetOutput));
    }

    bool WinogradKernel3x3SpecialTest(size_t block, size_t pad, SimdBool trans, const FuncWF & ff, const FuncWI & fi, const FuncWO & fo)
    {
        bool result = true;

        Size _core(3, 3), _block(block, block), _pad(pad, pad);

        result = result && WinogradSpecialTest(EPS*1, 72, 112, 96, 64, _block, _core, _pad, _pad, trans, ff, fi, fo);
        result = result && WinogradSpecialTest(EPS*1, 144, 56, 48, 128, _block, _core, _pad, _pad, trans, ff, fi, fo);
        result = result && WinogradSpecialTest(EPS*2, 288, 28, 24, 256, _block, _core, _pad, _pad, trans, ff, fi, fo);

        return result;
    }

    bool WinogradKernel3x3SpecialTest(size_t block, const FuncWF & ff, const FuncWI & fi, const FuncWO & fo)
    {
        bool result = true;

        result = result && WinogradKernel3x3SpecialTest(block, 0, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel3x3SpecialTest(block, 0, ::SimdTrue, ff, fi, fo);
        result = result && WinogradKernel3x3SpecialTest(block, 1, ::SimdFalse, ff, fi, fo);
        result = result && WinogradKernel3x3SpecialTest(block, 1, ::SimdTrue, ff, fi, fo);

        return result;
    }

    bool WinogradKernel3x3Block2x2SpecialTest()
    {
        return WinogradKernel3x3SpecialTest(2, FUNC_WF(SimdWinogradKernel3x3Block2x2SetFilter), FUNC_WI(SimdWinogradKernel3x3Block2x2SetInput), FUNC_WO(SimdWinogradKernel3x3Block2x2SetOutput));
    }

    bool WinogradKernel3x3Block3x3SpecialTest()
    {
        return WinogradKernel3x3SpecialTest(3, FUNC_WF(SimdWinogradKernel3x3Block3x3SetFilter), FUNC_WI(SimdWinogradKernel3x3Block3x3SetInput), FUNC_WO(SimdWinogradKernel3x3Block3x3SetOutput));
    }

    bool WinogradKernel3x3Block4x4SpecialTest()
    {
        return WinogradKernel3x3SpecialTest(4, FUNC_WF(SimdWinogradKernel3x3Block4x4SetFilter), FUNC_WI(SimdWinogradKernel3x3Block4x4SetInput), FUNC_WO(SimdWinogradKernel3x3Block4x4SetOutput));
    }
#endif
}

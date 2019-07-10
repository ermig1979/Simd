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

    bool Winograd3x3SetFilterAutoTest()
    {
        bool result = true;

        result = result && WinogradSetFilterAutoTest(3, 3, FUNC_WF(Simd::Base::Winograd3x3SetFilter), FUNC_WF(SimdWinograd3x3SetFilter));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetFilterAutoTest(3, 3, FUNC_WF(Simd::Sse::Winograd3x3SetFilter), FUNC_WF(SimdWinograd3x3SetFilter));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetFilterAutoTest(3, 3, FUNC_WF(Simd::Avx::Winograd3x3SetFilter), FUNC_WF(SimdWinograd3x3SetFilter));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetFilterAutoTest(3, 3, FUNC_WF(Simd::Avx512f::Winograd3x3SetFilter), FUNC_WF(SimdWinograd3x3SetFilter));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetFilterAutoTest(3, 3, FUNC_WF(Simd::Neon::Winograd3x3SetFilter), FUNC_WF(SimdWinograd3x3SetFilter));
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
            typedef void(*FuncPtr)(const float * src, size_t srcChannels, size_t srcHeight, size_t srcWidth, float * dst, size_t dstStride, SimdBool pad, SimdBool trans);

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
                func(src.Data(), srcC, srcH, srcW, dst.Data(), dst.Size(1), (SimdBool)pad, (SimdBool)trans);
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

        result = result && WinogradSetInputAutoTest(block, core, 0, 0, f1, f2);
        result = result && WinogradSetInputAutoTest(block, core, 0, 1, f1, f2);
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

    bool Winograd3x3SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(3, 3, FUNC_WI(Simd::Base::Winograd3x3SetInput), FUNC_WI(SimdWinograd3x3SetInput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetInputAutoTest(3, 3, FUNC_WI(Simd::Sse::Winograd3x3SetInput), FUNC_WI(SimdWinograd3x3SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetInputAutoTest(3, 3, FUNC_WI(Simd::Avx::Winograd3x3SetInput), FUNC_WI(SimdWinograd3x3SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetInputAutoTest(3, 3, FUNC_WI(Simd::Avx512f::Winograd3x3SetInput), FUNC_WI(SimdWinograd3x3SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetInputAutoTest(3, 3, FUNC_WI(Simd::Neon::Winograd3x3SetInput), FUNC_WI(SimdWinograd3x3SetInput));
#endif

        return result;
    }

    bool Winograd4x3SetInputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetInputAutoTest(4, 3, FUNC_WI(Simd::Base::Winograd4x3SetInput), FUNC_WI(SimdWinograd4x3SetInput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetInputAutoTest(4, 3, FUNC_WI(Simd::Sse::Winograd4x3SetInput), FUNC_WI(SimdWinograd4x3SetInput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetInputAutoTest(4, 3, FUNC_WI(Simd::Avx::Winograd4x3SetInput), FUNC_WI(SimdWinograd4x3SetInput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetInputAutoTest(4, 3, FUNC_WI(Simd::Avx512f::Winograd4x3SetInput), FUNC_WI(SimdWinograd4x3SetInput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetInputAutoTest(4, 3, FUNC_WI(Simd::Neon::Winograd4x3SetInput), FUNC_WI(SimdWinograd4x3SetInput));
#endif

        return result;
    }

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

    bool Winograd3x3SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(3, 3, FUNC_WO(Simd::Base::Winograd3x3SetOutput), FUNC_WO(SimdWinograd3x3SetOutput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetOutputAutoTest(3, 3, FUNC_WO(Simd::Sse::Winograd3x3SetOutput), FUNC_WO(SimdWinograd3x3SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetOutputAutoTest(3, 3, FUNC_WO(Simd::Avx::Winograd3x3SetOutput), FUNC_WO(SimdWinograd3x3SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetOutputAutoTest(3, 3, FUNC_WO(Simd::Avx512f::Winograd3x3SetOutput), FUNC_WO(SimdWinograd3x3SetOutput));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetOutputAutoTest(3, 3, FUNC_WO(Simd::Neon::Winograd3x3SetOutput), FUNC_WO(SimdWinograd3x3SetOutput));
#endif

        return result;
    }

    bool Winograd4x3SetOutputAutoTest()
    {
        bool result = true;

        result = result && WinogradSetOutputAutoTest(4, 3, FUNC_WO(Simd::Base::Winograd4x3SetOutput), FUNC_WO(SimdWinograd4x3SetOutput));

#ifdef SIMD_SSE_ENABLE
        if (Simd::Sse::Enable)
            result = result && WinogradSetOutputAutoTest(4, 3, FUNC_WO(Simd::Sse::Winograd4x3SetOutput), FUNC_WO(SimdWinograd4x3SetOutput));
#endif 

#ifdef SIMD_AVX_ENABLE
        if (Simd::Avx::Enable)
            result = result && WinogradSetOutputAutoTest(4, 3, FUNC_WO(Simd::Avx::Winograd4x3SetOutput), FUNC_WO(SimdWinograd4x3SetOutput));
#endif 

#ifdef SIMD_AVX512F_ENABLE
        if (Simd::Avx512f::Enable)
            result = result && WinogradSetOutputAutoTest(4, 3, FUNC_WO(Simd::Avx512f::Winograd4x3SetOutput), FUNC_WO(SimdWinograd4x3SetOutput));
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && WinogradSetOutputAutoTest(4, 3, FUNC_WO(Simd::Neon::Winograd4x3SetOutput), FUNC_WO(SimdWinograd4x3SetOutput));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    void ImgToCol(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t core, SimdBool pad, float * dst)
    {
        size_t padY = pad ? (core - 1) / 2 : 0;
        size_t padX = pad ? (core - 1) / 2 : 0;
        size_t dstH = srcH - (pad ? 0 : core - 1);
        size_t dstW = srcW - (pad ? 0 : core - 1);
        const ptrdiff_t bodySize = dstW - 2*padX;
        for (size_t c = 0; c < srcC; ++c)
        {
            for (size_t ky = 0; ky < core; ++ky)
            {
                for (size_t kx = 0; kx < core; ++kx)
                {
                    size_t sy = ky - padY;
                    for (size_t dy = 0; dy < dstH; ++dy, ++sy)
                    {
                        if (sy < srcH)
                        {
                            size_t sx = kx - padX, dx = 0;
                            const float * psrc = src + sy * srcW;
                            for (; dx < padX; ++dx, ++sx)
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

    void ImgToRow(const float * src, size_t srcC, size_t srcH, size_t srcW, size_t core, SimdBool pad, float * dst)
    {
        size_t padY = pad ? (core - 1) / 2 : 0;
        size_t padX = pad ? (core - 1) / 2 : 0;
        size_t dstH = srcH - (pad ? 0 : core - 1);
        size_t dstW = srcW - (pad ? 0 : core - 1);
        for (size_t dy = 0; dy < dstH; ++dy)
        {
            for (size_t dx = 0; dx < dstW; ++dx)
            {
                for (size_t ky = 0; ky < core; ky++)
                {
                    size_t sy = dy + ky - padY;
                    if (sy < srcH)
                    {
                        for (size_t kx = 0; kx < core; kx++)
                        {
                            size_t sx = dx + kx - padX;
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
                        memset(dst, 0, core * srcC * sizeof(float));
                        dst += core * srcC;
                    }
                }
            }
        }
    }

    bool WinogradSpecialTest(float eps, size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t block, size_t core, SimdBool pad, SimdBool trans, FuncWF ff, FuncWI fi, FuncWO fo)
    {
        bool result = true;

        size_t dstW = pad ? srcW : srcW - core + 1;
        size_t dstH = pad ? srcH : srcH - core + 1;
        size_t count = Simd::Square(block + core - 1);
        size_t tileH = (dstH + block - 1) / block;
        size_t tileW = (dstW + block - 1) / block;

        ff.Update(trans);
        fi.Update(srcC, srcH, srcW, pad, trans);
        fo.Update(dstC, dstH, dstW, trans);

        TEST_LOG_SS(Info, "Test " << ff.description << ", " << fi.description << ", " << fo.description << "].");

        Tensor32f src({ trans ? srcH : srcC, trans ? srcW : srcH, trans ? srcC : srcW });
        FillRandom(src.Data(), src.Size(), -1.0, 1.0f);
        Tensor32f filter({ trans ? core : dstC, trans ? core : srcC, trans ? srcC : core, trans ? dstC : core });
        FillRandom(filter.Data(), filter.Size(), -1.0, 1.0f);

        Tensor32f gemmInput({ trans ? dstH : srcC, trans ? dstW : core * core, trans ? core * core : dstH, trans ? srcC : dstW });
        Tensor32f gemmDst({ trans ? dstH : dstC, trans ? dstW : dstH, trans ? dstC : dstW });
        Tensor32f winogradFilter({ count,  trans ? srcC : dstC, trans ? dstC : srcC });
        Tensor32f winogradInput({ count, trans ? tileH : srcC, trans ? tileW : tileH, trans ? srcC : tileW });
        Tensor32f winogradOutput({ count, trans ? tileH : dstC, trans ? tileW : tileH, trans ? dstC : tileW });
        Tensor32f winogradDst({ trans ? dstH : dstC, trans ? dstW : dstH, trans ? dstC : dstW });

        float _0 = 0, _1 = 1;
        if (trans)
        {
            ImgToRow(src.Data(), srcC, srcH, srcW, core, pad, gemmInput.Data());
            size_t M = dstH * dstW, N = dstC, K = srcC * core * core;
            SimdGemm32fNN(M, N, K, &_1, gemmInput.Data(), K, filter.Data(), N, &_0, gemmDst.Data(), N);
        }
        else
        {
            ImgToCol(src.Data(), srcC, srcH, srcW, core, pad, gemmInput.Data());
            size_t M = dstC, N = dstH * dstW, K = srcC * core * core;
            SimdGemm32fNN(M, N, K, &_1, filter.Data(), K, gemmInput.Data(), N, &_0, gemmDst.Data(), N);
        }

        ff.func(filter.Data(), srcC*dstC, winogradFilter.Data(), trans);
        fi.func(src.Data(), srcC, srcH, srcW, winogradInput.Data(), winogradInput.Size(1), pad, trans);
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

    bool WinogradSpecialTest(size_t block, size_t core, SimdBool pad, SimdBool trans, const FuncWF & ff, const FuncWI & fi, const FuncWO & fo)
    {
        bool result = true;

        result = result && WinogradSpecialTest(EPS*1, 72, 112, 96, 64, block, core, pad, trans, ff, fi, fo);
        result = result && WinogradSpecialTest(EPS*1, 144, 56, 48, 128, block, core, pad, trans, ff, fi, fo);
        result = result && WinogradSpecialTest(EPS*2, 288, 28, 24, 256, block, core, pad, trans, ff, fi, fo);

        return result;
    }

    bool WinogradSpecialTest(size_t block, size_t core, const FuncWF & ff, const FuncWI & fi, const FuncWO & fo)
    {
        bool result = true;

        result = result && WinogradSpecialTest(block, core, ::SimdFalse, ::SimdFalse, ff, fi, fo);
        result = result && WinogradSpecialTest(block, core, ::SimdFalse, ::SimdTrue, ff, fi, fo);
        result = result && WinogradSpecialTest(block, core, ::SimdTrue, ::SimdFalse, ff, fi, fo);
        result = result && WinogradSpecialTest(block, core, ::SimdTrue, ::SimdTrue, ff, fi, fo);

        return result;
    }

    bool Winograd2x3SpecialTest()
    {
        return WinogradSpecialTest(2, 3, FUNC_WF(SimdWinograd2x3SetFilter), FUNC_WI(SimdWinograd2x3SetInput), FUNC_WO(SimdWinograd2x3SetOutput));
    }

    bool Winograd3x3SpecialTest()
    {
        return WinogradSpecialTest(3, 3, FUNC_WF(SimdWinograd3x3SetFilter), FUNC_WI(SimdWinograd3x3SetInput), FUNC_WO(SimdWinograd3x3SetOutput));
    }

    bool Winograd4x3SpecialTest()
    {
        return WinogradSpecialTest(4, 3, FUNC_WF(SimdWinograd4x3SetFilter), FUNC_WI(SimdWinograd4x3SetInput), FUNC_WO(SimdWinograd4x3SetOutput));
    }
}

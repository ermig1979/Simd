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
#include "Test/TestData.h"

namespace Test
{
    namespace
    {
        struct Func
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t size, uint8_t * dst);

            FuncPtr func;
            String description;

            Func(const FuncPtr & f, const String & d) : func(f), description(d) {}

            void Call(const View & src, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(src.data, src.width, dst.data);
            }
        };
    }

#define FUNC(function) Func(function, #function)

    bool ReorderAutoTest(int size, const Func & f1, const Func & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " [" << size << "].");

        View s(size, 1, View::Gray8, NULL, TEST_ALIGN(size));
        FillRandom(s);

        View d1(size, 1, View::Gray8, NULL, TEST_ALIGN(size));
        View d2(size, 1, View::Gray8, NULL, TEST_ALIGN(size));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool ReorderAutoTest(const Func & f1, const Func & f2, int bytes)
    {
        bool result = true;

        result = result && ReorderAutoTest(W*H, f1, f2);
        result = result && ReorderAutoTest(W*H + O*bytes, f1, f2);
        result = result && ReorderAutoTest(W*H - O*bytes, f1, f2);

        return result;
    }

    bool Reorder16bitAutoTest()
    {
        bool result = true;

        result = result && ReorderAutoTest(FUNC(Simd::Base::Reorder16bit), FUNC(SimdReorder16bit), 2);

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse2::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse41::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx2::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx512bw::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Vmx::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Neon::Reorder16bit), FUNC(SimdReorder16bit), 2);
#endif

        return result;
    }

    bool Reorder32bitAutoTest()
    {
        bool result = true;

        result = result && ReorderAutoTest(FUNC(Simd::Base::Reorder32bit), FUNC(SimdReorder32bit), 4);

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse2::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse41::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx2::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx512bw::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Vmx::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Neon::Reorder32bit), FUNC(SimdReorder32bit), 4);
#endif

        return result;
    }

    bool Reorder64bitAutoTest()
    {
        bool result = true;

        result = result && ReorderAutoTest(FUNC(Simd::Base::Reorder64bit), FUNC(SimdReorder64bit), 8);

#ifdef SIMD_SSE2_ENABLE
        if (Simd::Sse2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse2::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Sse41::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx2::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Avx512bw::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif

#ifdef SIMD_VMX_ENABLE
        if (Simd::Vmx::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Vmx::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ReorderAutoTest(FUNC(Simd::Neon::Reorder64bit), FUNC(SimdReorder64bit), 8);
#endif

        return result;
    }

    //-----------------------------------------------------------------------

    bool ReorderDataTest(bool create, int size, const Func & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << size << "].");

        View s(size, 1, View::Gray8, NULL, TEST_ALIGN(size));
        View d1(size, 1, View::Gray8, NULL, TEST_ALIGN(size));
        View d2(size, 1, View::Gray8, NULL, TEST_ALIGN(size));

        if (create)
        {
            FillRandom(s);
            TEST_SAVE(s);

            f.Call(s, d1);

            TEST_SAVE(d1);
        }
        else
        {
            TEST_LOAD(s);
            TEST_LOAD(d1);

            f.Call(s, d2);

            TEST_SAVE(d2);

            result = result && Compare(d1, d2, 0, true, 64);
        }

        return result;
    }

    bool Reorder16bitDataTest(bool create)
    {
        bool result = true;

        result = result && ReorderDataTest(create, DH, FUNC(SimdReorder16bit));

        return result;
    }

    bool Reorder32bitDataTest(bool create)
    {
        bool result = true;

        result = result && ReorderDataTest(create, DH, FUNC(SimdReorder32bit));

        return result;
    }

    bool Reorder64bitDataTest(bool create)
    {
        bool result = true;

        result = result && ReorderDataTest(create, DH, FUNC(SimdReorder64bit));

        return result;
    }
}

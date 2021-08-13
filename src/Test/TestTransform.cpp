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
        struct FuncTI
        {
            typedef void(*FuncPtr)(const uint8_t * src, size_t srcStride, size_t width, size_t height, size_t pixelSize, SimdTransformType transform, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            String desc;

            FuncTI(const FuncPtr & f, const String & d) : func(f), desc(d) {}

            void Update(::SimdTransformType transform, size_t size)
            {
                std::stringstream ss;
                ss << desc << "[";
                switch (transform)
                {
                case ::SimdTransformRotate0:            ss << "N0"; break;
                case ::SimdTransformRotate90:           ss << "N1"; break;
                case ::SimdTransformRotate180:          ss << "N2"; break;
                case ::SimdTransformRotate270:          ss << "N3"; break;
                case ::SimdTransformTransposeRotate0:   ss << "T0"; break;
                case ::SimdTransformTransposeRotate90:  ss << "T1"; break;
                case ::SimdTransformTransposeRotate180: ss << "T2"; break;
                case ::SimdTransformTransposeRotate270: ss << "T3"; break;
                default:
                    assert(0);
                }
                ss << "-" << size << "]";
                desc = ss.str();
            }


            void Call(const View & src, SimdTransformType transform, View & dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                func(src.data, src.stride, src.width, src.height, src.PixelSize(), transform, dst.data, dst.stride);
            }
        };
    }

#define FUNC_TI(function) \
    FuncTI(function, std::string(#function))

    bool TransformImageAutoTest(::SimdTransformType transform, View::Format format, int width, int height, FuncTI f1, FuncTI f2)
    {
        bool result = true;

        f1.Update(transform, View::PixelSize(format));
        f2.Update(transform, View::PixelSize(format));

        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " [" << width << ", " << height << "].");

        View s(width, height, format, NULL, TEST_ALIGN(width));
        FillRandom(s);
#if 0
        if (format == View::Bgra32)
        {
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    s.At<Simd::Pixel::Bgra32>(x, y) = Simd::Pixel::Bgra32(x, y, Random(255), Random(255));
        }
        if (format == View::Bgr24)
        {
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                    s.At<Simd::Pixel::Bgr24>(x, y) = Simd::Pixel::Bgr24(y, x, Random(255));// Simd::Pixel::Bgr24(3 * x + 0, 3 * x + 1, 3 * x + 2);
        }
        if (format == View::Uv16)
        {
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x)
                {
                    s.Row<uint8_t>(y)[2 * x + 0] = y;
                    s.Row<uint8_t>(y)[2 * x + 1] = x;
                }
        }
#endif

        Size ds = Simd::TransformSize(s.Size(), transform);
        View d1(ds.x, ds.y, format, NULL, TEST_ALIGN(width));
        View d2(ds.x, ds.y, format, NULL, TEST_ALIGN(width));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(s, transform, d1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(s, transform, d2));

        result = result && Compare(d1, d2, 0, true, 64);

        return result;
    }

    bool TransformImageAutoTest(const FuncTI & f1, const FuncTI & f2)
    {
        bool result = true;

        for (::SimdTransformType transform = ::SimdTransformRotate0; transform <= ::SimdTransformTransposeRotate270; transform = ::SimdTransformType(transform + 1))
        {
            for (View::Format format = View::Gray8; format <= View::Bgra32; format = View::Format(format + 1))
            {
                //if (transform != ::SimdTransformTransposeRotate0 || format != View::Bgr24)
                //    continue;
                result = result && TransformImageAutoTest(transform, format, W, H, f1, f2);
                result = result && TransformImageAutoTest(transform, format, W + O, H - O, f1, f2);
            }
        }

        return result;
    }

    bool TransformImageAutoTest()
    {
        bool result = true;

        result = result && TransformImageAutoTest(FUNC_TI(Simd::Base::TransformImage), FUNC_TI(SimdTransformImage));

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable && W >= Simd::Avx2::A)
            result = result && TransformImageAutoTest(FUNC_TI(Simd::Avx512bw::TransformImage), FUNC_TI(SimdTransformImage));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable && W >= Simd::Avx2::A)
            result = result && TransformImageAutoTest(FUNC_TI(Simd::Avx2::TransformImage), FUNC_TI(SimdTransformImage));
#endif 

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable && W >= Simd::Sse41::A)
            result = result && TransformImageAutoTest(FUNC_TI(Simd::Sse41::TransformImage), FUNC_TI(SimdTransformImage));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable && W >= Simd::Neon::HA)
            result = result && TransformImageAutoTest(FUNC_TI(Simd::Neon::TransformImage), FUNC_TI(SimdTransformImage));
#endif 

        return result;
    }
}

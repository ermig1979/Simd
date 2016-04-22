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

#ifdef SIMD_DETECTION_ENABLE

namespace Test
{
    typedef std::map<size_t, View> Samples;
    std::recursive_mutex g_mutex;
    Samples g_samples;

    View GetSample(const Size & size)
    {
        std::lock_guard<std::recursive_mutex> lock(g_mutex);

        TEST_ALIGN(size.x);

        View & dst = g_samples[size.x];
        if (dst.format == View::Gray8)
            return dst;

        std::string path = "../../data/image/lena.pgm";
        View obj;
        if (!Load(obj, path))
        {
            TEST_LOG_SS(Error, "Can't load test image '" << path << "' !");
            return dst;
        }

        dst.Recreate(size.x, size.y, View::Gray8, NULL, TEST_ALIGN(size.x));

        uint8_t min, max, average;
        Simd::GetStatistic(obj, min, max, average);
        FillRandom(dst, (average + min) / 2, (average + max) / 2);

        if (obj.width < dst.width || obj.height < dst.height)
        {
            size_t rows = dst.height / obj.height * 2;
            size_t cols = dst.width / obj.width * 2;
            for (size_t row = 0; row < rows; ++row)
            {
                size_t y = dst.height * (row * 2 + 1) / (2 * rows);
                for (size_t col = 0; col < cols; ++col)
                {
                    size_t x = dst.width * (col * 2 + 1) / (2 * cols);
                    size_t s = (obj.width*2 + Random((int)obj.width)) / 10;

                    View resized(s, s, View::Gray8);
                    Simd::ResizeBilinear(obj, resized);

                    std::vector<uint8_t> profile(s, 255);
                    for (size_t i = 0, n = s / 4; i < n; ++i)
                        profile[s - i - 1] = profile[i] = uint8_t(i * 255 / n);
                    View alpha(s, s, View::Gray8);
                    Simd::VectorProduct(profile.data(), profile.data(), alpha);

                    Point p(x - s / 2, y - s / 2);
                    Simd::AlphaBlending(resized, alpha, dst.Region(p, p + Size(s, s)).Ref());
                }
            }
        }
        else
        {
            View _obj = obj.Region(obj.Size() * 4 / 7, View::MiddleCenter);
            size_t size = Simd::Min(dst.width, dst.height);
            View _dst = dst.Region(Size(size, size), View::MiddleCenter);
            Simd::ResizeBilinear(_obj, _dst);
        }

        return dst;
    }

    void Annotate(const View & src, const View & mask, size_t w, size_t h)
    {
        View dst(src.Size(), View::Gray8);
        Simd::Copy(src, dst);
        for (size_t row = 0; row < mask.height - h; ++row)
        {
            for (size_t col = 0; col < mask.width - w; ++col)
            {
                if (mask.At<uint8_t>(col, row) == 0)
                    continue;
                Simd::FillFrame(dst.Region(col, row, col + w, row + h).Ref(), Rect(1, 1, w - 1, h - 1), 255);
            }
        }
        Save(dst, "dst.pgm");
    }

    namespace
    {
        struct FuncD
        {
            typedef void(*FuncPtr)(const void * hid, const uint8_t * mask, size_t maskStride, 
                ptrdiff_t left, ptrdiff_t top, ptrdiff_t right, ptrdiff_t bottom, uint8_t * dst, size_t dstStride);

            FuncPtr func;
            std::string description;

            FuncD(const FuncPtr & f, const std::string & d) : func(f), description(d) {}

            void Call(const void * hid, const View & mask, const Rect & rect, View & dst) const
            {
                TEST_PERFORMANCE_TEST(description);
                func(hid, mask.data, mask.stride, rect.left, rect.top, rect.right, rect.bottom, dst.data, dst.stride);
            }
        };
    }

#define FUNC_D(function) FuncD(function, #function)

    bool DetectionHaarDetectAutoTest(const void * data, int width, int height, bool throughColumn, const FuncD & f1, const FuncD & f2)
    {
        bool result = true;

        TEST_LOG_SS(Info, "Test " << f1.description << " & " << f2.description << " for size [" << width << "," << height << "].");

        View src = GetSample(Size(width, height));
        if (src.format == View::None)
            return false;

        View sum(width + 1, height + 1, View::Int32);
        View sqsum(width + 1, height + 1, View::Int32);
        View tilted(width + 1, height + 1, View::Int32);

        void * hid = SimdDetectionHaarInit(data, sum.data, sum.stride, sum.width, sum.height, 
            sqsum.data, sqsum.stride, tilted.data, tilted.stride, throughColumn ? 1 : 0);
        if (hid == NULL)
        {
            TEST_LOG_SS(Error, "Can't init haar cascade!");
            return false;
        }

        View mask(width, height, View::Gray8);
        Simd::Fill(mask, 1);

        size_t w = SimdDetectionWindowWidth(data), h = SimdDetectionWindowHeight(data);
        Rect rect(0, 0, width - w, height - h);

        View dst1(width, height, View::Gray8);
        Simd::Fill(dst1, 0);

        View dst2(width, height, View::Gray8);
        Simd::Fill(dst2, 0);

        if(SimdDetectionHaarHasTilted(hid))
            Simd::Integral(src, sum, sqsum, tilted);
        else
            Simd::Integral(src, sum, sqsum);

        SimdDetectionHaarPrepare(hid);

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f1.Call(hid, mask, rect, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(f2.Call(hid, mask, rect, dst2));

        result = result && Compare(dst1, dst2, 0, true, 32);

        SimdDetectionHaarFree(hid);

        Annotate(src, dst1, w, h);

        return result;
    }

    bool DetectionHaarDetectAutoTest(const char * path, bool throughColumn, const FuncD & f1, const FuncD & f2)
    {
        bool result = true;

        void * data = SimdDetectionDataLoadA(path);
        if(data == NULL)
        {
            TEST_LOG_SS(Error, "Can't load haar cascade '" << path << "' !");
            return false;
        }

        result = result && DetectionHaarDetectAutoTest(data, W, H, throughColumn, f1, f2);
        //result = result && DetectionHaarDetectAutoTest(data, W + O, H - O, throughColumn, f1, f2);

        SimdDetectionDataFree(data);

        return result;
    }

    bool DetectionHaarDetectAutoTest(bool throughColumn, const FuncD & f1, const FuncD & f2)
    {
        bool result = true;

        result = result && DetectionHaarDetectAutoTest("../../data/cascade/haar_face_0.xml", throughColumn, f1, f2);
        //result = result && DetectionHaarDetectAutoTest("../../data/cascade/haar_face_1.xml", throughColumn, f1, f2);

        return result;
    }

    bool DetectionHaarDetect32fpAutoTest()
    {
        bool result = true;

        result = result && DetectionHaarDetectAutoTest(false, FUNC_D(Simd::Base::DetectionHaarDetect32fp), FUNC_D(SimdDetectionHaarDetect32fp));

        return result;
    }

    //-----------------------------------------------------------------------

    bool DetectionHaarDetect32fpDataTest(bool create, int width, int height, const FuncD & f)
    {
        bool result = true;

        Data data(f.description);

        TEST_LOG_SS(Info, (create ? "Create" : "Verify") << " test " << f.description << " [" << width << ", " << height << "].");

        if (create)
        {

        }
        else
        {

        }

        return result;
    }

    bool DetectionHaarDetect32fpDataTest(bool create)
    {
        bool result = true;

        result = result && DetectionHaarDetect32fpDataTest(create, DW, DH, FUNC_D(SimdDetectionHaarDetect32fp));

        return result;
    }
}

#endif

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

//-----------------------------------------------------------------------------

#ifdef TEST_PERFORMANCE_TEST_ENABLE
#define SIMD_CHECK_PERFORMANCE() TEST_PERFORMANCE_TEST_(__FUNCTION__)
#endif

#include "Simd/SimdImageMatcher.hpp"

namespace Test
{
    typedef Simd::ImageMatcher<size_t, Simd::Allocator> ImageMatcher;
    typedef Simd::Point<double> FPoint;
    typedef std::shared_ptr<View> ViewPtr;
    typedef std::vector<ViewPtr> ViewPtrs;
    typedef Test::Sums Indexes;

    void Fill(View & dst)
    {
        std::vector<uint8_t> v(dst.height);
        int v1 = Random(255), v2 = Random(255);
        for (int i = 0, n = (int)v.size(); i < n; ++i)
            v[i] = (i*v1 + (n - 1 - i)*v2) / (n - 1);
        std::vector<uint8_t> h(dst.width);
        int h1 = Random(255), h2 = Random(255);
        for (int i = 0, n = (int)h.size(); i < n; ++i)
            h[i] = (i*h1 + (n - 1 - i)*h2) / (n - 1);
        Simd::VectorProduct(v.data(), h.data(), dst);
    }

    void Multiply(const View & src, size_t factor, bool normalized, ViewPtrs & dst)
    {
        View alpha(src.Size(), View::Gray8);
        Simd::Fill(alpha, 64);
        for (size_t i = 0; i < factor; ++i)
        {
            ViewPtr shifted(new View(src.Size(), View::Gray8));
            if (normalized)
            {
                FPoint shift(Random() * 2 - 1, Random() * 2 - 1);
                Simd::ShiftBilinear(src, src, shift, Rect(src.Size()), *shifted);
            }
            else
            {
                Fill(*shifted);
                Simd::AlphaBlending(src, alpha, *shifted);
            }
            dst.push_back(shifted);
        }
    }

    bool CreateSamples(const Size & size, size_t factor, bool normalized, ViewPtrs & dst)
    {
        dst.clear();
        for (size_t i = 0, n = 10, current = 0, total = 0; i < n; ++i)
        {
            String path = ROOT_PATH + "/data/image/digit/" + char('0' + i) + ".pgm";
            View pooled;
            if (!pooled.Load(path))
            {
                TEST_LOG_SS(Error, "Can't load test image '" << path << "' !");
                return false;
            }
            Size number = pooled.Size() / size, shift;
            total += number.x*number.y;
            for (shift.y = 0; shift.y < number.y; ++shift.y)
                for (shift.x = 0; shift.x < number.x; ++shift.x, ++current)
                    Multiply(pooled.Region(shift*size, shift*size + size), factor, normalized, dst);
        }
        return true;
    }

    const size_t g_numbers[] = { 200, 2000, 20000 };
    const char * g_names[] = { "D0", "D1", "D3" };

    void PerformFiltration(const ViewPtrs & src, size_t size, double threshold, size_t type, bool normalized, Indexes & dst)
    {
        double time = GetTime();
        ImageMatcher matcher;
        matcher.Init(threshold, ImageMatcher::Hash16x16, g_numbers[type], normalized);
        for (size_t i = 0; i < src.size(); ++i)
        {
            ImageMatcher::HashPtr hash = matcher.Create(*src[i], i);
            ImageMatcher::Results results;
            if (!matcher.Find(hash, results))
            {
                matcher.Add(hash);
                dst.push_back((uint32_t)i);
            }
            if (i % 100 == 0)
            {
                std::cout << "Current : " << std::setprecision(1) << std::fixed << (100.0*i / src.size()) << "%). \r";
            }
        }
        TEST_LOG_SS(Info, "Filtration performance for " << g_names[type] << " : " << std::setprecision(3) << std::fixed << (GetTime() - time) << " s. ");
    }

    bool ImageMatcherSpecialTest()
    {
        bool result = true;

        const Size size(16, 16);
#ifdef NDEBUG
        const size_t factor = 20;
#else
        const size_t factor = 2;
#endif
        const double threshold = 0.03;
        bool normalized = false;

        ViewPtrs samples;
        if (!CreateSamples(size, factor, normalized, samples))
            return false;

        Indexes is0;
        PerformFiltration(samples, size.x, threshold, 0, normalized, is0);

        Indexes is1;
        PerformFiltration(samples, size.x, threshold, 1, normalized, is1);

        Indexes is2;
        PerformFiltration(samples, size.x, threshold, 2, normalized, is2);

        result = Compare(is0, is1, 0, true, 0, "D1");

        result = Compare(is1, is2, 0, true, 0, "D3");

        return result;
    }
}

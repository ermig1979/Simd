/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Test/TestCompare.h"
#include "Test/TestTensor.h"
#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdFont.hpp"
#include "Simd/SimdSynet.h"
#include "Test/TestRandom.h"

namespace Test
{
    void FillSequence(View & view)
    {
        for (size_t i = 0, n = view.DataSize(); i < n; ++i)
            view.data[i] = uint8_t(i);
    }

    template <size_t N> struct Color
    {
        uint8_t val[N];

        Color(uint8_t v = 0)
        {
            for (size_t i = 0; i < N; ++i)
                val[i] = v;
        }
    };

    template <size_t N> Color<N> RandomColor(uint8_t lo = 0, uint8_t hi = 255)
    {
        Color<N> color;
        for (size_t i = 0; i < N; ++i)
            color.val[i] = lo + Random(hi - lo + 1);
        return color;
    }

    template<size_t N> void FillPicture(View & view, uint64_t flag)
    {
        typedef Test::Color<N> Color;
        if (flag & 1)
        {
            Simd::Fill(view, 15);
        }
        if (flag & 2)
        {
            size_t d = view.height / 20;
            Rect rect(d, d, view.width - d, view.height - d);
            Simd::DrawRectangle(view, rect, RandomColor<N>(), d/2);
        }
        if (flag & 4)
        {
            size_t size = Simd::Min<size_t>(view.height / 2, 256);
            Simd::Font font(size);
            font.Draw(view, "A1", View::MiddleCenter, RandomColor<N>(128, 255));
        }
        if (flag & 8)
        {
            Rect rect(0, 0, view.width - 1, view.height - 1);
            Simd::DrawRectangle(view, rect, RandomColor<N>(255), 1);
        }
    }

    void FillPicture(View & view, uint64_t flag)
    {
        switch (view.PixelSize())
        {
        case 1: FillPicture<1>(view, flag); break;
        case 2: FillPicture<2>(view, flag); break;
        case 3: FillPicture<3>(view, flag); break;
        case 4: FillPicture<4>(view, flag); break;
        }
    }

    //---------------------------------------------------------------------------------------------

    template<class Color> Color GetColor(uint8_t b, uint8_t g, uint8_t r)
    {
        return Color(b, g, r);
    }

    template<> uint8_t GetColor<uint8_t>(uint8_t b, uint8_t g, uint8_t r)
    {
        return uint8_t((int(b) + int(g) + int(r)) / 3);
    }

    template<> Simd::Pixel::Rgb24 GetColor<Simd::Pixel::Rgb24>(uint8_t b, uint8_t g, uint8_t r)
    {
        return Simd::Pixel::Rgb24(r, g, b);
    }

    template<> Simd::Pixel::Rgba32 GetColor<Simd::Pixel::Rgba32>(uint8_t b, uint8_t g, uint8_t r)
    {
        return Simd::Pixel::Rgba32(r, g, b);
    }

    template<class Color> void DrawTestImage(View& canvas, int rects, int labels)
    {
        ::srand(0);
        int w = int(canvas.width), h = int(canvas.height);
        Simd::Fill(canvas, 0);

        for (int i = 0; i < rects; i++)
        {
            ptrdiff_t x1 = Random(w * 5 / 4) - w / 8;
            ptrdiff_t y1 = Random(h * 5 / 4) - h / 8;
            ptrdiff_t x2 = Random(w * 5 / 4) - w / 8;
            ptrdiff_t y2 = Random(h * 5 / 4) - h / 8;
            Rect rect(std::min(x1, x2), std::min(y1, y2), std::max(x1, x2), std::max(y1, y2));
            Color foreground = GetColor<Color>(Random(255), Random(255), Random(255));
            Simd::DrawFilledRectangle(canvas, rect, foreground);
        }

        String text = "First_string,\nSecond-line.";
        Simd::Font font(16);
        for (int i = 0; i < labels; i++)
        {
            ptrdiff_t x = Random(w) - w / 3;
            ptrdiff_t y = Random(h) - h / 6;
            Color foreground = GetColor<Color>(Random(255), Random(255), Random(255));
            font.Resize(Random(h / 4) + 16);
            font.Draw(canvas, text, Point(x, y), foreground);
        }

        font.Resize(h / 5);
        font.Draw(canvas, "B", View::BottomLeft, GetColor<Color>(255, 0, 0));
        font.Draw(canvas, "G", View::BottomCenter, GetColor<Color>(0, 255, 0));
        font.Draw(canvas, "R", View::BottomRight, GetColor<Color>(0, 0, 255));
    }

    void CreateTestImage(View& canvas, int rects, int labels)
    {
        switch (canvas.format)
        {
        case View::Gray8: DrawTestImage<uint8_t>(canvas, rects, labels); break;
        case View::Bgr24: DrawTestImage<Simd::Pixel::Bgr24>(canvas, rects, labels); break;
        case View::Bgra32: DrawTestImage<Simd::Pixel::Bgra32>(canvas, rects, labels); break;
        case View::Rgb24: DrawTestImage<Simd::Pixel::Rgb24>(canvas, rects, labels); break;
        case View::Rgba32: DrawTestImage<Simd::Pixel::Rgba32>(canvas, rects, labels); break;
        default: assert(0); break;
        }
    }

    //---------------------------------------------------------------------------------------------

    uint8_t g_rand8u[UINT16_MAX];
    bool InitRand8u()
    {
        for (size_t i = 0, n = UINT16_MAX; i < n; ++i)
            g_rand8u[i] = ::rand();
        return true;
    }
    bool g_rand8u_inited = InitRand8u();
    SIMD_INLINE const uint8_t * Rand8u()
    {
        return g_rand8u + (::rand()&INT16_MAX);
    }

    int16_t g_rand16i[UINT16_MAX];
    bool InitRand16i()
    {
        for (size_t i = 0, n = UINT16_MAX; i < n; ++i)
            g_rand16i[i] = (::rand() & INT16_MAX);
        return true;
    }
    bool g_rand16i_inited = InitRand16i();
    SIMD_INLINE const int16_t* Rand16i()
    {
        return g_rand16i + (::rand() & INT16_MAX);
    }

    uint16_t g_rand16u[UINT16_MAX];
    bool InitRand16u()
    {
        for (size_t i = 0, n = UINT16_MAX; i < n; ++i)
            g_rand16u[i] = (::rand() & UINT16_MAX);
        return true;
    }
    bool g_rand16u_inited = InitRand16u();
    SIMD_INLINE const uint16_t* Rand16u()
    {
        return g_rand16u + (::rand() & INT16_MAX);
    }

    void FillRandom(View & view, uint8_t lo, uint8_t hi)
    {
        assert(view.data);

        size_t width = view.width*View::PixelSize(view.format);
        bool fast = (lo == 0) && (hi == 255);
        for (size_t row = 0; row < view.height; ++row)
        {
            ptrdiff_t offset = row*view.stride;
            if (fast)
            {
                for (size_t col = 0; col < width; col += INT16_MAX)
                    memcpy(view.data + offset + col, Rand8u(), std::min<size_t>(INT16_MAX, width - col));
            }
            else
            {
                for (size_t col = 0; col < width; ++col, ++offset)
                    view.data[offset] = lo + Random(hi - lo + 1);
            }
        }
    }

    void FillRandom2(View & view, uint8_t lo, uint8_t hi, uint8_t step)
    {
        assert(view.data && view.height && view.width && view.format == View::Gray8);

        for (size_t row = 0; row < view.height; ++row)
        {
            if (row & 1)
            {
                for (ptrdiff_t col = view.width - 1; col >= 0; --col)
                {
                    int l = lo, h = hi;
                    if (row)
                    {
                        int v = view.At<uint8_t>(col, row - 1);
                        l = std::max(v - step, l);
                        h = std::min(v + step, h);
                    }
                    if (col != view.width - 1)
                    {
                        int v = view.At<uint8_t>(col + 1, row);
                        l = std::max(v - step / 2, l);
                        h = std::min(v + step / 2, h);
                    }
                    int r = h - l + 1;
                    int v = l + Random(r);
                    view.At<uint8_t>(col, row) = v;
                }
            }
            else
            {
                for (size_t col = 0; col < view.width; ++col)
                {
                    int l = lo, h = hi;
                    if (row)
                    {
                        int v = view.At<uint8_t>(col, row - 1);
                        l = std::max(v - step, l);
                        h = std::min(v + step, h);
                    }
                    if (col)
                    {
                        int v = view.At<uint8_t>(col - 1, row);
                        l = std::max(v - step / 2, l);
                        h = std::min(v + step / 2, h);
                    }
                    int r = h - l + 1;
                    int v = l + Random(r);
                    view.At<uint8_t>(col, row) = v;
                }
            }
        }

        View buff(view.Size(), View::Gray8);
        Simd::GaussianBlur3x3(view, buff);
        view.Swap(buff);
    }

    void FillRandomMask(View & view, uint8_t index)
    {
        assert(view.data);

        size_t width = view.width*View::PixelSize(view.format);
        for (size_t row = 0; row < view.height; ++row)
        {
            ptrdiff_t offset = row*view.stride;
            const uint8_t * rand = Rand8u();
            for (size_t col = 0; col < width; ++col, ++offset)
                view.data[offset] = (rand[col] & 1) ? index : 0;
        }
    }

    void FillRhombMask(View & mask, const Rect & rect, uint8_t index)
    {
        assert(mask.format == View::Gray8 && Rect(mask.Size()).Contains(rect));

        Simd::Fill(mask, 0);

        Point c = rect.Center();
        for (ptrdiff_t row = rect.top; row < rect.bottom; ++row)
        {
            ptrdiff_t indent = Simd::Abs(row - c.y)*rect.Width() / rect.Height();
            ptrdiff_t left = rect.left + indent;
            ptrdiff_t right = rect.right - indent;
            ptrdiff_t offset = row*mask.stride + left;
            const uint8_t * rand = Rand8u();
            for (ptrdiff_t col = left; col < right; ++col, ++offset)
                mask.data[offset] = (rand[col] & 1) ? index : 0;
        }
    }

    void FillRandom32f(View & view, float lo, float hi)
    {
        assert(view.format == View::Float);

        bool fast = view.Area() > 100000;
        float boost = (hi - lo) / UCHAR_MAX;
        for (size_t row = 0; row < view.height; ++row)
        {
            if (fast)
            {
                for (size_t col = 0; col < view.width; col += INT16_MAX)
                {
                    size_t size = std::min<size_t>(INT16_MAX, view.width - col);
                    float * dst = & view.At<float>(col, row);
                    const uint8_t * src = Rand8u();
                    for (size_t i = 0; i < size; ++i)
                        dst[i] = lo + boost*src[i];
                }
            }
            else
            {
                for (size_t col = 0; col < view.width; ++col)
                    view.At<float>(col, row) = lo + (hi - lo)*(float)Random();
            }
        }
    }

    void FillRandom16u(View& view, uint16_t lo, uint16_t hi)
    {
        assert(view.format == View::Int16);

        bool fast = view.Area() > 100000;
        for (size_t row = 0; row < view.height; ++row)
        {
            if (fast)
            {
                for (size_t col = 0; col < view.width; col += INT16_MAX)
                    memcpy(view.Row<uint16_t>(row) + col, Rand16u(), 2*std::min<size_t>(INT16_MAX, view.width - col));
            }
            else
            {
                for (size_t col = 0; col < view.width; ++col)
                    view.At<uint16_t>(col, row) = uint16_t(lo + Random(1 + hi - lo));
            }
        }
    }

    void FillRandom(float * data, size_t size, float lo, float hi)
    {
        bool fast = size > 100000;
        float boost = (hi - lo) / SHRT_MAX;
        if (fast)
        {
            for (size_t i = 0; i < size; i += INT16_MAX)
            {
                size_t n = std::min<size_t>(INT16_MAX, size - i);
                const int16_t * src = Rand16i();
                for (size_t j = 0; j < n; ++j)
                    data[i + j] = lo + boost * src[j];
            }
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
                data[i] = lo + (hi - lo)*(float)Random();
        }
    }

    void FillRandom(Buffer32f & buffer, float lo, float hi)
    {
        FillRandom(buffer.data(), buffer.size(), lo, hi);
    }

    void FillRandom(Tensor32f & tensor, float lo, float hi)
    {
        FillRandom(tensor.Data(), tensor.Size(), lo, hi);
    }

    void FillRandom(uint8_t* data, size_t size, uint8_t lo, uint8_t hi)
    {
        bool fast = (lo == 0) && (hi == 255);
        if (fast)
        {
            for (size_t i = 0; i < size; i += INT16_MAX)
                memcpy(data + i, Rand8u(), std::min<size_t>(INT16_MAX, size - i));
        }
        else
        {
            for (size_t i = 0; i < size; ++i)
                data[i] = lo + Random(hi - lo + 1);
        }
    }

    void FillRandom(Tensor8u & tensor, uint8_t lo, uint8_t hi)
    {
        FillRandom(tensor.Data(), tensor.Size(), lo, hi);
    }

    void FillRandom(Tensor8i& tensor, int8_t lo, int8_t hi)
    {
        for (size_t i = 0; i < tensor.Size(); ++i)
            tensor.Data()[i] = lo + Random(hi - lo + 1);
    }

    void FillRandom(Tensor32i& tensor, int32_t lo, int32_t hi)
    {
        for (size_t i = 0; i < tensor.Size(); ++i)
            tensor.Data()[i] = lo + Random(hi - lo + 1);
    }

    //-------------------------------------------------------------------------

    void FillRandom(Tensor32f & tensor, float* min, float* max, size_t channels, int negative, float upper, float range)
    {
        const float lower = negative ? -upper : 0.0f;
        Buffer32f buf(channels * 2);
        FillRandom(buf, lower + range, upper - range);
        for (size_t i = 0; i < channels; ++i)
        {
            min[i] = negative ? std::min(buf[i * 2 + 0], buf[i * 2 + 1]) - range : 0;
            max[i] = std::max(buf[i * 2 + 0], buf[i * 2 + 1]) + range;
        }
        FillRandom(tensor, 0.0f, 1.0f);
        if (tensor.Count() == 4)
        {
            for (size_t b = 0; b < tensor.Axis(0); ++b)
            {
                if (tensor.Format() == SimdTensorFormatNhwc)
                {
                    for (size_t y = 0; y < tensor.Axis(1); ++y)
                        for (size_t x = 0; x < tensor.Axis(2); ++x)
                            for (size_t c = 0; c < tensor.Axis(3); ++c)
                                tensor.Data({ b, y, x, c })[0] = min[c] + tensor.Data({ b, y, x, c })[0] * (max[c] - min[c]);
                }
                else
                {
                    for (size_t c = 0; c < tensor.Axis(1); ++c)
                        for (size_t y = 0; y < tensor.Axis(2); ++y)
                            for (size_t x = 0; x < tensor.Axis(3); ++x)
                                tensor.Data({ b, c, y, x })[0] = min[c] + tensor.Data({ b, c, y, x })[0] * (max[c] - min[c]);
                }
            }
        }
        else if (tensor.Count() == 3)
        {
            for (size_t b = 0; b < tensor.Axis(0); ++b)
            {
                if (tensor.Format() == SimdTensorFormatNhwc)
                {
                    for (size_t s = 0; s < tensor.Axis(1); ++s)
                        for (size_t c = 0; c < tensor.Axis(2); ++c)
                            tensor.Data({ b, s, c })[0] = min[c] + tensor.Data({ b, s, c })[0] * (max[c] - min[c]);
                }
                else
                {
                    for (size_t c = 0; c < tensor.Axis(1); ++c)
                        for (size_t s = 0; s < tensor.Axis(2); ++s)
                            tensor.Data({ b, c, s })[0] = min[c] + tensor.Data({ b, c, s })[0] * (max[c] - min[c]);
                }
            }
        }
        else
            assert(0);
    }
}

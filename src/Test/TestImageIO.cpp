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

#include "Simd/SimdImageLoad.h"
#include "Simd/SimdImageSave.h"

#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdFont.hpp"

namespace Test
{
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

        //for (int y = 0; y < h; y++)
        //    for (int x = 0; x < w; x++)
        //        canvas.At<Color>(x, y) = GetColor<Color>(Random(2), Random(2), Random(2));
        //}

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
        }
    }

    //-------------------------------------------------------------------------

    namespace
    {
        struct FuncSM
        {
            typedef Simd::ImageSaveToMemoryPtr FuncPtr;

            FuncPtr func;
            String desc;

            FuncSM(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(View::Format format, SimdImageFileType file, int quality)
            {
                desc = desc + "[" + ToString(format) + "-" + ToString(file) + 
                    (file == SimdImageFileJpeg ? String("-") + ToString(quality) : String("")) + "]";
            }

            void Call(const View& src, SimdImageFileType file, int quality, uint8_t** data, size_t* size) const
            {
                TEST_PERFORMANCE_TEST(desc);
                *data = func(src.data, src.stride, src.width, src.height, (SimdPixelFormatType)src.format, file, quality, size);
            }
        };
    }

#define FUNC_SM(func) \
    FuncSM(func, std::string(#func))

    bool GetTestImage(View& image, size_t width, size_t height, View::Format format, const String& desc1, const String& desc2)
    {
        if (REAL_IMAGE.empty())
        {
            TEST_LOG_SS(Info, "Test " << desc1 << " & " << desc2 << " [" << width << ", " << height << "].");
            image.Recreate(width, height, format, NULL, TEST_ALIGN(width));
#if 0
            ::srand(0);
            FillRandom(image);
            //src.Load("error.ppm", format);
#else
            CreateTestImage(image, 10, 10);
#endif        
        }
        else
        {
            String path = ROOT_PATH + "/data/image/" + REAL_IMAGE;
            if (!image.Load(path, format))
            {
                TEST_LOG_SS(Error, "Can't load image from '" << path << "'!");
                return false;
            }
            TEST_LOG_SS(Info, "Test " << desc1 << " & " << desc2 << " at " << REAL_IMAGE << " [" << image.width << "x" << image.height << "].");
        }
        TEST_ALIGN(SIMD_ALIGN);
        return true;
    }

    bool ImageSaveToMemoryAutoTest(size_t width, size_t height, View::Format format, SimdImageFileType file, int quality, FuncSM f1, FuncSM f2)
    {
        bool result = true;

        f1.Update(format, file, quality);
        f2.Update(format, file, quality);

        View src;
        if (!GetTestImage(src, width, height, format, f1.desc, f2.desc))
            return false;

        uint8_t* data1 = NULL, * data2 = NULL;
        size_t size1 = 0, size2 = 0;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (data1) Simd::Free(data1); f1.Call(src, file, quality, &data1, &size1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (data2) SimdFree(data2); f2.Call(src, file, quality, &data2, &size2));

        if (file == SimdImageFileJpeg)
        {
            View dst1, dst2;
            if (dst1.Load(data1, size1, format) && dst2.Load(data2, size2, format))
            {
                int differenceMax = REAL_IMAGE.empty() ? 4 : 4;
                result = result && Compare(dst1, dst2, differenceMax, true, 64, 0, "dst1 & dst2");
                if (!result)
                {
                    dst1.Save((ToString(format) + "_" + ToString(quality) + "_1.jpg").c_str(), file, quality);
                    dst2.Save((ToString(format) + "_" + ToString(quality) + "_2.jpg").c_str(), file, quality);
                    src.Save("error.ppm", SimdImageFilePpmTxt);
                }
            }
            else
            {
                TEST_LOG_SS(Error, "Can't load images from memory!");
                result = false;
            }
        }
        else
            result = result && Compare(data1, size1, data2, size2, 0, true, 64);

        if (data1)
            Simd::Free(data1);
        if (data2)
            SimdFree(data2);

        if (file == SimdImageFilePpmBin)
            src.Save((ToString(format) + ".ppm").c_str(), file, quality);
        if(file == SimdImageFilePng)
            src.Save((ToString(format) + ".png").c_str(), file, quality);
        if (file == SimdImageFileJpeg)
            src.Save((ToString(format) + "_" + ToString(quality) + ".jpg").c_str(), file, quality);

        return result;
    }

    bool ImageSaveToMemoryAutoTest(View::Format format, SimdImageFileType file, int quality, const FuncSM& f1, const FuncSM& f2)
    {
        bool result = true;

        result = result && ImageSaveToMemoryAutoTest(W, H, format, file, quality, f1, f2);
#if !defined(TEST_REAL_IMAGE)
        result = result && ImageSaveToMemoryAutoTest(W + O, H - O, format, file, quality, f1, f2);
#endif
        return result;
    }

    bool ImageSaveToMemoryAutoTest(const FuncSM & f1, const FuncSM& f2)
    {
        bool result = true;

        View::Format formats[5] = { View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24, View::Rgba32 };
        for (int format = 0; format < 5; format++)
        {
            for (int file = (int)SimdImageFilePng; file <= (int)SimdImageFilePng; file++)
            {
                if (file == SimdImageFileJpeg)
                {
                    result = result && ImageSaveToMemoryAutoTest(formats[format], (SimdImageFileType)file, 100, f1, f2);
                    result = result && ImageSaveToMemoryAutoTest(formats[format], (SimdImageFileType)file, 95, f1, f2);
                    result = result && ImageSaveToMemoryAutoTest(formats[format], (SimdImageFileType)file, 10, f1, f2);
                }
                result = result && ImageSaveToMemoryAutoTest(formats[format], (SimdImageFileType)file, 65, f1, f2);
            }
        }

        return result;
    }

    bool ImageSaveToMemoryAutoTest()
    {
        bool result = true;

        result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Base::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Sse41::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Avx2::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Avx512bw::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ImageSaveToMemoryAutoTest(FUNC_SM(Simd::Neon::ImageSaveToMemory), FUNC_SM(SimdImageSaveToMemory));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    namespace
    {
        struct FuncLM
        {
            typedef Simd::ImageLoadFromMemoryPtr FuncPtr;

            FuncPtr func;
            String desc;

            FuncLM(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(View::Format format, SimdImageFileType file, int quality)
            {
                desc = desc + "[" + ToString(format) + "-" + ToString(file) +
                    (file == SimdImageFileJpeg ? String("-") + ToString(quality) : String("")) + "]";
            }

            void Call(const uint8_t* data, size_t size, View::Format format, View& dst) const
            {
                TEST_PERFORMANCE_TEST(desc);
                ((View::Format&)dst.format) = format;
                *(uint8_t**)&dst.data = func(data, size, (size_t*)&dst.stride, (size_t*)&dst.width, (size_t*)&dst.height, (SimdPixelFormatType*)&dst.format);
            }
        };
    }

#define FUNC_LM(func) \
    FuncLM(func, std::string(#func))

    bool SaveLoadCompatible(View::Format format, SimdImageFileType file, int quality)
    {
        if (file == SimdImageFilePgmTxt || file == SimdImageFilePgmBin)
            return format == View::Gray8;
        if (file == SimdImageFilePpmTxt || file == SimdImageFilePpmBin)
            return format != View::Bgra32 && format != View::Rgba32;
        return false;
    }

    bool ImageLoadFromMemoryAutoTest(size_t width, size_t height, View::Format format, SimdImageFileType file, int quality, FuncLM f1, FuncLM f2)
    {
        bool result = true;

        f1.Update(format, file, quality);
        f2.Update(format, file, quality);

        View src;
        if (!GetTestImage(src, width, height, format, f1.desc, f2.desc))
            return false;

        size_t size = 0;
        uint8_t* data = SimdImageSaveToMemory(src.data, src.stride, src.width, src.height, (SimdPixelFormatType)src.format, file, quality, &size);

        View dst1, dst2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (dst1.data) Simd::Free(dst1.data); f1.Call(data, size, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (dst2.data) SimdFree(dst2.data); f2.Call(data, size, format, dst2));

        if (file == SimdImageFileJpeg)
        {
            int differenceMax = REAL_IMAGE.empty() ? 4 : 4;
            result = result && Compare(dst1, dst2, differenceMax, true, 64, 0, "dst1 & dst2");
            if (!result)
            {
                dst1.Save((ToString(format) + "_" + ToString(quality) + "_1.jpg").c_str(), file, quality);
                dst2.Save((ToString(format) + "_" + ToString(quality) + "_2.jpg").c_str(), file, quality);
                src.Save("error.ppm", SimdImageFilePpmTxt);
            }
        }
        else
        {
            if (dst1.data && dst2.data)
                result = result && Compare(dst1, dst2, 0, true, 64, 0, "dst1 & dst2");
            if (dst1.data && SaveLoadCompatible(format, file, quality))
                result = result && Compare(dst1, src, 0, true, 64, 0, "dst1 & src");
        }

        if (dst1.data)
            Simd::Free(dst1.data);
        if (dst2.data)
            SimdFree(dst2.data);
        SimdFree(data);

        return result;
    }

    bool ImageLoadFromMemoryAutoTest(View::Format format, SimdImageFileType file, int quality, FuncLM f1, FuncLM f2)
    {
        bool result = true;

        result = result && ImageLoadFromMemoryAutoTest(W, H, format, file, quality, f1, f2);
        result = result && ImageLoadFromMemoryAutoTest(W + O, H - O, format, file, quality, f1, f2);

        return result;
    }

    bool ImageLoadFromMemoryAutoTest(const FuncLM& f1, const FuncLM& f2)
    {
        bool result = true;

        View::Format formats[5] = { View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24, View::Rgba32 };
        for (int format = 0; format < 5; format++)
        {
            for (int file = (int)SimdImageFilePng; file <= (int)SimdImageFilePng; file++)
            {
                if (file == SimdImageFileJpeg)
                {
                    result = result && ImageLoadFromMemoryAutoTest(formats[format], (SimdImageFileType)file, 100, f1, f2);
                    result = result && ImageLoadFromMemoryAutoTest(formats[format], (SimdImageFileType)file, 95, f1, f2);
                    result = result && ImageLoadFromMemoryAutoTest(formats[format], (SimdImageFileType)file, 10, f1, f2);
                }
                result = result && ImageLoadFromMemoryAutoTest(formats[format], (SimdImageFileType)file, 65, f1, f2);
            }
        }

        return result;
    }

    bool ImageLoadFromMemoryAutoTest()
    {
        bool result = true;

        result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Base::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Sse41::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Avx2::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Avx512bw::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));
#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Neon::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));
#endif 

        return result;
    }
}

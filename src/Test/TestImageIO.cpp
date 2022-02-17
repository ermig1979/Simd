/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Test/TestFile.h"
#include "Test/TestPerformance.h"
#include "Test/TestData.h"
#include "Test/TestString.h"

#include "Simd/SimdImageLoad.h"
#include "Simd/SimdImageSave.h"

#include "Simd/SimdDrawing.hpp"
#include "Simd/SimdFont.hpp"

namespace Test
{
    const int DebugImageSave = 1;

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

    bool FileLoad(const char* path, uint8_t** data, size_t* size)
    {
        bool result = false;
        ::FILE* file = ::fopen(path, "rb");
        if (file)
        {
            ::fseek(file, 0, SEEK_END);
            *size = ::ftell(file);
            *data = (uint8_t*)SimdAllocate(*size, SimdAlignment());
            ::fseek(file, 0, SEEK_SET);
            result = ::fread(*data, 1, *size, file) == *size;
            ::fclose(file);
        }
        return result;
    }

    bool FileSave(const uint8_t* data, size_t size, const char *path)
    {
        bool result = false;
        ::FILE* file = ::fopen(path, "wb");
        if (file)
        {
            result = ::fwrite(data, 1, size, file) == size;
            ::fclose(file);
        }
        return result;
    }

    bool GetTestImage(View& image, size_t width, size_t height, View::Format format, 
        const String& desc1, const String& desc2, SimdImageFileType file, int quality, uint8_t ** data, size_t *size)
    {
        bool result = true;
        String path;
        if (REAL_IMAGE.empty())
        {
            TEST_LOG_SS(Info, "Test " << desc1 << " & " << desc2 << " [" << width << ", " << height << "].");
            image.Recreate(width, height, format, NULL, TEST_ALIGN(width));
#if 0
            ::srand(0);
            FillRandom(image);
#else
            CreateTestImage(image, 10, 10);
#endif
        }
        else
        {
            path = ROOT_PATH + "/data/image/" + REAL_IMAGE;
            if (!image.Load(path, format))
            {
                TEST_LOG_SS(Error, "Can't load image from '" << path << "'!");
                return false;
            }
            TEST_LOG_SS(Info, "Test " << desc1 << " & " << desc2 << " at " << REAL_IMAGE << " [" << image.width << "x" << image.height << "].");
        }
        if (data && size)
        {
            if(ToLower(ExtensionByPath(path)) == ToExtension(file))
                result = FileLoad(path.c_str(), data, size);
            else
            {
                *data = SimdImageSaveToMemory(image.data, image.stride, image.width, image.height, (SimdPixelFormatType)image.format, file, quality, size);
                result = *data && *size;
            }
        }        
        TEST_ALIGN(SIMD_ALIGN);
        return result;
    }

    bool SaveTestImage(const View& image, SimdImageFileType file, int quality, const String & suffix = "")
    {
        if (file < SimdImageFilePpmBin)
            return true;
        std::stringstream ss;
        ss << ToString(image.format);
        if (file == SimdImageFilePpmBin)
            ss << suffix << ".ppm";
        else if (file == SimdImageFilePng)
            ss << suffix << ".png";
        if (file == SimdImageFileJpeg)
            ss << "_" << ToString(quality) << suffix << ".jpg";
        const String dir = "_out";
        String path = MakePath(dir, ss.str());
        return CreatePathIfNotExist(dir, false) && image.Save(path, file, quality);
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

    bool ImageSaveToMemoryAutoTest(size_t width, size_t height, View::Format format, SimdImageFileType file, int quality, FuncSM f1, FuncSM f2)
    {
        bool result = true;

        f1.Update(format, file, quality);
        f2.Update(format, file, quality);

        View src;
        if (!GetTestImage(src, width, height, format, f1.desc, f2.desc, file, quality, NULL, NULL))
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
                    SaveTestImage(dst1, file, quality, "_1");
                    SaveTestImage(dst2, file, quality, "_2");
                    SaveTestImage(src, SimdImageFilePpmBin, 100, "_error");
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

        if(DebugImageSave)
            SaveTestImage(src, file, quality);

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

        std::vector<View::Format> formats({ View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24, View::Rgba32});
        for (int format = 0; format < formats.size(); format++)
        {
            for (int file = (int)SimdImageFileJpeg; file <= (int)SimdImageFileJpeg; file++)
            {
                if (file == SimdImageFileJpeg)
                {
                    result = result && ImageSaveToMemoryAutoTest(formats[format], (SimdImageFileType)file, 100, f1, f2);
                    result = result && ImageSaveToMemoryAutoTest(formats[format], (SimdImageFileType)file, 95, f1, f2);
                    result = result && ImageSaveToMemoryAutoTest(formats[format], (SimdImageFileType)file, 85, f1, f2);
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
        struct FuncSNJM
        {
            typedef uint8_t* (*FuncPtr)(const uint8_t* y, size_t yStride, const uint8_t* uv, size_t uvStride,
                size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size);

            FuncPtr func;
            String desc;

            FuncSNJM(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(int quality, SimdYuvType yuvType)
            {
                desc = desc + "[" + ToString(quality) + "]";
            }

            void Call(const View& y, const View& uv, SimdYuvType yuvType, int quality, uint8_t** data, size_t* size) const
            {
                TEST_PERFORMANCE_TEST(desc);
                *data = func(y.data, y.stride, uv.data, uv.stride, y.width, y.height, yuvType, quality, size);
            }
        };
    }

#define FUNC_SNJM(func) \
    FuncSNJM(func, std::string(#func))

    bool Nv12SaveAsJpegToMemoryAutoTest(size_t width, size_t height, SimdYuvType yuvType, int quality, FuncSNJM f1, FuncSNJM f2)
    {
        bool result = true;

        f1.Update(quality, yuvType);
        f2.Update(quality, yuvType);

        View bgra;
        if (!GetTestImage(bgra, width, height, View::Bgra32, f1.desc, f2.desc, SimdImageFileJpeg, quality, NULL, NULL))
            return false;

        View y(width, height, View::Gray8);
        View u(width / 2, height / 2, View::Gray8);
        View v(width / 2, height / 2, View::Gray8);
        SimdBgraToYuv420pV2(bgra.data, bgra.stride, bgra.width, bgra.height, y.data, y.stride, u.data, u.stride, v.data, v.stride, yuvType);

        View uv(width / 2, height / 2, View::Uv16);
        Simd::InterleaveUv(u, v, uv);

        uint8_t* data1 = NULL, * data2 = NULL;
        size_t size1 = 0, size2 = 0;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (data1) Simd::Free(data1); f1.Call(y, uv, yuvType, quality, &data1, &size1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (data2) SimdFree(data2); f2.Call(y, uv, yuvType, quality, &data2, &size2));

        View dst1, dst2;
        if (dst1.Load(data1, size1, View::Bgra32) && dst2.Load(data2, size2, View::Bgra32))
        {
            int differenceMax = REAL_IMAGE.empty() ? 4 : 4;
            result = result && Compare(dst1, dst2, differenceMax, true, 64, 0, "dst1 & dst2");
            if (!result)
            {
                SaveTestImage(dst1, SimdImageFileJpeg, quality, "_1");
                SaveTestImage(dst2, SimdImageFileJpeg, quality, "_2");
                SaveTestImage(bgra, SimdImageFilePpmBin, 100, "_error");
            }
        }
        else
        {
            TEST_LOG_SS(Error, "Can't load images from memory!");
            result = false;
        }

        if (data1)
        {
            if (DebugImageSave)
            {
                const String dir = "_out";
                if (CreatePathIfNotExist(dir, false))
                {
                    String path = MakePath(dir, String("Bgra32_") + ToString(quality) + "_nv12.jpg");
                    FileSave(data1, size1, path.c_str());
                }
            }
            Simd::Free(data1);
        }
        if (data2)
            SimdFree(data2);

        if(DebugImageSave)
            SaveTestImage(bgra, SimdImageFileJpeg, quality, "_src");

        return result;
    }

    bool Nv12SaveAsJpegToMemoryAutoTest(const FuncSNJM& f1, const FuncSNJM& f2)
    {
        bool result = true;

        Ints qualities({ 100, 95, 85, 65, 10 });
        std::vector<SimdYuvType> yuvTypes({ SimdYuvTrect871 });

        for (size_t t = 0; t < yuvTypes.size() && result; ++t)
        {
            for (size_t q = 0; q < qualities.size() && result; ++q)
            {
                result = result && Nv12SaveAsJpegToMemoryAutoTest(W, H, yuvTypes[t], qualities[q], f1, f2);
#if !defined(TEST_REAL_IMAGE)
                result = result && Nv12SaveAsJpegToMemoryAutoTest(W + E, H - E, yuvTypes[t], qualities[q], f1, f2);
#endif
            }
        }

        return result;
    }

    bool Nv12SaveAsJpegToMemoryAutoTest()
    {
        bool result = true;

        result = result && Nv12SaveAsJpegToMemoryAutoTest(FUNC_SNJM(Simd::Base::Nv12SaveAsJpegToMemory), FUNC_SNJM(SimdNv12SaveAsJpegToMemory));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && Nv12SaveAsJpegToMemoryAutoTest(FUNC_SNJM(Simd::Sse41::Nv12SaveAsJpegToMemory), FUNC_SNJM(SimdNv12SaveAsJpegToMemory));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Nv12SaveAsJpegToMemoryAutoTest(FUNC_SNJM(Simd::Avx2::Nv12SaveAsJpegToMemory), FUNC_SNJM(SimdNv12SaveAsJpegToMemory));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Nv12SaveAsJpegToMemoryAutoTest(FUNC_SNJM(Simd::Avx512bw::Nv12SaveAsJpegToMemory), FUNC_SNJM(SimdNv12SaveAsJpegToMemory));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    namespace
    {
        struct FuncSYJM
        {
            typedef uint8_t* (*FuncPtr)(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride,
                size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size);

            FuncPtr func;
            String desc;

            FuncSYJM(const FuncPtr& f, const String& d) : func(f), desc(d) {}

            void Update(int quality, SimdYuvType yuvType)
            {
                desc = desc + "[" + ToString(quality) + "]";
            }

            void Call(const View& y, const View& u, const View& v, SimdYuvType yuvType, int quality, uint8_t** data, size_t* size) const
            {
                TEST_PERFORMANCE_TEST(desc);
                *data = func(y.data, y.stride, u.data, u.stride, v.data, v.stride, y.width, y.height, yuvType, quality, size);
            }
        };
    }

#define FUNC_SYJM(func) \
    FuncSYJM(func, std::string(#func))

    bool Yuv420pSaveAsJpegToMemoryAutoTest(size_t width, size_t height, SimdYuvType yuvType, int quality, FuncSYJM f1, FuncSYJM f2)
    {
        bool result = true;

        assert(width % 2 == 0 && height % 2 == 0);

        f1.Update(quality, yuvType);
        f2.Update(quality, yuvType);

        View bgra;
        if (!GetTestImage(bgra, width, height, View::Bgra32, f1.desc, f2.desc, SimdImageFileJpeg, quality, NULL, NULL))
            return false;

        View y(width, height, View::Gray8);
        View u(width / 2, height / 2, View::Gray8);
        View v(width / 2, height / 2, View::Gray8);
        SimdBgraToYuv420pV2(bgra.data, bgra.stride, bgra.width, bgra.height, y.data, y.stride, u.data, u.stride, v.data, v.stride, yuvType);

        uint8_t* data1 = NULL, * data2 = NULL;
        size_t size1 = 0, size2 = 0;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (data1) Simd::Free(data1); f1.Call(y, u, v, yuvType, quality, &data1, &size1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (data2) SimdFree(data2); f2.Call(y, u, v, yuvType, quality, &data2, &size2));

        View dst1, dst2;
        if (dst1.Load(data1, size1, View::Bgra32) && dst2.Load(data2, size2, View::Bgra32))
        {
            int differenceMax = REAL_IMAGE.empty() ? 4 : 4;
            result = result && Compare(dst1, dst2, differenceMax, true, 64, 0, "dst1 & dst2");
            if (!result)
            {
                SaveTestImage(dst1, SimdImageFileJpeg, quality, "_1");
                SaveTestImage(dst2, SimdImageFileJpeg, quality, "_2");
                SaveTestImage(bgra, SimdImageFilePpmBin, 100, "_error");
            }
        }
        else
        {
            TEST_LOG_SS(Error, "Can't load images from memory!");
            result = false;
        }

        if (data1)
        {
            if (DebugImageSave)
            {
                const String dir = "_out";
                if (CreatePathIfNotExist(dir, false))
                {
                    String path = MakePath(dir, String("Bgra32_") + ToString(quality) + "_yuv420p.jpg");
                    FileSave(data1, size1, path.c_str());
                }
            }
            Simd::Free(data1);
        }
        if (data2)
            SimdFree(data2);

        if(DebugImageSave)
            SaveTestImage(bgra, SimdImageFileJpeg, quality, "_src");

        return result;
    }

    bool Yuv420pSaveAsJpegToMemoryAutoTest(const FuncSYJM& f1, const FuncSYJM& f2)
    {
        bool result = true;

        Ints qualities({100, 95, 85, 65, 10});
        std::vector<SimdYuvType> yuvTypes({ SimdYuvTrect871 });

        for (size_t t = 0; t < yuvTypes.size() && result; ++t)
        {
            for (size_t q = 0; q < qualities.size() && result; ++q)
            {
                result = result && Yuv420pSaveAsJpegToMemoryAutoTest(W, H, yuvTypes[t], qualities[q], f1, f2);
#if !defined(TEST_REAL_IMAGE)
                result = result && Yuv420pSaveAsJpegToMemoryAutoTest(W + E, H - E, yuvTypes[t], qualities[q], f1, f2);
#endif
            }
        }

        return result;
    }

    bool Yuv420pSaveAsJpegToMemoryAutoTest()
    {
        bool result = true;

        result = result && Yuv420pSaveAsJpegToMemoryAutoTest(FUNC_SYJM(Simd::Base::Yuv420pSaveAsJpegToMemory), FUNC_SYJM(SimdYuv420pSaveAsJpegToMemory));

#ifdef SIMD_SSE41_ENABLE
        if (Simd::Sse41::Enable)
            result = result && Yuv420pSaveAsJpegToMemoryAutoTest(FUNC_SYJM(Simd::Sse41::Yuv420pSaveAsJpegToMemory), FUNC_SYJM(SimdYuv420pSaveAsJpegToMemory));
#endif 

#ifdef SIMD_AVX2_ENABLE
        if (Simd::Avx2::Enable)
            result = result && Yuv420pSaveAsJpegToMemoryAutoTest(FUNC_SYJM(Simd::Avx2::Yuv420pSaveAsJpegToMemory), FUNC_SYJM(SimdYuv420pSaveAsJpegToMemory));
#endif 

#ifdef SIMD_AVX512BW_ENABLE
        if (Simd::Avx512bw::Enable)
            result = result && Yuv420pSaveAsJpegToMemoryAutoTest(FUNC_SYJM(Simd::Avx512bw::Yuv420pSaveAsJpegToMemory), FUNC_SYJM(SimdYuv420pSaveAsJpegToMemory));
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
        size_t size = 0;
        uint8_t* data = NULL;
        if (!GetTestImage(src, width, height, format, f1.desc, f2.desc, file, quality, &data, &size))
            return false;

        View dst1, dst2;

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (dst1.data) Simd::Free(dst1.data); f1.Call(data, size, format, dst1));

        TEST_EXECUTE_AT_LEAST_MIN_TIME(if (dst2.data) SimdFree(dst2.data); f2.Call(data, size, format, dst2));

        if (file == SimdImageFileJpeg)
        {
            int differenceMax = REAL_IMAGE.empty() ? 4 : 4;
            result = result && Compare(dst1, dst2, differenceMax, true, 64, 0, "dst1 & dst2");
            if (!result)
            {
                SaveTestImage(dst1, file, quality, "_1");
                SaveTestImage(dst2, file, quality, "_2");
                SaveTestImage(src, SimdImageFilePpmBin, 100, "_error");
            }
        }
        else
        {
            if (dst1.data && dst2.data)
                result = result && Compare(dst1, dst2, 0, true, 64, 0, "dst1 & dst2");
            if (dst1.data && SaveLoadCompatible(format, file, quality))
                result = result && Compare(dst1, src, 0, true, 64, 0, "dst1 & src");
        }

        SaveTestImage(dst1, file, quality);

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

        std::vector<View::Format> formats = { /*View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24, */View::Rgba32 };
        for (size_t format = 0; format < formats.size(); format++)
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

//#ifdef SIMD_AVX2_ENABLE
//        if (Simd::Avx2::Enable)
//            result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Avx2::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));
//#endif 
//
//#ifdef SIMD_AVX512BW_ENABLE
//        if (Simd::Avx512bw::Enable)
//            result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Avx512bw::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));
//#endif 

#ifdef SIMD_NEON_ENABLE
        if (Simd::Neon::Enable)
            result = result && ImageLoadFromMemoryAutoTest(FUNC_LM(Simd::Neon::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));
#endif 

        return result;
    }

    //-----------------------------------------------------------------------

    bool ImageLoadFromMemorySpecialTest(const String & name, View::Format format, const FuncLM& f1, const FuncLM& f2)
    {
        bool result = true;

        String path = ROOT_PATH + "/data/image/" + name;
        TEST_LOG_SS(Info, "Test " << f1.desc << " & " << f2.desc << " at " << path << " for " << ToString(format) << ".");

        size_t size = 0;
        uint8_t* data = NULL;
        if (!FileLoad(path.c_str(), &data, &size))
            return false;

        View dst1, dst2;

        f1.Call(data, size, format, dst1);

        f2.Call(data, size, format, dst2);

        int differenceMax = ToLower(ExtensionByPath(path)) == "png" ? 0 : 4;

        result = result && Compare(dst1, dst2, differenceMax, true, 64, 0, "dst1 & dst2");

        if (!result)
        {
            SimdImageFileType file = SimdImageFilePng;
            SaveTestImage(dst1, file, 100, "_1");
            SaveTestImage(dst2, file, 100, "_2");
        }

        SimdFree(data);

        return result;
    }

    bool ImageLoadFromMemorySpecialTest(const String& name, const FuncLM& f1, const FuncLM& f2)
    {
        bool result = true;

        std::vector<View::Format> formats = { View::Gray8, View::Bgr24, View::Bgra32, View::Rgb24, View::Rgba32 };
        for (size_t format = 0; format < formats.size(); format++)
            result = result && ImageLoadFromMemorySpecialTest(name, formats[format], f1, f2);

        return result;
    }

    bool ImageLoadFromMemorySpecialTest(const FuncLM& f1, const FuncLM& f2)
    {
        bool result = true;

#if 1
        result = result && ImageLoadFromMemorySpecialTest("png/basn0g01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn0g02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn0g04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn0g08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn0g16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/basn2c08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn2c16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/basn3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn3p08.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/basn4a08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn4a16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/basn6a08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basn6a16.png", f1, f2);
#endif
#if 1
        result = result && ImageLoadFromMemorySpecialTest("png/basi0g01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi0g02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi0g04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi0g08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi0g16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/basi2c08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi2c16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/basi3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi3p08.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/basi4a08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi4a16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/basi6a08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/basi6a16.png", f1, f2);
#endif
#if 1
        result = result && ImageLoadFromMemorySpecialTest("png/s01i3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s01n3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s02i3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s02n3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s03i3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s03n3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s04i3p01.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s04n3p01.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/s05i3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s05n3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s06i3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s06n3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s07i3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s07n3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s08i3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s08n3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s09i3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s09n3p02.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/s32i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s32n3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s33i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s33n3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s34i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s34n3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s35i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s35n3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s36i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s36n3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s37i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s37n3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s38i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s38n3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s39i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s39n3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s40i3p04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/s40n3p04.png", f1, f2);
#endif
#if 1
        result = result && ImageLoadFromMemorySpecialTest("png/bgai4a08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/bgai4a16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/bgan6a08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/bgan6a16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/bgbn4a08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/bggn4a16.png", f1, f2);

        result = result && ImageLoadFromMemorySpecialTest("png/bgwn6a08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/bgyn6a16.png", f1, f2);
#endif
#if 1
        result = result && ImageLoadFromMemorySpecialTest("png/tbbn0g04.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tbbn2c16.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tbbn3p08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tbgn2c16.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tbgn3p08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tbrn2c08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tbwn0g16.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tbwn3p08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tbyn3p08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tm3n3p02.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tp0n0g08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tp0n2c08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tp0n3p08.png", f1, f2);
        result = result && ImageLoadFromMemorySpecialTest("png/tp1n3p08.png", f1, f2);
#endif

        return result;
    }

    bool ImageLoadFromMemorySpecialTest()
    {
        bool result = true;

        result = result && ImageLoadFromMemorySpecialTest(FUNC_LM(Simd::Base::ImageLoadFromMemory), FUNC_LM(SimdImageLoadFromMemory));

        return result;
    }
}

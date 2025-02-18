/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdImageSave.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

#include <stdio.h>

#include <memory>
#include <sstream>

#if defined(_MSC_VER)
#pragma warning (push)
#pragma warning (disable: 4996)
#endif

namespace Simd
{ 
    SIMD_INLINE String ToLower(const String& src)
    {
        String dst(src);
        for (size_t i = 0; i < dst.size(); ++i)
        {
            if (dst[i] <= 'Z' && dst[i] >= 'A')
                dst[i] = dst[i] - ('Z' - 'z');
        }
        return dst;
    }

    SimdBool ImageSaveToFile(const ImageSaveToMemoryPtr saver, const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, const char* path)
    {
        SimdBool result = SimdFalse;
        if (file == SimdImageFileUndefined && path)
        {
            const String& str(path);
            size_t pos = str.find_last_of(".");
            if (pos != String::npos)
            {
                String ext = ToLower(str.substr(pos + 1));
                if (ext == "pgm")
                    file = SimdImageFilePgmBin;
                else if (ext == "ppm")
                    file = SimdImageFilePpmBin;
                else if (ext == "png")
                    file = SimdImageFilePng;
                else if (ext == "jpg" || ext == "jpeg")
                {
                    file = SimdImageFileJpeg;
                    if (quality == 100)
                        quality = 85;
                }
                else if (ext == "bmp")
                    file = SimdImageFileBmp;
            }
        }
        size_t size;
        uint8_t * data = saver(src, stride, width, height, format, file, quality, &size);
        if (data)
        {
            ::FILE* file = ::fopen(path, "wb");
            if (file)
            {
                if (::fwrite(data, 1, size, file) == size)
                    result = SimdTrue;
                ::fclose(file);
            }
            Simd::Free(data);
        }
        return result;
    }

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        ImageSaver* CreateImageSaver(const ImageSaverParam& param)
        {
            switch (param.file)
            {
            case SimdImageFilePgmTxt: return new ImagePgmTxtSaver(param);
            case SimdImageFilePgmBin: return new ImagePgmBinSaver(param);
            case SimdImageFilePpmTxt: return new ImagePpmTxtSaver(param);
            case SimdImageFilePpmBin: return new ImagePpmBinSaver(param);
            case SimdImageFilePng:    return new ImagePngSaver(param);
            case SimdImageFileJpeg:   return new ImageJpegSaver(param);
            case SimdImageFileBmp:   return new ImageBmpSaver(param);
            default:
                return NULL;
            }
        }

        uint8_t* ImageSaveToMemory(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, size_t* size)
        {
            ImageSaverParam param(width, height, format, file, quality);
            if (param.Validate())
            {
                Holder<ImageSaver> saver(CreateImageSaver(param));
                if (saver)
                {
                    if (saver->ToStream(src, stride))
                        return saver->Release(size);
                }
            }
            return NULL;
        }
    }
}

#if defined(_MSC_VER)
#pragma warning (pop)
#endif

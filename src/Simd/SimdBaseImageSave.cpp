/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdImageSave.h"
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
    SimdBool ImageSaveToFile(const ImageSaveToMemoryPtr saver, const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, const char* path)
    {
        SimdBool result = SimdFalse;
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

    namespace Base
    {
        //---------------------------------------------------------------------

        ImagePgmTxtSaver::ImagePgmTxtSaver(const ImageSaverParam& param)
            : ImageSaver(param)
        {
        }

        bool ImagePgmTxtSaver::ToStream(const uint8_t* src, size_t stride)
        {
            std::stringstream hs;
            hs << "P5\n" << _param.width << " " << _param.height << "\n255\n";
            _stream.Write(hs.str().c_str(), hs.str().size());
            if (_param.format != SimdPixelFormatGray8)
                _buffer.Resize(_param.width);
            for (size_t row = 0; row < _param.height; ++row)
            {
                const uint8_t* tmp = src;
                if (_param.format != SimdPixelFormatGray8)
                {
                    if (_param.format == SimdPixelFormatBgr24)
                        BgrToGray(src, _param.width, 1, stride, _buffer.data, _param.width);
                    else if (_param.format == SimdPixelFormatBgra32)
                        BgraToGray(src, _param.width, 1, stride, _buffer.data, _param.width);
                    else
                        assert(0);
                    tmp = _buffer.data;
                }
                std::stringstream rs;
                for (size_t col = 0; col < _param.width; ++col)
                {
                    int val = tmp[col];
                    if (val < 100)
                        rs << " ";
                    if (val < 10)
                        rs << " ";
                    rs << val << " ";
                    if (rs.str().size() >= 64 || col == _param.width - 1)
                    {
                        rs << "\n";
                        _stream.Write(rs.str().c_str(), rs.str().size());
                        rs = std::stringstream();
                    }
                }
                src += stride;
            }
            return true;
        }

        //---------------------------------------------------------------------

        ImageSaver* CreateImageSaver(const ImageSaverParam& param)
        {
            switch (param.file)
            {
            case SimdImageFilePgmTxt: return new ImagePgmTxtSaver(param);
            case SimdImageFilePgmBin: return NULL;
            case SimdImageFilePpmTxt: return NULL;
            case SimdImageFilePpmBin: return NULL;
            default:
                return NULL;
            }
        }

        uint8_t* ImageSaveToMemory(const uint8_t* src, size_t stride, size_t width, size_t height, SimdPixelFormatType format, SimdImageFileType file, int quality, size_t* size)
        {
            ImageSaverParam param(width, height, format, file, quality);
            if (param.Validate())
            {
                std::unique_ptr<ImageSaver> saver(CreateImageSaver(param));
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

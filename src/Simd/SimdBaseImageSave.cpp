/*
* Simd Library (http://ermig1979.github.io/Simd).
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

    //-------------------------------------------------------------------------

    namespace Base
    {
        ImagePxmSaver::ImagePxmSaver(const ImageSaverParam& param)
            : ImageSaver(param)
            , _convert(NULL)
        {
            _block = _param.height;
            if (_param.file == SimdImageFilePgmTxt || _param.file == SimdImageFilePgmBin)
            {
                _size = _param.width * 1;
                if (_param.format != SimdPixelFormatGray8)
                {
                    _block = Simd::RestrictRange<size_t>(Base::AlgCacheL1() / _size, 1, _param.height);
                    _buffer.Resize(_block * _size);
                }
            }
            else if (_param.file == SimdImageFilePpmTxt || _param.file == SimdImageFilePpmBin)
            {
                _size = _param.width * 3;
                if (_param.format != SimdPixelFormatRgb24)
                {
                    _block = Simd::RestrictRange<size_t>(Base::AlgCacheL1() / _size, 1, _param.height);
                    _buffer.Resize(_block * _size);
                }
            }
            else
                assert(0);
        }

        void ImagePxmSaver::WriteHeader(size_t version)
        {
            std::stringstream header;
            header << "P" << version << "\n" << _param.width << " " << _param.height << "\n255\n";
            _stream.Write(header.str().c_str(), header.str().size());
        }

        uint8_t g_pxmPrint[256][4];
        bool PxmPrintInit()
        {
            for (int i = 0; i < 256; ++i)
            {
                int d0 = i / 100;
                int d1 = (i / 10) % 10;
                int d2 = i % 10;
                g_pxmPrint[i][0] = d0 ? '0' + d0 : ' ';
                g_pxmPrint[i][1] = (d1 || d0) ? '0' + d1 : ' ';
                g_pxmPrint[i][2] = '0' + d2;
                g_pxmPrint[i][3] = ' ';
            }
            return true;
        }
        bool g_pxmPrintInited = PxmPrintInit();

        //---------------------------------------------------------------------

        ImagePgmTxtSaver::ImagePgmTxtSaver(const ImageSaverParam& param)
            : ImagePxmSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatBgr24: _convert = Base::BgrToGray; break;
            case SimdPixelFormatBgra32: _convert = Base::BgraToGray; break;
            case SimdPixelFormatRgb24: _convert = Base::RgbToGray; break;
            case SimdPixelFormatRgba32: _convert = Base::RgbaToGray; break;
            default: break;
            }
        }

        bool ImagePgmTxtSaver::ToStream(const uint8_t* src, size_t stride)
        {
            size_t grayStride = _param.format == SimdPixelFormatGray8 ? stride : _size;
            _stream.Reserve(32 + _param.height * (_param.width * 4 + DivHi(_param.width, 17)));
            WriteHeader(2);
            for (size_t row = 0; row < _param.height;)
            {
                size_t block = Simd::Min(row + _block, _param.height) - row;
                const uint8_t* gray = src;
                if (_param.format != SimdPixelFormatGray8)
                {
                    _convert(src, _param.width, block, stride, _buffer.data, grayStride);
                    gray = _buffer.data;
                }                
                for (size_t b = 0; b < block; ++b)
                {
                    uint8_t string[70];
                    for (size_t col = 0, offset = 0; col < _param.width; ++col)
                    {
                        *(uint32_t*)(string + offset) = *(uint32_t*)g_pxmPrint[gray[col]];
                        offset += 4;
                        if (offset >= 68 || col == _param.width - 1)
                        {
                            string[offset++] = '\n';
                            _stream.Write(string, offset);
                            offset = 0;
                        }
                    }
                    gray += grayStride;
                }
                src += stride * block;
                row += block;
            }
            return true;
        }

        //---------------------------------------------------------------------

        ImagePgmBinSaver::ImagePgmBinSaver(const ImageSaverParam& param)
            : ImagePxmSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatBgr24: _convert = Base::BgrToGray; break;
            case SimdPixelFormatBgra32: _convert = Base::BgraToGray; break;
            case SimdPixelFormatRgb24: _convert = Base::RgbToGray; break;
            case SimdPixelFormatRgba32: _convert = Base::RgbaToGray; break;
            default: break;
            }
        }

        bool ImagePgmBinSaver::ToStream(const uint8_t* src, size_t stride)
        {
            size_t grayStride = _param.format == SimdPixelFormatGray8 ? stride : _size;
            _stream.Reserve(32 + _param.height * _size);
            WriteHeader(5);
            for (size_t row = 0; row < _param.height;)
            {
                size_t block = Simd::Min(row + _block, _param.height) - row;
                const uint8_t* gray = src;
                if (_param.format != SimdPixelFormatGray8)
                {
                    _convert(src, _param.width, block, stride, _buffer.data, grayStride);
                    gray = _buffer.data;
                }
                for (size_t b = 0; b < block; ++b)
                {
                    _stream.Write(gray, _size);
                    gray += grayStride;
                }
                src += stride * block;
                row += block;
            }
            return true;
        }

        //---------------------------------------------------------------------

        ImagePpmTxtSaver::ImagePpmTxtSaver(const ImageSaverParam& param)
            : ImagePxmSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _convert = Base::GrayToBgr; break;
            case SimdPixelFormatBgr24: _convert = Base::BgrToRgb; break;
            case SimdPixelFormatBgra32: _convert = Base::BgraToRgb; break;
            case SimdPixelFormatRgba32: _convert = Base::BgraToBgr; break;
            default: break;
            }
        }

        bool ImagePpmTxtSaver::ToStream(const uint8_t* src, size_t stride)
        {
            size_t rgbStride = _param.format == SimdPixelFormatRgb24 ? stride : _size;
            _stream.Reserve(32 + _param.height * (_param.width * 13 + DivHi(_param.width, 5)));
            WriteHeader(3);
            for (size_t row = 0; row < _param.height;)
            {
                size_t block = Simd::Min(row + _block, _param.height) - row;
                const uint8_t* rgb = src;
                if (_param.format != SimdPixelFormatRgb24)
                {
                    _convert(src, _param.width, block, stride, _buffer.data, rgbStride);
                    rgb = _buffer.data;
                }
                for (size_t b = 0; b < block; ++b)
                {
                    uint8_t string[70];
                    for (size_t col = 0, offset = 0; col < _size; col += 3)
                    {
                        ((uint32_t*)(string + offset))[0] = *(uint32_t*)g_pxmPrint[rgb[col + 0]];
                        ((uint32_t*)(string + offset))[1] = *(uint32_t*)g_pxmPrint[rgb[col + 1]];
                        ((uint32_t*)(string + offset))[2] = *(uint32_t*)g_pxmPrint[rgb[col + 2]];
                        offset += 12;
                        if (offset >= 68 || col == _size - 3)
                        {
                            string[offset++] = '\n';
                            _stream.Write(string, offset);
                            offset = 0;
                        }
                        else
                        {
                            string[offset++] = ' ';
                            string[offset++] = ' ';
                        }
                    }
                    rgb += rgbStride;
                }
                src += stride * block;
                row += block;
            }
            return true;
        }

        //---------------------------------------------------------------------

        ImagePpmBinSaver::ImagePpmBinSaver(const ImageSaverParam& param)
            : ImagePxmSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _convert = Base::GrayToBgr; break;
            case SimdPixelFormatBgr24: _convert = Base::BgrToRgb; break;
            case SimdPixelFormatBgra32: _convert = Base::BgraToRgb; break;
            case SimdPixelFormatRgba32: _convert = Base::BgraToBgr; break;
            default: break;
            }
        }

        bool ImagePpmBinSaver::ToStream(const uint8_t* src, size_t stride)
        {
            size_t rgbStride = _param.format == SimdPixelFormatRgb24 ? stride : _size;
            _stream.Reserve(32 + _param.height * _size);
            WriteHeader(6);
            for (size_t row = 0; row < _param.height;)
            {
                size_t block = Simd::Min(row + _block, _param.height) - row;
                const uint8_t* rgb = src;
                if (_param.format != SimdPixelFormatRgb24)
                {
                    _convert(src, _param.width, block, stride, _buffer.data, rgbStride);
                    rgb = _buffer.data;
                }
                for (size_t b = 0; b < block; ++b)
                {
                    _stream.Write(rgb, _size);
                    rgb += rgbStride;
                }
                src += stride * block;
                row += block;
            }
            return true;
        }

        //---------------------------------------------------------------------

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

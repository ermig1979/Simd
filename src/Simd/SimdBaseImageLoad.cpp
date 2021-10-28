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
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

#include <stdio.h>

#if defined(_MSC_VER)
#pragma warning (push)
#pragma warning (disable: 4996)
#endif

namespace Simd
{
    uint8_t* ImageLoadFromFile(const ImageLoadFromMemoryPtr loader, const char* path, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format)
    {
        uint8_t* data = NULL;
        ::FILE* file = ::fopen(path, "rb");
        if (file)
        {
            ::fseek(file, 0, SEEK_END);
            Array8u buffer(::ftell(file));
            ::fseek(file, 0, SEEK_SET);
            if (::fread(buffer.data, 1, buffer.size, file) == buffer.size)
                data = loader(buffer.data, buffer.size, stride, width, height, format);
            ::fclose(file);
        }
        return data;
    }

    //-------------------------------------------------------------------------

    ImageLoaderParam::ImageLoaderParam(const uint8_t* d, size_t s, SimdPixelFormatType f)
        : data(d)
        , size(s)
        , format(f)
        , file(SimdImageFileUndefined)
    {
    }

    bool ImageLoaderParam::Validate()
    {
        if (size >= 3)
        {
            if (data[0] == 'P' && data[2] == '\n')
            {
                if (data[1] == '2')
                    file = SimdImageFilePgmTxt;
                if (data[1] == '3')
                    file = SimdImageFilePpmTxt;
                if (data[1] == '5')
                    file = SimdImageFilePgmBin;
                if (data[1] == '6')
                    file = SimdImageFilePpmBin;
            }
        }
        if (size >= 8)
        {
            const uint8_t SIGNATURE[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
            if(memcmp(data, SIGNATURE, 8) == 0)
                file = SimdImageFilePng;
        }
        if (size >= 2)
        {
            if (data[0] == 0xFF && data[1] == 0xD8)
                file = SimdImageFileJpeg;
        }
        return
            file != SimdImageFileUndefined && 
                (format == SimdPixelFormatNone || format == SimdPixelFormatGray8 || 
                format == SimdPixelFormatBgr24 || format == SimdPixelFormatBgra32 || 
                format == SimdPixelFormatRgb24 || format == SimdPixelFormatRgba32);
    }
        
    namespace Base
    {
        ImagePxmLoader::ImagePxmLoader(const ImageLoaderParam& param)
            : ImageLoader(param)
            , _toAny(NULL)
            , _toBgra(NULL)
        {
        }

        bool ImagePxmLoader::ReadHeader(size_t version)
        {
            if (_stream.Size() < 3 ||
                _stream.Data()[0] != 'P' ||
                _stream.Data()[1] != '0' + version ||
                _stream.Data()[2] != '\n')
                return false;
            _stream.Seek(3);
            uint32_t width, height, max;
            if (!(_stream.ReadUnsigned(width) && _stream.ReadUnsigned(height) && _stream.ReadUnsigned(max)))
                return false;
            if (!(width > 0 && height > 0 && max == 255))
                return false;
            uint8_t byte;
            if (!(_stream.Read(byte) && byte == '\n'))
                return false;
            _image.Recreate(width, height, (Image::Format)_param.format);
            _block = height;
            if (_param.file == SimdImageFilePgmTxt || _param.file == SimdImageFilePgmBin)
            {
                _size = width * 1;
                if (_param.format != SimdPixelFormatGray8)
                {
                    _block = Simd::RestrictRange<size_t>(Base::AlgCacheL1() / _size, 1, height);
                    _buffer.Resize(_block * _size);
                }
            }
            else if (_param.file == SimdImageFilePpmTxt || _param.file == SimdImageFilePpmBin)
            {
                _size = width * 3;
                if (_param.format != SimdPixelFormatRgb24)
                {
                    _block = Simd::RestrictRange<size_t>(Base::AlgCacheL1() / _size, 1, height);
                    _buffer.Resize(_block * _size);
                }
            }
            else
                return false;
            SetConverters();
            return true;
        }

        //-------------------------------------------------------------------------

        ImagePgmTxtLoader::ImagePgmTxtLoader(const ImageLoaderParam& param)
            : ImagePxmLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatGray8;
        }

        bool ImagePgmTxtLoader::FromStream()
        {
            if (!ReadHeader(2))
                return false;
            size_t grayStride = _param.format == SimdPixelFormatGray8 ? _image.stride : _size;
            for (size_t row = 0; row < _image.height;)
            {
                size_t block = Simd::Min(row + _block, _image.height) - row;
                uint8_t * gray = _param.format == SimdPixelFormatGray8 ? _image.Row<uint8_t>(row) : _buffer.data;
                for (size_t b = 0; b < block; ++b)
                {
                    for (size_t i = 0; i < _size; ++i)
                    {
                        if (!_stream.ReadUnsigned(gray[i]))
                            return false;
                    }
                    gray += grayStride;
                }
                if(_param.format == SimdPixelFormatBgr24 || _param.format == SimdPixelFormatRgb24)
                    _toAny(_buffer.data, _image.width, block, _size, _image.Row<uint8_t>(row), _image.stride);
                if (_param.format == SimdPixelFormatBgra32 || _param.format == SimdPixelFormatRgba32)
                    _toBgra(_buffer.data, _image.width, block, _size, _image.Row<uint8_t>(row), _image.stride, 0xFF);
                row += block;
            }
            return true;
        }

        void ImagePgmTxtLoader::SetConverters()
        {
            switch (_param.format)
            {
            case SimdPixelFormatBgr24: _toAny = Base::GrayToBgr; break;
            case SimdPixelFormatBgra32: _toBgra = Base::GrayToBgra; break;
            case SimdPixelFormatRgb24: _toAny = Base::GrayToBgr; break;
            case SimdPixelFormatRgba32: _toBgra = Base::GrayToBgra; break;
            default: break;
            }
        }

        //-------------------------------------------------------------------------

        ImagePgmBinLoader::ImagePgmBinLoader(const ImageLoaderParam& param)
            : ImagePxmLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatGray8;
        }

        bool ImagePgmBinLoader::FromStream()
        {
            if (!ReadHeader(5))
                return false;
            size_t grayStride = _param.format == SimdPixelFormatGray8 ? _image.stride : _size;
            for (size_t row = 0; row < _image.height;)
            {
                size_t block = Simd::Min(row + _block, _image.height) - row;
                uint8_t* gray = _param.format == SimdPixelFormatGray8 ? _image.Row<uint8_t>(row) : _buffer.data;
                for (size_t b = 0; b < block; ++b)
                {
                    if (_stream.Read(_size, gray) != _size)
                        return false;
                    gray += grayStride;
                }
                if (_param.format == SimdPixelFormatBgr24 || _param.format == SimdPixelFormatRgb24)
                    _toAny(_buffer.data, _image.width, block, _size, _image.Row<uint8_t>(row), _image.stride);
                if (_param.format == SimdPixelFormatBgra32 || _param.format == SimdPixelFormatRgba32)
                    _toBgra(_buffer.data, _image.width, block, _size, _image.Row<uint8_t>(row), _image.stride, 0xFF);
                row += block;
            }
            return true;
        }

        void ImagePgmBinLoader::SetConverters()
        {
            switch (_param.format)
            {
            case SimdPixelFormatBgr24: _toAny = Base::GrayToBgr; break;
            case SimdPixelFormatBgra32: _toBgra = Base::GrayToBgra; break;
            case SimdPixelFormatRgb24: _toAny = Base::GrayToBgr; break;
            case SimdPixelFormatRgba32: _toBgra = Base::GrayToBgra; break;
            default: break;
            }
        }

        //-------------------------------------------------------------------------

        ImagePpmTxtLoader::ImagePpmTxtLoader(const ImageLoaderParam& param)
            : ImagePxmLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgb24;
        }

        bool ImagePpmTxtLoader::FromStream()
        {
            if (!ReadHeader(3))
                return false;
            size_t rgbStride = _param.format == SimdPixelFormatRgb24 ? _image.stride : _size;
            for (size_t row = 0; row < _image.height;)
            {
                size_t block = Simd::Min(row + _block, _image.height) - row;
                uint8_t* rgb = _param.format == SimdPixelFormatRgb24 ? _image.Row<uint8_t>(row) : _buffer.data;
                for (size_t b = 0; b < block; ++b)
                {
                    for (size_t i = 0; i < _size; ++i)
                    {
                        if (!_stream.ReadUnsigned(rgb[i]))
                            return false;
                    }
                    rgb += rgbStride;
                }
                if (_param.format == SimdPixelFormatGray8 || _param.format == SimdPixelFormatBgr24)
                    _toAny(_buffer.data, _image.width, block, _size, _image.Row<uint8_t>(row), _image.stride);
                if (_param.format == SimdPixelFormatBgra32 || _param.format == SimdPixelFormatRgba32)
                    _toBgra(_buffer.data, _image.width, block, _size, _image.Row<uint8_t>(row), _image.stride, 0xFF);
                row += block;
            }
            return true;
        }

        void ImagePpmTxtLoader::SetConverters()
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _toAny = Base::RgbToGray; break;
            case SimdPixelFormatBgr24: _toAny = Base::BgrToRgb; break;
            case SimdPixelFormatBgra32: _toBgra = Base::RgbToBgra; break;
            case SimdPixelFormatRgba32: _toBgra = Base::BgrToBgra; break;
            default: break;
            }
        }

        //-------------------------------------------------------------------------

        ImagePpmBinLoader::ImagePpmBinLoader(const ImageLoaderParam& param)
            : ImagePxmLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgb24;
        }

        bool ImagePpmBinLoader::FromStream()
        {
            if (!ReadHeader(6))
                return false;
            size_t rgbStride = _param.format == SimdPixelFormatRgb24 ? _image.stride : _size;
            for (size_t row = 0; row < _image.height;)
            {
                size_t block = Simd::Min(row + _block, _image.height) - row;
                uint8_t* rgb = _param.format == SimdPixelFormatRgb24 ? _image.Row<uint8_t>(row) : _buffer.data;
                for (size_t b = 0; b < block; ++b)
                {
                    if (_stream.Read(_size, rgb) != _size)
                        return false;
                    rgb += rgbStride;
                }
                if (_param.format == SimdPixelFormatGray8 || _param.format == SimdPixelFormatBgr24)
                    _toAny(_buffer.data, _image.width, block, _size, _image.Row<uint8_t>(row), _image.stride);
                if (_param.format == SimdPixelFormatBgra32 || _param.format == SimdPixelFormatRgba32)
                    _toBgra(_buffer.data, _image.width, block, _size, _image.Row<uint8_t>(row), _image.stride, 0xFF);
                row += block;
            }
            return true;
        }

        void ImagePpmBinLoader::SetConverters()
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _toAny = Base::RgbToGray; break;
            case SimdPixelFormatBgr24: _toAny = Base::BgrToRgb; break;
            case SimdPixelFormatBgra32: _toBgra = Base::RgbToBgra; break;
            case SimdPixelFormatRgba32: _toBgra = Base::BgrToBgra; break;
            default: break;
            }
        }

        //-------------------------------------------------------------------------

        ImageLoader* CreateImageLoader(const ImageLoaderParam& param)
        {
            switch (param.file)
            {
            case SimdImageFilePgmTxt: return new ImagePgmTxtLoader(param);
            case SimdImageFilePgmBin: return new ImagePgmBinLoader(param);
            case SimdImageFilePpmTxt: return new ImagePpmTxtLoader(param);
            case SimdImageFilePpmBin: return new ImagePpmBinLoader(param);
            case SimdImageFilePng: return new ImagePngLoader(param);
            case SimdImageFileJpeg: return new ImageJpegLoader(param);
            default:
                return NULL;
            }
        }

        uint8_t* ImageLoadFromMemory(const uint8_t* data, size_t size, size_t* stride, size_t* width, size_t* height, SimdPixelFormatType* format)
        {
            ImageLoaderParam param(data, size, *format);
            if (param.Validate())
            {
                Holder<ImageLoader> loader(CreateImageLoader(param));
                if (loader)
                {
                    if (loader->FromStream())
                        return loader->Release(stride, width, height, format);
                }
            }
            return NULL;
        }
    }
}

#if defined(_MSC_VER)
#pragma warning (pop)
#endif

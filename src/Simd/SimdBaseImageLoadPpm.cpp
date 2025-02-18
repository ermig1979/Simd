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

namespace Simd
{
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

        //-------------------------------------------------------------------------------------------------

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

        //-------------------------------------------------------------------------------------------------

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

        //-------------------------------------------------------------------------------------------------

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

        //-------------------------------------------------------------------------------------------------

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
    }
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
        ImageBmpLoader::ImageBmpLoader(const ImageLoaderParam& param)
            : ImageLoader(param)
            , _toAny(NULL)
            , _toBgra(NULL)
        {

        }

        ImageBmpLoader::~ImageBmpLoader()
        {

        }

        bool ImageBmpLoader::FromStream()
        {
            if (!ParseHeader())
                return false;

            SetConverters();

            _image.Recreate(_width, _height, (Image::Format)_param.format);

            uint8_t* dst = _image.data + (_image.height - 1) * _image.stride;
            for (size_t row = 0; row < _image.height; ++row)
            {
                if (!_stream.CanRead(_size + _pad))
                    return false;
                const uint8_t* src = _stream.Current();
                if (_toAny)
                    _toAny(src, _image.width, 1, _size, dst, _image.stride);
                else if (_toBgra)
                    _toBgra(src, _image.width, 1, _size, dst, _image.stride, 0xFF);
                else
                    memcpy(dst, src, _size);
                if (!_stream.Skip(_size + _pad))
                    return false;
                dst -= _image.stride;
            }

            return true;
        }

        bool ImageBmpLoader::ParseHeader()
        {
            if (_stream.Get8u() != 'B' || _stream.Get8u() != 'M')
                return false;
            if (!_stream.Skip(12))
                return false;
            uint32_t headerSize = 0;
            if (!_stream.Read32u(headerSize) || !(headerSize == 12 || headerSize == 40 || headerSize == 56 || headerSize == 108 || headerSize == 124))
                return false;
            if (headerSize == 12)
            {
                uint16_t w, h;
                if (!_stream.Read16u(w) || !_stream.Read16u(h) || w * h == 0)
                    return false;
                _width = w;
                _height = h;
            }
            else 
            {
                if (!_stream.Read32u(_width) || !_stream.Read32u(_height) || _width * _height == 0)
                    return false;
            }
            uint16_t planes, bpp;
            if (!_stream.Read16u(planes) || !_stream.Read16u(bpp) || planes != 1)
                return false;
            _bpp = bpp;
            if (headerSize != 12)
            {
                uint32_t compress;
                if (!_stream.Read32u(compress) || !(compress == 0 || (compress == 3 && (_bpp == 8 || _bpp == 16 || _bpp == 32))))
                    return false;
                if (!_stream.Skip(20))
                    return false;
                if (headerSize == 40 || headerSize == 56)
                {
                    if (headerSize == 56)
                    {
                        if (!_stream.Skip(16))
                            return false;
                    }
                }
                else
                {
                    if (!_stream.Read32u(_mr) || !_stream.Read32u(_mg) || !_stream.Read32u(_mb) || !_stream.Read32u(_ma))
                        return false;
                    if (headerSize == 108)
                    {
                        if (!_stream.Skip(52))
                            return false;
                    }
                    else
                    {
                        if (!_stream.Skip(68))
                            return false;
                    }

                }
                if (compress == 0)
                {
                    if (_bpp == 8)
                    {
                        for (int i = 0; i < 256; ++i)
                        {
                            uint32_t color;
                            if(!_stream.Read32u(color) || color != (i | (i << 8) | (i << 16)))
                            {
                                return false;
                            }
                        }
                    }
                    else if (_bpp == 16)
                    {
                        _mr = 31u << 10;
                        _mg = 31u << 5;
                        _mb = 31u << 0;
                        _ma = 0;
                    }
                    else if (_bpp == 32)
                    {
                        _mr = 0xffu << 16;
                        _mg = 0xffu << 8;
                        _mb = 0xffu << 0;
                        _ma = 0xffu << 24;
                    }
                    else
                    {
                        _mr = 0;
                        _mg = 0;
                        _mb = 0;
                        _ma = 0;
                    }
                }

            }
            if (_bpp < 8)
                return false;  
            if (_bpp == 8)
            {
                _size = _width;
                _pad = (int32_t)AlignHi(_size, 4) - _size;
            }
            else if (_bpp == 24)
            {
                _size = _width * 3;
                _pad = (int32_t)AlignHi(_size, 4) - _size;
            }
            else
            {
                _size = _width * 4;
                _pad = 0;
            }
            if (_param.format == SimdPixelFormatNone)
            {
                if (_bpp == 32)
                    _param.format = SimdPixelFormatBgra32;
                else if (_bpp == 24)
                    _param.format = SimdPixelFormatBgr24;
                else if (_bpp == 8)
                    _param.format = SimdPixelFormatGray8;
            }
            return true;
        }

        void ImageBmpLoader::SetConverters()
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _toAny = (_bpp == 32 ? Base::BgraToGray : (_bpp == 24 ? Base::BgrToGray : NULL)); break;
            case SimdPixelFormatBgr24: _toAny = (_bpp == 32 ? (ToAnyPtr)Base::BgraToBgr : NULL); break;
            case SimdPixelFormatRgb24: _toAny = (_bpp == 32 ? Base::BgraToRgb : Base::BgrToRgb); break;
            case SimdPixelFormatBgra32: _toBgra = (_bpp == 32 ? NULL : (ToBgraPtr)Base::BgrToBgra); break;
            case SimdPixelFormatRgba32: 
                if (_bpp == 32)
                    _toAny = Base::BgraToRgba;
                else 
                    _toBgra = (ToBgraPtr)Base::RgbToBgra; 
                break;
            default: break;
            }
        }
    }
}


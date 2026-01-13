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

namespace Simd
{ 
    namespace Base
    {
        ImageBmpSaver::ImageBmpSaver(const ImageSaverParam& param)
            : ImageSaver(param)
            , _convert(NULL)
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _convert = NULL; _pixel = 1; break;
            case SimdPixelFormatBgr24: _pixel = 3; break;
            case SimdPixelFormatBgra32: _pixel = 4; break;
            case SimdPixelFormatRgb24: _convert = Base::BgrToRgb; _pixel = 3; break;
            case SimdPixelFormatRgba32: _convert = Base::BgraToRgba; _pixel = 4; break;
            default: break;
            }
            _size = _param.width * _pixel;
            _pad = AlignHi(_size, 4) - _size;
        }

        bool ImageBmpSaver::ToStream(const uint8_t* src, size_t stride)
        {
            size_t reserve = 14 + 40 + _param.height * (_size + _pad);
            if(_pixel == 1)
            {
                reserve += 1024;
            }
            else if (_pixel == 4)
            {
                reserve += 68;
            }
            _stream.Reserve(reserve);

            WriteHeader();

            uint32_t zero = 0;
            src += (_param.height - 1) * stride;
            for (size_t row = 0; row < _param.height; ++ row)
            {
                uint8_t* dst = _stream.Current();
                _stream.Seek(_stream.Pos() + _size);
                if (_convert)
                    _convert(src, _param.width, 1, stride, dst, _size + _pad);
                else
                    memcpy(dst, src, _size);
                if (_pad)
                    _stream.Write(&zero, _pad);
                src -= stride;
            }
            return true;
        }

        void ImageBmpSaver::WriteHeader()
        {
            _stream.Write8u('B');
            _stream.Write8u('M');
            if (_pixel == 4)
            {
                uint32_t data[] = {
                    uint32_t(14 + 108 + (_size + _pad) * _param.height),
                    0, 14 + 108, 108,
                    uint32_t(_param.width), uint32_t(_param.height),
                    0x00200001, 3, 0, 0, 0, 0, 0,
                    0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                _stream.Write(data, sizeof(data));
            }
            else if (_pixel == 3)
            {
                uint32_t data[] = { 
                    uint32_t(14 + 40 + (_size + _pad) * _param.height),
                    0, 14 + 40, 40,
                    uint32_t(_param.width), uint32_t(_param.height), 
                    0x00180001, 0, 0, 0, 0, 0, 0 };
                _stream.Write(data, sizeof(data));
            }
            else
            {
                uint32_t data[] = { 
                    uint32_t(14 + 40 + 1024 + (_size + _pad) * _param.height),
                    0, 14 + 40 + 1024, 40,
                    uint32_t(_param.width), uint32_t(_param.height), 
                    0x00080001, 0, 0, 0, 0, 0, 0 };
                _stream.Write(data, sizeof(data));
                for(int i = 0; i < 256; ++i)
                {
                    uint32_t color = i | (i << 8) | (i << 16);
                    _stream.Write(&color, sizeof(color));
                }
            }
        }
    }
}

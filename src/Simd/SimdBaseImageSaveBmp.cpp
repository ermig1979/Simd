/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdMemory.h"
#include "Simd/SimdImageSave.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{ 
    namespace Base
    {
        typedef struct
        {
            unsigned char buffer[64];
            int buf_used;
            OutputMemoryStream *stream;
        } stbi__write_context;

        typedef unsigned int stbiw_uint32;

        static void stbiw__write_flush(stbi__write_context* s)
        {
            if (s->buf_used) {
                s->stream->Write(&s->buffer, s->buf_used);
                s->buf_used = 0;
            }
        }

        static void stbiw__write1(stbi__write_context* s, unsigned char a)
        {
            if ((size_t)s->buf_used + 1 > sizeof(s->buffer))
                stbiw__write_flush(s);
            s->buffer[s->buf_used++] = a;
        }

        static void stbiw__write3(stbi__write_context* s, unsigned char a, unsigned char b, unsigned char c)
        {
            int n;
            if ((size_t)s->buf_used + 3 > sizeof(s->buffer))
                stbiw__write_flush(s);
            n = s->buf_used;
            s->buf_used = n + 3;
            s->buffer[n + 0] = a;
            s->buffer[n + 1] = b;
            s->buffer[n + 2] = c;
        }

        static void stbiw__write_pixel(stbi__write_context* s, int rgb_dir, int comp, int write_alpha, int expand_mono, unsigned char* d)
        {
            unsigned char bg[3] = { 255, 0, 255 }, px[3];
            int k;

            if (write_alpha < 0)
                stbiw__write1(s, d[comp - 1]);

            switch (comp) {
            case 2: // 2 pixels = mono + alpha, alpha is written separately, so same as 1-channel case
            case 1:
                if (expand_mono)
                    stbiw__write3(s, d[0], d[0], d[0]); // monochrome bmp
                else
                    stbiw__write1(s, d[0]);  // monochrome TGA
                break;
            case 4:
                if (!write_alpha) {
                    // composite against pink background
                    for (k = 0; k < 3; ++k)
                        px[k] = bg[k] + ((d[k] - bg[k]) * d[3]) / 255;
                    stbiw__write3(s, px[1 - rgb_dir], px[1], px[1 + rgb_dir]);
                    break;
                }
                /* FALLTHROUGH */
            case 3:
                stbiw__write3(s, d[1 - rgb_dir], d[1], d[1 + rgb_dir]);
                break;
            }
            if (write_alpha > 0)
                stbiw__write1(s, d[comp - 1]);
        }

        static void stbiw__write_pixels(stbi__write_context* s, int rgb_dir, int vdir, int x, int y, int stride, int comp, void* data, int write_alpha, int scanline_pad, int expand_mono)
        {
            stbiw_uint32 zero = 0;
            int i, j, j_end;

            if (y <= 0)
                return;

            if (vdir < 0) {
                j_end = -1; j = y - 1;
            }
            else {
                j_end = y; j = 0;
            }

            for (; j != j_end; j += vdir) 
            {
                for (i = 0; i < x; ++i) 
                {
                    unsigned char* d = (unsigned char*)data + j * stride + i * comp;
                    stbiw__write_pixel(s, rgb_dir, comp, write_alpha, expand_mono, d);
                }
                stbiw__write_flush(s);
                s->stream->Write(&zero, scanline_pad);
            }
        }

        static int stbiw__outfile(stbi__write_context* s, int rgb_dir, int vdir, int x, int y, int stride, int comp, int expand_mono, void* data, int alpha, int pad)
        {
            if (y < 0 || x < 0) 
            {
                return 0;
            }
            else 
            {
                stbiw__write_pixels(s, rgb_dir, vdir, x, y, stride, comp, data, alpha, pad, expand_mono);
                return 1;
            }
        }

        static int stbi_write_bmp_core(stbi__write_context* s, int x, int y, int stride, int comp, const void* data)
        {
            if (comp != 4) 
            {
                int pad = (-x * 3) & 3;
                return stbiw__outfile(s, -1, -1, x, y, stride, comp, 1, (void*)data, 0, pad);
            }
            else 
            {
                return stbiw__outfile(s, -1, -1, x, y, stride, comp, 1, (void*)data, 1, 0);
            }
        }

        //--------------------------------------------------------------------------------------------------

        ImageBmpSaver::ImageBmpSaver(const ImageSaverParam& param)
            : ImageSaver(param)
            , _convert(NULL)
        {
            _block = _param.height;
            switch (_param.format)
            {
            case SimdPixelFormatGray8: _pixel = 1; break;
            case SimdPixelFormatBgr24: _convert = Base::BgrToRgb; _pixel = 3; break;
            case SimdPixelFormatBgra32: _convert = Base::BgraToRgba; _pixel = 4; break;
            case SimdPixelFormatRgb24: _pixel = 3; break;
            case SimdPixelFormatRgba32: _pixel = 4; break;
            default: break;
            }
            _size = _param.width * _pixel;
        }

        bool ImageBmpSaver::ToStream(const uint8_t* src, size_t stride)
        {
            if (_convert)
                _buffer.Resize(_block * _size);
            size_t bufStride = _convert ? _size : stride;
            _stream.Reserve(256 + _param.height * AlignHi(_size, 4));

            stbi__write_context s = { 0 };
            s.stream = &_stream;

            WriteHeader();

            for (size_t row = 0; row < _param.height;)
            {
                size_t block = Simd::Min(row + _block, _param.height) - row;
                const uint8_t* buf = src;
                if (_convert)
                {
                    _convert(src, _param.width, block, stride, _buffer.data, bufStride);
                    buf = _buffer.data;
                } 

                if (!stbi_write_bmp_core(&s, _param.width, _param.height, bufStride, _pixel, buf))
                    return false;

                src += stride * block;
                row += block;
            }
            return true;
        }

        bool ImageBmpSaver::WriteHeader()
        {
            _stream.Write8u('B');
            _stream.Write8u('M');
            if (_param.format == SimdPixelFormatBgra32 || _param.format == SimdPixelFormatRgba32)
            {
                uint32_t data[] = {
                    uint32_t(14 + 108 + AlignHi(_size, 4) * _param.height),
                    0, 14 + 108, 108,
                    uint32_t(_param.width), uint32_t(_param.height),
                    0x00200001, 3, 0, 0, 0, 0, 0,
                    0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000,
                    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
                _stream.Write(data, sizeof(data));
            }
            else
            {
                uint32_t data[] = { 
                    uint32_t(14 + 40 + AlignHi(_size, 4) * _param.height), 
                    0, 14 + 40, 40,
                    uint32_t(_param.width), uint32_t(_param.height), 
                    0x00180001, 0, 0, 0, 0, 0, 0 };
                _stream.Write(data, sizeof(data));
            }
            return true;
        }
    }
}

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

#include <cstdarg>

#ifndef STBIW_ASSERT
#include <assert.h>
#define STBIW_ASSERT(x) assert(x)
#endif

namespace Simd
{ 
    namespace Base
    {
#define STBIW_UCHAR(x) (unsigned char) ((x) & 0xff)

        typedef void stbi_write_func(void* context, void* data, int size);

        static void MemoryStreamWrite(void* context, void* data, int size)
        {
            ((OutputMemoryStream*)context)->Write(data, size);
        }

        typedef struct
        {
            stbi_write_func* func;
            void* context;
            unsigned char buffer[64];
            int buf_used;
            OutputMemoryStream *stream;
        } stbi__write_context;

        typedef unsigned int stbiw_uint32;
        typedef int stb_image_write_test[sizeof(stbiw_uint32) == 4 ? 1 : -1];

        static void stbiw__writefv(stbi__write_context* s, const char* fmt, va_list v)
        {
            while (*fmt) {
                switch (*fmt++) {
                case ' ': break;
                case '1': {
                    unsigned char x = STBIW_UCHAR(va_arg(v, int));
                    s->func(s->context, &x, 1);
                    break;
                }
                case '2': {
                    int x = va_arg(v, int);
                    unsigned char b[2];
                    b[0] = STBIW_UCHAR(x);
                    b[1] = STBIW_UCHAR(x >> 8);
                    s->func(s->context, b, 2);
                    break;
                }
                case '4': {
                    stbiw_uint32 x = va_arg(v, int);
                    unsigned char b[4];
                    b[0] = STBIW_UCHAR(x);
                    b[1] = STBIW_UCHAR(x >> 8);
                    b[2] = STBIW_UCHAR(x >> 16);
                    b[3] = STBIW_UCHAR(x >> 24);
                    s->func(s->context, b, 4);
                    break;
                }
                default:
                    STBIW_ASSERT(0);
                    return;
                }
            }
        }

        static void stbiw__writef(stbi__write_context* s, const char* fmt, ...)
        {
            va_list v;
            va_start(v, fmt);
            stbiw__writefv(s, fmt, v);
            va_end(v);
        }

        static void stbiw__write_flush(stbi__write_context* s)
        {
            if (s->buf_used) {
                s->func(s->context, &s->buffer, s->buf_used);
                s->buf_used = 0;
            }
        }

        static void stbiw__putc(stbi__write_context* s, unsigned char c)
        {
            s->func(s->context, &c, 1);
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
                s->func(s->context, &zero, scanline_pad);
            }
        }

        static int stbiw__outfile(stbi__write_context* s, int rgb_dir, int vdir, int x, int y, int stride, int comp, int expand_mono, void* data, int alpha, int pad, const char* fmt, ...)
        {
            if (y < 0 || x < 0) 
            {
                return 0;
            }
            else 
            {
                va_list v;
                va_start(v, fmt);
                stbiw__writefv(s, fmt, v);
                va_end(v);
                stbiw__write_pixels(s, rgb_dir, vdir, x, y, stride, comp, data, alpha, pad, expand_mono);
                return 1;
            }
        }

        static int stbi_write_bmp_core(stbi__write_context* s, int x, int y, int stride, int comp, const void* data)
        {
            if (comp != 4) 
            {
                int pad = (-x * 3) & 3;
                return stbiw__outfile(s, -1, -1, x, y, stride, comp, 1, (void*)data, 0, pad,
                    "11 4 22 4" "4 44 22 444444",
                    'B', 'M', 14 + 40 + (x * 3 + pad) * y, 0, 0, 14 + 40,  // file header
                    40, x, y, 1, 24, 0, 0, 0, 0, 0, 0);             // bitmap header
            }
            else 
            {
                return stbiw__outfile(s, -1, -1, x, y, stride, comp, 1, (void*)data, 1, 0,
                    "11 4 22 4" "4 44 22 444444 4444 4 444 444 444 444",
                    'B', 'M', 14 + 108 + x * y * 4, 0, 0, 14 + 108, // file header
                    108, x, y, 1, 32, 3, 0, 0, 0, 0, 0, 0xff0000, 0xff00, 0xff, 0xff000000u, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0); // bitmap V4 header
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
            s.context = s.stream;
            s.func = MemoryStreamWrite;
            //stbi__start_write_callbacks(&s, func, context);
            //return stbi_write_bmp_core(&s, x, y, comp, data);

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

                //for (size_t b = 0; b < block; ++b)
                //{
                //    uint8_t string[70];
                //    for (size_t col = 0, offset = 0; col < _param.width; ++col)
                //    {
                //        *(uint32_t*)(string + offset) = *(uint32_t*)g_pxmPrint[gray[col]];
                //        offset += 4;
                //        if (offset >= 68 || col == _param.width - 1)
                //        {
                //            string[offset++] = '\n';
                //            _stream.Write(string, offset);
                //            offset = 0;
                //        }
                //    }
                //    gray += grayStride;
                //}
                src += stride * block;
                row += block;
            }
            return true;
        }
    }
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse2.h"
#include "Simd/SimdSse41.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) 
    namespace Sse41
    {
        typedef unsigned char png_uc;
        typedef unsigned short png_us;

        typedef uint16_t png__uint16;
        typedef uint32_t png__uint32;

#define png_inline SIMD_INLINE
#define PNG_ASSERT assert
#define PNG_MALLOC(sz)           malloc(sz)
#define PNG_REALLOC(p,newsz)     realloc(p,newsz)
#define PNG_FREE(p)              free(p)
#define PNG_REALLOC_SIZED(p,oldsz,newsz) PNG_REALLOC(p,newsz)
#define STBIDEF static

#ifdef _MSC_VER
#define PNG_NOTUSED(v)  (void)(v)
#else
#define PNG_NOTUSED(v)  (void)sizeof(v)
#endif

#define PNG__BYTECAST(x)  ((png_uc) ((x) & 255))  // truncate int to byte without warnings
#define PNG_MAX_DIMENSIONS (1 << 24)

        static int png__err(const char* str, const char* stub)
        {
            return 0;
        }

#define png__errpuc(x,y)  ((unsigned char *)(size_t) (png__err(x,y)?NULL:NULL))

        static void* png__malloc(size_t size)
        {
            return PNG_MALLOC(size);
        }

        typedef struct
        {
            int      (*read)  (void* user, char* data, int size);   // fill 'data' with 'size' bytes.  return number of bytes actually read
            void     (*skip)  (void* user, int n);                 // skip the next 'n' bytes, or 'unget' the last -n bytes if negative
            int      (*eof)   (void* user);                       // returns nonzero if we are at end of file/data
        } png_io_callbacks;

        typedef struct
        {
            png__uint32 img_x, img_y;
            int img_n, img_out_n;

            png_io_callbacks io;
            void* io_user_data;

            int read_from_callbacks;
            int buflen;
            png_uc buffer_start[128];
            int callback_already_read;

            png_uc* img_buffer, * img_buffer_end;
            png_uc* img_buffer_original, * img_buffer_original_end;
        } png__context;

        typedef struct
        {
            int bits_per_channel;
            int num_channels;
            int channel_order;
        } png__result_info;

        enum
        {
            PNG__SCAN_load = 0,
            PNG__SCAN_type,
            PNG__SCAN_header
        };

        enum
        {
            PNG_ORDER_RGB,
            PNG_ORDER_BGR
        };

        static void png__rewind(png__context* s)
        {
            // conceptually rewind SHOULD rewind to the beginning of the stream,
            // but we just rewind to the beginning of the initial buffer, because
            // we only use it after doing 'test', which only ever looks at at most 92 bytes
            s->img_buffer = s->img_buffer_original;
            s->img_buffer_end = s->img_buffer_original_end;
        }

        static void png__refill_buffer(png__context* s)
        {
            int n = (s->io.read)(s->io_user_data, (char*)s->buffer_start, s->buflen);
            s->callback_already_read += (int)(s->img_buffer - s->img_buffer_original);
            if (n == 0) {
                // at end of file, treat same as if from memory, but need to handle case
                // where s->img_buffer isn't pointing to safe memory, e.g. 0-byte file
                s->read_from_callbacks = 0;
                s->img_buffer = s->buffer_start;
                s->img_buffer_end = s->buffer_start + 1;
                *s->img_buffer = 0;
            }
            else {
                s->img_buffer = s->buffer_start;
                s->img_buffer_end = s->buffer_start + n;
            }
        }

        png_inline static png_uc png__get8(png__context* s)
        {
            if (s->img_buffer < s->img_buffer_end)
                return *s->img_buffer++;
            if (s->read_from_callbacks) {
                png__refill_buffer(s);
                return *s->img_buffer++;
            }
            return 0;
        }

        static int png__get16be(png__context* s)
        {
            int z = png__get8(s);
            return (z << 8) + png__get8(s);
        }

        static png__uint32 png__get32be(png__context* s)
        {
            png__uint32 z = png__get16be(s);
            return (z << 16) + png__get16be(s);
        }

        png_inline static int png__at_eof(png__context* s)
        {
            if (s->io.read) {
                if (!(s->io.eof)(s->io_user_data)) return 0;
                // if feof() is true, check if buffer = end
                // special case: we've only got the special 0 character at the end
                if (s->read_from_callbacks == 0) return 1;
            }

            return s->img_buffer >= s->img_buffer_end;
        }

        static void png__skip(png__context* s, int n)
        {
            if (n == 0) return;  // already there!
            if (n < 0) {
                s->img_buffer = s->img_buffer_end;
                return;
            }
            if (s->io.read) {
                int blen = (int)(s->img_buffer_end - s->img_buffer);
                if (blen < n) {
                    s->img_buffer = s->img_buffer_end;
                    (s->io.skip)(s->io_user_data, n - blen);
                    return;
                }
            }
            s->img_buffer += n;
        }

        static int png__getn(png__context* s, png_uc* buffer, int n)
        {
            if (s->io.read) {
                int blen = (int)(s->img_buffer_end - s->img_buffer);
                if (blen < n) {
                    int res, count;

                    memcpy(buffer, s->img_buffer, blen);

                    count = (s->io.read)(s->io_user_data, (char*)buffer + blen, n - blen);
                    res = (count == (n - blen));
                    s->img_buffer = s->img_buffer_end;
                    return res;
                }
            }

            if (s->img_buffer + n <= s->img_buffer_end) {
                memcpy(buffer, s->img_buffer, n);
                s->img_buffer += n;
                return 1;
            }
            else
                return 0;
        }

        static int png__addsizes_valid(int a, int b)
        {
            if (b < 0) return 0;
            // now 0 <= b <= INT_MAX, hence also
            // 0 <= INT_MAX - b <= INTMAX.
            // And "a + b <= INT_MAX" (which might overflow) is the
            // same as a <= INT_MAX - b (no overflow)
            return a <= INT_MAX - b;
        }

        // returns 1 if the product is valid, 0 on overflow.
        // negative factors are considered invalid.
        static int png__mul2sizes_valid(int a, int b)
        {
            if (a < 0 || b < 0) return 0;
            if (b == 0) return 1; // mul-by-0 is always safe
            // portable way to check for no overflows in a*b
            return a <= INT_MAX / b;
        }

        // returns 1 if "a*b + add" has no negative terms/factors and doesn't overflow
        static int png__mad2sizes_valid(int a, int b, int add)
        {
            return png__mul2sizes_valid(a, b) && png__addsizes_valid(a * b, add);
        }

        // returns 1 if "a*b*c + add" has no negative terms/factors and doesn't overflow
        static int png__mad3sizes_valid(int a, int b, int c, int add)
        {
            return png__mul2sizes_valid(a, b) && png__mul2sizes_valid(a * b, c) &&
                png__addsizes_valid(a * b * c, add);
        }

        // returns 1 if "a*b*c*d + add" has no negative terms/factors and doesn't overflow
        static int png__mad4sizes_valid(int a, int b, int c, int d, int add)
        {
            return png__mul2sizes_valid(a, b) && png__mul2sizes_valid(a * b, c) &&
                png__mul2sizes_valid(a * b * c, d) && png__addsizes_valid(a * b * c * d, add);
        }

        // mallocs with size overflow checking
        static void* png__malloc_mad2(int a, int b, int add)
        {
            if (!png__mad2sizes_valid(a, b, add)) return NULL;
            return png__malloc(a * b + add);
        }

        static void* png__malloc_mad3(int a, int b, int c, int add)
        {
            if (!png__mad3sizes_valid(a, b, c, add)) return NULL;
            return png__malloc(a * b * c + add);
        }

        static void* png__malloc_mad4(int a, int b, int c, int d, int add)
        {
            if (!png__mad4sizes_valid(a, b, c, d, add)) return NULL;
            return png__malloc(a * b * c * d + add);
        }

        static png_uc png__compute_y(int r, int g, int b)
        {
            return (png_uc)(((r * 77) + (g * 150) + (29 * b)) >> 8);
        }

        static unsigned char* png__convert_format(unsigned char* data, int img_n, int req_comp, unsigned int x, unsigned int y)
        {
            int i, j;
            unsigned char* good;

            if (req_comp == img_n) return data;
            PNG_ASSERT(req_comp >= 1 && req_comp <= 4);

            good = (unsigned char*)png__malloc_mad3(req_comp, x, y, 0);
            if (good == NULL) {
                PNG_FREE(data);
                return png__errpuc("outofmem", "Out of memory");
            }

            for (j = 0; j < (int)y; ++j) {
                unsigned char* src = data + j * x * img_n;
                unsigned char* dest = good + j * x * req_comp;

#define PNG__COMBO(a,b)  ((a)*8+(b))
#define PNG__CASE(a,b)   case PNG__COMBO(a,b): for(i=x-1; i >= 0; --i, src += a, dest += b)
                // convert source image with img_n components to one with req_comp components;
                // avoid switch per pixel, so use switch per scanline and massive macros
                switch (PNG__COMBO(img_n, req_comp)) {
                    PNG__CASE(1, 2) { dest[0] = src[0]; dest[1] = 255; } break;
                    PNG__CASE(1, 3) { dest[0] = dest[1] = dest[2] = src[0]; } break;
                    PNG__CASE(1, 4) { dest[0] = dest[1] = dest[2] = src[0]; dest[3] = 255; } break;
                    PNG__CASE(2, 1) { dest[0] = src[0]; } break;
                    PNG__CASE(2, 3) { dest[0] = dest[1] = dest[2] = src[0]; } break;
                    PNG__CASE(2, 4) { dest[0] = dest[1] = dest[2] = src[0]; dest[3] = src[1]; } break;
                    PNG__CASE(3, 4) { dest[0] = src[0]; dest[1] = src[1]; dest[2] = src[2]; dest[3] = 255; } break;
                    PNG__CASE(3, 1) { dest[0] = png__compute_y(src[0], src[1], src[2]); } break;
                    PNG__CASE(3, 2) { dest[0] = png__compute_y(src[0], src[1], src[2]); dest[1] = 255; } break;
                    PNG__CASE(4, 1) { dest[0] = png__compute_y(src[0], src[1], src[2]); } break;
                    PNG__CASE(4, 2) { dest[0] = png__compute_y(src[0], src[1], src[2]); dest[1] = src[3]; } break;
                    PNG__CASE(4, 3) { dest[0] = src[0]; dest[1] = src[1]; dest[2] = src[2]; } break;
                default: PNG_ASSERT(0); PNG_FREE(data); PNG_FREE(good); return png__errpuc("unsupported", "Unsupported format conversion");
                }
#undef PNG__CASE
            }

            PNG_FREE(data);
            return good;
        }

        static png__uint16 png__compute_y_16(int r, int g, int b)
        {
            return (png__uint16)(((r * 77) + (g * 150) + (29 * b)) >> 8);
        }

        static png__uint16* png__convert_format16(png__uint16* data, int img_n, int req_comp, unsigned int x, unsigned int y)
        {
            int i, j;
            png__uint16* good;

            if (req_comp == img_n) return data;
            PNG_ASSERT(req_comp >= 1 && req_comp <= 4);

            good = (png__uint16*)png__malloc(req_comp * x * y * 2);
            if (good == NULL) {
                PNG_FREE(data);
                return (png__uint16*)png__errpuc("outofmem", "Out of memory");
            }

            for (j = 0; j < (int)y; ++j) {
                png__uint16* src = data + j * x * img_n;
                png__uint16* dest = good + j * x * req_comp;

#define PNG__COMBO(a,b)  ((a)*8+(b))
#define PNG__CASE(a,b)   case PNG__COMBO(a,b): for(i=x-1; i >= 0; --i, src += a, dest += b)
                // convert source image with img_n components to one with req_comp components;
                // avoid switch per pixel, so use switch per scanline and massive macros
                switch (PNG__COMBO(img_n, req_comp)) {
                    PNG__CASE(1, 2) { dest[0] = src[0]; dest[1] = 0xffff; } break;
                    PNG__CASE(1, 3) { dest[0] = dest[1] = dest[2] = src[0]; } break;
                    PNG__CASE(1, 4) { dest[0] = dest[1] = dest[2] = src[0]; dest[3] = 0xffff; } break;
                    PNG__CASE(2, 1) { dest[0] = src[0]; } break;
                    PNG__CASE(2, 3) { dest[0] = dest[1] = dest[2] = src[0]; } break;
                    PNG__CASE(2, 4) { dest[0] = dest[1] = dest[2] = src[0]; dest[3] = src[1]; } break;
                    PNG__CASE(3, 4) { dest[0] = src[0]; dest[1] = src[1]; dest[2] = src[2]; dest[3] = 0xffff; } break;
                    PNG__CASE(3, 1) { dest[0] = png__compute_y_16(src[0], src[1], src[2]); } break;
                    PNG__CASE(3, 2) { dest[0] = png__compute_y_16(src[0], src[1], src[2]); dest[1] = 0xffff; } break;
                    PNG__CASE(4, 1) { dest[0] = png__compute_y_16(src[0], src[1], src[2]); } break;
                    PNG__CASE(4, 2) { dest[0] = png__compute_y_16(src[0], src[1], src[2]); dest[1] = src[3]; } break;
                    PNG__CASE(4, 3) { dest[0] = src[0]; dest[1] = src[1]; dest[2] = src[2]; } break;
                default: PNG_ASSERT(0); PNG_FREE(data); PNG_FREE(good); return (png__uint16*)png__errpuc("unsupported", "Unsupported format conversion");
                }
#undef PNG__CASE
            }

            PNG_FREE(data);
            return good;
        }

        // fast-way is faster to check than jpeg huffman, but slow way is slower
#define PNG__ZFAST_BITS  9 // accelerate all cases in default tables
#define PNG__ZFAST_MASK  ((1 << PNG__ZFAST_BITS) - 1)

// zlib-style huffman encoding
// (jpegs packs from left, zlib from right, so can't share code)
        typedef struct
        {
            png__uint16 fast[1 << PNG__ZFAST_BITS];
            png__uint16 firstcode[16];
            int maxcode[17];
            png__uint16 firstsymbol[16];
            png_uc  size[288];
            png__uint16 value[288];
        } png__zhuffman;

        png_inline static int png__bitreverse16(int n)
        {
            n = ((n & 0xAAAA) >> 1) | ((n & 0x5555) << 1);
            n = ((n & 0xCCCC) >> 2) | ((n & 0x3333) << 2);
            n = ((n & 0xF0F0) >> 4) | ((n & 0x0F0F) << 4);
            n = ((n & 0xFF00) >> 8) | ((n & 0x00FF) << 8);
            return n;
        }

        png_inline static int png__bit_reverse(int v, int bits)
        {
            PNG_ASSERT(bits <= 16);
            // to bit reverse n bits, reverse 16 and shift
            // e.g. 11 bits, bit reverse and shift away 5
            return png__bitreverse16(v) >> (16 - bits);
        }

        static int png__zbuild_huffman(png__zhuffman* z, const png_uc* sizelist, int num)
        {
            int i, k = 0;
            int code, next_code[16], sizes[17];

            // DEFLATE spec for generating codes
            memset(sizes, 0, sizeof(sizes));
            memset(z->fast, 0, sizeof(z->fast));
            for (i = 0; i < num; ++i)
                ++sizes[sizelist[i]];
            sizes[0] = 0;
            for (i = 1; i < 16; ++i)
                if (sizes[i] > (1 << i))
                    return png__err("bad sizes", "Corrupt PNG");
            code = 0;
            for (i = 1; i < 16; ++i) {
                next_code[i] = code;
                z->firstcode[i] = (png__uint16)code;
                z->firstsymbol[i] = (png__uint16)k;
                code = (code + sizes[i]);
                if (sizes[i])
                    if (code - 1 >= (1 << i)) return png__err("bad codelengths", "Corrupt PNG");
                z->maxcode[i] = code << (16 - i); // preshift for inner loop
                code <<= 1;
                k += sizes[i];
            }
            z->maxcode[16] = 0x10000; // sentinel
            for (i = 0; i < num; ++i) {
                int s = sizelist[i];
                if (s) {
                    int c = next_code[s] - z->firstcode[s] + z->firstsymbol[s];
                    png__uint16 fastv = (png__uint16)((s << 9) | i);
                    z->size[c] = (png_uc)s;
                    z->value[c] = (png__uint16)i;
                    if (s <= PNG__ZFAST_BITS) {
                        int j = png__bit_reverse(next_code[s], s);
                        while (j < (1 << PNG__ZFAST_BITS)) {
                            z->fast[j] = fastv;
                            j += (1 << s);
                        }
                    }
                    ++next_code[s];
                }
            }
            return 1;
        }

        // zlib-from-memory implementation for PNG reading
        //    because PNG allows splitting the zlib stream arbitrarily,
        //    and it's annoying structurally to have PNG call ZLIB call PNG,
        //    we require PNG read all the IDATs and combine them into a single
        //    memory buffer

        typedef struct
        {
            png_uc* zbuffer, * zbuffer_end;
            int num_bits;
            png__uint32 code_buffer;

            char* zout;
            char* zout_start;
            char* zout_end;
            int   z_expandable;

            png__zhuffman z_length, z_distance;
        } png__zbuf;

        png_inline static int png__zeof(png__zbuf* z)
        {
            return (z->zbuffer >= z->zbuffer_end);
        }

        png_inline static png_uc png__zget8(png__zbuf* z)
        {
            return png__zeof(z) ? 0 : *z->zbuffer++;
        }

        static void png__fill_bits(png__zbuf* z)
        {
            do {
                if (z->code_buffer >= (1U << z->num_bits)) {
                    z->zbuffer = z->zbuffer_end;  /* treat this as EOF so we fail. */
                    return;
                }
                z->code_buffer |= (unsigned int)png__zget8(z) << z->num_bits;
                z->num_bits += 8;
            } while (z->num_bits <= 24);
        }

        png_inline static unsigned int png__zreceive(png__zbuf* z, int n)
        {
            unsigned int k;
            if (z->num_bits < n) png__fill_bits(z);
            k = z->code_buffer & ((1 << n) - 1);
            z->code_buffer >>= n;
            z->num_bits -= n;
            return k;
        }

        static int png__zhuffman_decode_slowpath(png__zbuf* a, png__zhuffman* z)
        {
            int b, s, k;
            // not resolved by fast table, so compute it the slow way
            // use jpeg approach, which requires MSbits at top
            k = png__bit_reverse(a->code_buffer, 16);
            for (s = PNG__ZFAST_BITS + 1; ; ++s)
                if (k < z->maxcode[s])
                    break;
            if (s >= 16) return -1; // invalid code!
            // code size is s, so:
            b = (k >> (16 - s)) - z->firstcode[s] + z->firstsymbol[s];
            if (b >= sizeof(z->size)) return -1; // some data was corrupt somewhere!
            if (z->size[b] != s) return -1;  // was originally an assert, but report failure instead.
            a->code_buffer >>= s;
            a->num_bits -= s;
            return z->value[b];
        }

        png_inline static int png__zhuffman_decode(png__zbuf* a, png__zhuffman* z)
        {
            int b, s;
            if (a->num_bits < 16) {
                if (png__zeof(a)) {
                    return -1;   /* report error for unexpected end of data. */
                }
                png__fill_bits(a);
            }
            b = z->fast[a->code_buffer & PNG__ZFAST_MASK];
            if (b) {
                s = b >> 9;
                a->code_buffer >>= s;
                a->num_bits -= s;
                return b & 511;
            }
            return png__zhuffman_decode_slowpath(a, z);
        }

        static int png__zexpand(png__zbuf* z, char* zout, int n)  // need to make room for n bytes
        {
            char* q;
            unsigned int cur, limit, old_limit;
            z->zout = zout;
            if (!z->z_expandable) return png__err("output buffer limit", "Corrupt PNG");
            cur = (unsigned int)(z->zout - z->zout_start);
            limit = old_limit = (unsigned)(z->zout_end - z->zout_start);
            if (UINT_MAX - cur < (unsigned)n) return png__err("outofmem", "Out of memory");
            while (cur + n > limit) {
                if (limit > UINT_MAX / 2) return png__err("outofmem", "Out of memory");
                limit *= 2;
            }
            q = (char*)PNG_REALLOC_SIZED(z->zout_start, old_limit, limit);
            PNG_NOTUSED(old_limit);
            if (q == NULL) return png__err("outofmem", "Out of memory");
            z->zout_start = q;
            z->zout = q + cur;
            z->zout_end = q + limit;
            return 1;
        }

        static const int png__zlength_base[31] = {
           3,4,5,6,7,8,9,10,11,13,
           15,17,19,23,27,31,35,43,51,59,
           67,83,99,115,131,163,195,227,258,0,0 };

        static const int png__zlength_extra[31] =
        { 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0 };

        static const int png__zdist_base[32] = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,
        257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,0,0 };

        static const int png__zdist_extra[32] =
        { 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };

        static int png__parse_huffman_block(png__zbuf* a)
        {
            SIMD_PERF_FUNC();

            char* zout = a->zout;
            for (;;) {
                int z = png__zhuffman_decode(a, &a->z_length);
                if (z < 256) {
                    if (z < 0) return png__err("bad huffman code", "Corrupt PNG"); // error in huffman codes
                    if (zout >= a->zout_end) {
                        if (!png__zexpand(a, zout, 1)) return 0;
                        zout = a->zout;
                    }
                    *zout++ = (char)z;
                }
                else {
                    png_uc* p;
                    int len, dist;
                    if (z == 256) {
                        a->zout = zout;
                        return 1;
                    }
                    z -= 257;
                    len = png__zlength_base[z];
                    if (png__zlength_extra[z]) len += png__zreceive(a, png__zlength_extra[z]);
                    z = png__zhuffman_decode(a, &a->z_distance);
                    if (z < 0) return png__err("bad huffman code", "Corrupt PNG");
                    dist = png__zdist_base[z];
                    if (png__zdist_extra[z]) dist += png__zreceive(a, png__zdist_extra[z]);
                    if (zout - a->zout_start < dist) return png__err("bad dist", "Corrupt PNG");
                    if (zout + len > a->zout_end) {
                        if (!png__zexpand(a, zout, len)) return 0;
                        zout = a->zout;
                    }
                    p = (png_uc*)(zout - dist);
                    if (dist == 1) { // run of one byte; common in images.
                        png_uc v = *p;
                        if (len) { do *zout++ = v; while (--len); }
                    }
                    else {
                        if (len) { do *zout++ = *p++; while (--len); }
                    }
                }
            }
        }

        static int png__compute_huffman_codes(png__zbuf* a)
        {
            static const png_uc length_dezigzag[19] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };
            png__zhuffman z_codelength;
            png_uc lencodes[286 + 32 + 137];//padding for maximum single op
            png_uc codelength_sizes[19];
            int i, n;

            int hlit = png__zreceive(a, 5) + 257;
            int hdist = png__zreceive(a, 5) + 1;
            int hclen = png__zreceive(a, 4) + 4;
            int ntot = hlit + hdist;

            memset(codelength_sizes, 0, sizeof(codelength_sizes));
            for (i = 0; i < hclen; ++i) {
                int s = png__zreceive(a, 3);
                codelength_sizes[length_dezigzag[i]] = (png_uc)s;
            }
            if (!png__zbuild_huffman(&z_codelength, codelength_sizes, 19)) return 0;

            n = 0;
            while (n < ntot) {
                int c = png__zhuffman_decode(a, &z_codelength);
                if (c < 0 || c >= 19) return png__err("bad codelengths", "Corrupt PNG");
                if (c < 16)
                    lencodes[n++] = (png_uc)c;
                else {
                    png_uc fill = 0;
                    if (c == 16) {
                        c = png__zreceive(a, 2) + 3;
                        if (n == 0) return png__err("bad codelengths", "Corrupt PNG");
                        fill = lencodes[n - 1];
                    }
                    else if (c == 17) {
                        c = png__zreceive(a, 3) + 3;
                    }
                    else if (c == 18) {
                        c = png__zreceive(a, 7) + 11;
                    }
                    else {
                        return png__err("bad codelengths", "Corrupt PNG");
                    }
                    if (ntot - n < c) return png__err("bad codelengths", "Corrupt PNG");
                    memset(lencodes + n, fill, c);
                    n += c;
                }
            }
            if (n != ntot) return png__err("bad codelengths", "Corrupt PNG");
            if (!png__zbuild_huffman(&a->z_length, lencodes, hlit)) return 0;
            if (!png__zbuild_huffman(&a->z_distance, lencodes + hlit, hdist)) return 0;
            return 1;
        }

        static int png__parse_uncompressed_block(png__zbuf* a)
        {
            png_uc header[4];
            int len, nlen, k;
            if (a->num_bits & 7)
                png__zreceive(a, a->num_bits & 7); // discard
             // drain the bit-packed data into header
            k = 0;
            while (a->num_bits > 0) {
                header[k++] = (png_uc)(a->code_buffer & 255); // suppress MSVC run-time check
                a->code_buffer >>= 8;
                a->num_bits -= 8;
            }
            if (a->num_bits < 0) return png__err("zlib corrupt", "Corrupt PNG");
            // now fill header the normal way
            while (k < 4)
                header[k++] = png__zget8(a);
            len = header[1] * 256 + header[0];
            nlen = header[3] * 256 + header[2];
            if (nlen != (len ^ 0xffff)) return png__err("zlib corrupt", "Corrupt PNG");
            if (a->zbuffer + len > a->zbuffer_end) return png__err("read past buffer", "Corrupt PNG");
            if (a->zout + len > a->zout_end)
                if (!png__zexpand(a, a->zout, len)) return 0;
            memcpy(a->zout, a->zbuffer, len);
            a->zbuffer += len;
            a->zout += len;
            return 1;
        }

        static int png__parse_zlib_header(png__zbuf* a)
        {
            int cmf = png__zget8(a);
            int cm = cmf & 15;
            /* int cinfo = cmf >> 4; */
            int flg = png__zget8(a);
            if (png__zeof(a)) return png__err("bad zlib header", "Corrupt PNG"); // zlib spec
            if ((cmf * 256 + flg) % 31 != 0) return png__err("bad zlib header", "Corrupt PNG"); // zlib spec
            if (flg & 32) return png__err("no preset dict", "Corrupt PNG"); // preset dictionary not allowed in png
            if (cm != 8) return png__err("bad compression", "Corrupt PNG"); // DEFLATE required for png
            // window = 1 << (8 + cinfo)... but who cares, we fully buffer output
            return 1;
        }

        static const png_uc png__zdefault_length[288] =
        {
           8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
           8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
           8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
           8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,
           8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
           9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
           9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
           9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9, 9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
           7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7, 7,7,7,7,7,7,7,7,8,8,8,8,8,8,8,8
        };
        static const png_uc png__zdefault_distance[32] =
        {
           5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
        };
        /*
        Init algorithm:
        {
           int i;   // use <= to match clearly with spec
           for (i=0; i <= 143; ++i)     png__zdefault_length[i]   = 8;
           for (   ; i <= 255; ++i)     png__zdefault_length[i]   = 9;
           for (   ; i <= 279; ++i)     png__zdefault_length[i]   = 7;
           for (   ; i <= 287; ++i)     png__zdefault_length[i]   = 8;

           for (i=0; i <=  31; ++i)     png__zdefault_distance[i] = 5;
        }
        */

        static int png__parse_zlib(png__zbuf* a, int parse_header)
        {
            int final, type;
            if (parse_header)
                if (!png__parse_zlib_header(a)) return 0;
            a->num_bits = 0;
            a->code_buffer = 0;
            do {
                final = png__zreceive(a, 1);
                type = png__zreceive(a, 2);
                if (type == 0) {
                    if (!png__parse_uncompressed_block(a)) return 0;
                }
                else if (type == 3) {
                    return 0;
                }
                else {
                    if (type == 1) {
                        // use fixed code lengths
                        if (!png__zbuild_huffman(&a->z_length, png__zdefault_length, 288)) return 0;
                        if (!png__zbuild_huffman(&a->z_distance, png__zdefault_distance, 32)) return 0;
                    }
                    else {
                        if (!png__compute_huffman_codes(a)) return 0;
                    }
                    if (!png__parse_huffman_block(a)) return 0;
                }
            } while (!final);
            return 1;
        }

        static int png__do_zlib(png__zbuf* a, char* obuf, int olen, int exp, int parse_header)
        {
            a->zout_start = obuf;
            a->zout = obuf;
            a->zout_end = obuf + olen;
            a->z_expandable = exp;

            return png__parse_zlib(a, parse_header);
        }

        STBIDEF char* png_zlib_decode_malloc_guesssize(const char* buffer, int len, int initial_size, int* outlen)
        {
            png__zbuf a;
            char* p = (char*)png__malloc(initial_size);
            if (p == NULL) return NULL;
            a.zbuffer = (png_uc*)buffer;
            a.zbuffer_end = (png_uc*)buffer + len;
            if (png__do_zlib(&a, p, initial_size, 1, 1)) {
                if (outlen) *outlen = (int)(a.zout - a.zout_start);
                return a.zout_start;
            }
            else {
                PNG_FREE(a.zout_start);
                return NULL;
            }
        }

        STBIDEF char* png_zlib_decode_malloc(char const* buffer, int len, int* outlen)
        {
            return png_zlib_decode_malloc_guesssize(buffer, len, 16384, outlen);
        }

        STBIDEF char* png_zlib_decode_malloc_guesssize_headerflag(const char* buffer, int len, int initial_size, int* outlen, int parse_header)
        {
            png__zbuf a;
            char* p = (char*)png__malloc(initial_size);
            if (p == NULL) return NULL;
            a.zbuffer = (png_uc*)buffer;
            a.zbuffer_end = (png_uc*)buffer + len;
            if (png__do_zlib(&a, p, initial_size, 1, parse_header)) {
                if (outlen) *outlen = (int)(a.zout - a.zout_start);
                return a.zout_start;
            }
            else {
                PNG_FREE(a.zout_start);
                return NULL;
            }
        }

        STBIDEF int png_zlib_decode_buffer(char* obuffer, int olen, char const* ibuffer, int ilen)
        {
            png__zbuf a;
            a.zbuffer = (png_uc*)ibuffer;
            a.zbuffer_end = (png_uc*)ibuffer + ilen;
            if (png__do_zlib(&a, obuffer, olen, 0, 1))
                return (int)(a.zout - a.zout_start);
            else
                return -1;
        }

        STBIDEF char* png_zlib_decode_noheader_malloc(char const* buffer, int len, int* outlen)
        {
            png__zbuf a;
            char* p = (char*)png__malloc(16384);
            if (p == NULL) return NULL;
            a.zbuffer = (png_uc*)buffer;
            a.zbuffer_end = (png_uc*)buffer + len;
            if (png__do_zlib(&a, p, 16384, 1, 0)) {
                if (outlen) *outlen = (int)(a.zout - a.zout_start);
                return a.zout_start;
            }
            else {
                PNG_FREE(a.zout_start);
                return NULL;
            }
        }

        STBIDEF int png_zlib_decode_noheader_buffer(char* obuffer, int olen, const char* ibuffer, int ilen)
        {
            png__zbuf a;
            a.zbuffer = (png_uc*)ibuffer;
            a.zbuffer_end = (png_uc*)ibuffer + ilen;
            if (png__do_zlib(&a, obuffer, olen, 0, 0))
                return (int)(a.zout - a.zout_start);
            else
                return -1;
        }


        // public domain "baseline" PNG decoder   v0.10  Sean Barrett 2006-11-18
        //    simple implementation
        //      - only 8-bit samples
        //      - no CRC checking
        //      - allocates lots of intermediate memory
        //        - avoids problem of streaming data between subsystems
        //        - avoids explicit window management
        //    performance
        //      - uses stb_zlib, a PD zlib implementation with fast huffman decoding

        typedef struct
        {
            png__uint32 length;
            png__uint32 type;
        } png__pngchunk;

        static png__pngchunk png__get_chunk_header(png__context* s)
        {
            png__pngchunk c;
            c.length = png__get32be(s);
            c.type = png__get32be(s);
            return c;
        }

        static int png__check_png_header(png__context* s)
        {
            static const png_uc png_sig[8] = { 137,80,78,71,13,10,26,10 };
            int i;
            for (i = 0; i < 8; ++i)
                if (png__get8(s) != png_sig[i]) return png__err("bad png sig", "Not a PNG");
            return 1;
        }

        typedef struct
        {
            png__context* s;
            png_uc* idata, * expanded, * out;
            int depth;
        } png__png;


        enum {
            PNG__F_none = 0,
            PNG__F_sub = 1,
            PNG__F_up = 2,
            PNG__F_avg = 3,
            PNG__F_paeth = 4,
            // synthetic filters used for first scanline to avoid needing a dummy row of 0s
            PNG__F_avg_first,
            PNG__F_paeth_first
        };

        static png_uc first_row_filter[5] =
        {
           PNG__F_none,
           PNG__F_sub,
           PNG__F_none,
           PNG__F_avg_first,
           PNG__F_paeth_first
        };

        static int png__paeth(int a, int b, int c)
        {
            int p = a + b - c;
            int pa = abs(p - a);
            int pb = abs(p - b);
            int pc = abs(p - c);
            if (pa <= pb && pa <= pc) return a;
            if (pb <= pc) return b;
            return c;
        }

        static const png_uc png__depth_scale_table[9] = { 0, 0xff, 0x55, 0, 0x11, 0,0,0, 0x01 };

        // create the png data from post-deflated data
        static int png__create_png_image_raw(png__png* a, png_uc* raw, png__uint32 raw_len, int out_n, png__uint32 x, png__uint32 y, int depth, int color)
        {
            int bytes = (depth == 16 ? 2 : 1);
            png__context* s = a->s;
            png__uint32 i, j, stride = x * out_n * bytes;
            png__uint32 img_len, img_width_bytes;
            int k;
            int img_n = s->img_n; // copy it into a local for later

            int output_bytes = out_n * bytes;
            int filter_bytes = img_n * bytes;
            int width = x;

            PNG_ASSERT(out_n == s->img_n || out_n == s->img_n + 1);
            a->out = (png_uc*)png__malloc_mad3(x, y, output_bytes, 0); // extra bytes to write off the end into
            if (!a->out) return png__err("outofmem", "Out of memory");

            if (!png__mad3sizes_valid(img_n, x, depth, 7)) return png__err("too large", "Corrupt PNG");
            img_width_bytes = (((img_n * x * depth) + 7) >> 3);
            img_len = (img_width_bytes + 1) * y;

            // we used to check for exact match between raw_len and img_len on non-interlaced PNGs,
            // but issue #276 reported a PNG in the wild that had extra data at the end (all zeros),
            // so just check for raw_len < img_len always.
            if (raw_len < img_len) return png__err("not enough pixels", "Corrupt PNG");

            for (j = 0; j < y; ++j) {
                png_uc* cur = a->out + stride * j;
                png_uc* prior;
                int filter = *raw++;

                if (filter > 4)
                    return png__err("invalid filter", "Corrupt PNG");

                if (depth < 8) {
                    if (img_width_bytes > x) return png__err("invalid width", "Corrupt PNG");
                    cur += x * out_n - img_width_bytes; // store output to the rightmost img_len bytes, so we can decode in place
                    filter_bytes = 1;
                    width = img_width_bytes;
                }
                prior = cur - stride; // bugfix: need to compute this after 'cur +=' computation above

                // if first row, use special filter that doesn't sample previous row
                if (j == 0) filter = first_row_filter[filter];

                // handle first byte explicitly
                for (k = 0; k < filter_bytes; ++k) {
                    switch (filter) {
                    case PNG__F_none: cur[k] = raw[k]; break;
                    case PNG__F_sub: cur[k] = raw[k]; break;
                    case PNG__F_up: cur[k] = PNG__BYTECAST(raw[k] + prior[k]); break;
                    case PNG__F_avg: cur[k] = PNG__BYTECAST(raw[k] + (prior[k] >> 1)); break;
                    case PNG__F_paeth: cur[k] = PNG__BYTECAST(raw[k] + png__paeth(0, prior[k], 0)); break;
                    case PNG__F_avg_first: cur[k] = raw[k]; break;
                    case PNG__F_paeth_first: cur[k] = raw[k]; break;
                    }
                }

                if (depth == 8) {
                    if (img_n != out_n)
                        cur[img_n] = 255; // first pixel
                    raw += img_n;
                    cur += out_n;
                    prior += out_n;
                }
                else if (depth == 16) {
                    if (img_n != out_n) {
                        cur[filter_bytes] = 255; // first pixel top byte
                        cur[filter_bytes + 1] = 255; // first pixel bottom byte
                    }
                    raw += filter_bytes;
                    cur += output_bytes;
                    prior += output_bytes;
                }
                else {
                    raw += 1;
                    cur += 1;
                    prior += 1;
                }

                // this is a little gross, so that we don't switch per-pixel or per-component
                if (depth < 8 || img_n == out_n) {
                    int nk = (width - 1) * filter_bytes;
#define PNG__CASE(f) \
             case f:     \
                for (k=0; k < nk; ++k)
                    switch (filter) {
                        // "none" filter turns into a memcpy here; make that explicit.
                    case PNG__F_none:         memcpy(cur, raw, nk); break;
                        PNG__CASE(PNG__F_sub) { cur[k] = PNG__BYTECAST(raw[k] + cur[k - filter_bytes]); } break;
                        PNG__CASE(PNG__F_up) { cur[k] = PNG__BYTECAST(raw[k] + prior[k]); } break;
                        PNG__CASE(PNG__F_avg) { cur[k] = PNG__BYTECAST(raw[k] + ((prior[k] + cur[k - filter_bytes]) >> 1)); } break;
                        PNG__CASE(PNG__F_paeth) { cur[k] = PNG__BYTECAST(raw[k] + png__paeth(cur[k - filter_bytes], prior[k], prior[k - filter_bytes])); } break;
                        PNG__CASE(PNG__F_avg_first) { cur[k] = PNG__BYTECAST(raw[k] + (cur[k - filter_bytes] >> 1)); } break;
                        PNG__CASE(PNG__F_paeth_first) { cur[k] = PNG__BYTECAST(raw[k] + png__paeth(cur[k - filter_bytes], 0, 0)); } break;
                    }
#undef PNG__CASE
                    raw += nk;
                }
                else {
                    PNG_ASSERT(img_n + 1 == out_n);
#define PNG__CASE(f) \
             case f:     \
                for (i=x-1; i >= 1; --i, cur[filter_bytes]=255,raw+=filter_bytes,cur+=output_bytes,prior+=output_bytes) \
                   for (k=0; k < filter_bytes; ++k)
                    switch (filter) {
                        PNG__CASE(PNG__F_none) { cur[k] = raw[k]; } break;
                        PNG__CASE(PNG__F_sub) { cur[k] = PNG__BYTECAST(raw[k] + cur[k - output_bytes]); } break;
                        PNG__CASE(PNG__F_up) { cur[k] = PNG__BYTECAST(raw[k] + prior[k]); } break;
                        PNG__CASE(PNG__F_avg) { cur[k] = PNG__BYTECAST(raw[k] + ((prior[k] + cur[k - output_bytes]) >> 1)); } break;
                        PNG__CASE(PNG__F_paeth) { cur[k] = PNG__BYTECAST(raw[k] + png__paeth(cur[k - output_bytes], prior[k], prior[k - output_bytes])); } break;
                        PNG__CASE(PNG__F_avg_first) { cur[k] = PNG__BYTECAST(raw[k] + (cur[k - output_bytes] >> 1)); } break;
                        PNG__CASE(PNG__F_paeth_first) { cur[k] = PNG__BYTECAST(raw[k] + png__paeth(cur[k - output_bytes], 0, 0)); } break;
                    }
#undef PNG__CASE

                    // the loop above sets the high byte of the pixels' alpha, but for
                    // 16 bit png files we also need the low byte set. we'll do that here.
                    if (depth == 16) {
                        cur = a->out + stride * j; // start at the beginning of the row again
                        for (i = 0; i < x; ++i, cur += output_bytes) {
                            cur[filter_bytes + 1] = 255;
                        }
                    }
                }
            }

            // we make a separate pass to expand bits to pixels; for performance,
            // this could run two scanlines behind the above code, so it won't
            // intefere with filtering but will still be in the cache.
            if (depth < 8) {
                for (j = 0; j < y; ++j) {
                    png_uc* cur = a->out + stride * j;
                    png_uc* in = a->out + stride * j + x * out_n - img_width_bytes;
                    // unpack 1/2/4-bit into a 8-bit buffer. allows us to keep the common 8-bit path optimal at minimal cost for 1/2/4-bit
                    // png guarante byte alignment, if width is not multiple of 8/4/2 we'll decode dummy trailing data that will be skipped in the later loop
                    png_uc scale = (color == 0) ? png__depth_scale_table[depth] : 1; // scale grayscale values to 0..255 range

                    // note that the final byte might overshoot and write more data than desired.
                    // we can allocate enough data that this never writes out of memory, but it
                    // could also overwrite the next scanline. can it overwrite non-empty data
                    // on the next scanline? yes, consider 1-pixel-wide scanlines with 1-bit-per-pixel.
                    // so we need to explicitly clamp the final ones

                    if (depth == 4) {
                        for (k = x * img_n; k >= 2; k -= 2, ++in) {
                            *cur++ = scale * ((*in >> 4));
                            *cur++ = scale * ((*in) & 0x0f);
                        }
                        if (k > 0) *cur++ = scale * ((*in >> 4));
                    }
                    else if (depth == 2) {
                        for (k = x * img_n; k >= 4; k -= 4, ++in) {
                            *cur++ = scale * ((*in >> 6));
                            *cur++ = scale * ((*in >> 4) & 0x03);
                            *cur++ = scale * ((*in >> 2) & 0x03);
                            *cur++ = scale * ((*in) & 0x03);
                        }
                        if (k > 0) *cur++ = scale * ((*in >> 6));
                        if (k > 1) *cur++ = scale * ((*in >> 4) & 0x03);
                        if (k > 2) *cur++ = scale * ((*in >> 2) & 0x03);
                    }
                    else if (depth == 1) {
                        for (k = x * img_n; k >= 8; k -= 8, ++in) {
                            *cur++ = scale * ((*in >> 7));
                            *cur++ = scale * ((*in >> 6) & 0x01);
                            *cur++ = scale * ((*in >> 5) & 0x01);
                            *cur++ = scale * ((*in >> 4) & 0x01);
                            *cur++ = scale * ((*in >> 3) & 0x01);
                            *cur++ = scale * ((*in >> 2) & 0x01);
                            *cur++ = scale * ((*in >> 1) & 0x01);
                            *cur++ = scale * ((*in) & 0x01);
                        }
                        if (k > 0) *cur++ = scale * ((*in >> 7));
                        if (k > 1) *cur++ = scale * ((*in >> 6) & 0x01);
                        if (k > 2) *cur++ = scale * ((*in >> 5) & 0x01);
                        if (k > 3) *cur++ = scale * ((*in >> 4) & 0x01);
                        if (k > 4) *cur++ = scale * ((*in >> 3) & 0x01);
                        if (k > 5) *cur++ = scale * ((*in >> 2) & 0x01);
                        if (k > 6) *cur++ = scale * ((*in >> 1) & 0x01);
                    }
                    if (img_n != out_n) {
                        int q;
                        // insert alpha = 255
                        cur = a->out + stride * j;
                        if (img_n == 1) {
                            for (q = x - 1; q >= 0; --q) {
                                cur[q * 2 + 1] = 255;
                                cur[q * 2 + 0] = cur[q];
                            }
                        }
                        else {
                            PNG_ASSERT(img_n == 3);
                            for (q = x - 1; q >= 0; --q) {
                                cur[q * 4 + 3] = 255;
                                cur[q * 4 + 2] = cur[q * 3 + 2];
                                cur[q * 4 + 1] = cur[q * 3 + 1];
                                cur[q * 4 + 0] = cur[q * 3 + 0];
                            }
                        }
                    }
                }
            }
            else if (depth == 16) {
                // force the image data from big-endian to platform-native.
                // this is done in a separate pass due to the decoding relying
                // on the data being untouched, but could probably be done
                // per-line during decode if care is taken.
                png_uc* cur = a->out;
                png__uint16* cur16 = (png__uint16*)cur;

                for (i = 0; i < x * y * out_n; ++i, cur16++, cur += 2) {
                    *cur16 = (cur[0] << 8) | cur[1];
                }
            }

            return 1;
        }

        static int png__create_png_image(png__png* a, png_uc* image_data, png__uint32 image_data_len, int out_n, int depth, int color, int interlaced)
        {
            int bytes = (depth == 16 ? 2 : 1);
            int out_bytes = out_n * bytes;
            png_uc* final;
            int p;
            if (!interlaced)
                return png__create_png_image_raw(a, image_data, image_data_len, out_n, a->s->img_x, a->s->img_y, depth, color);

            // de-interlacing
            final = (png_uc*)png__malloc_mad3(a->s->img_x, a->s->img_y, out_bytes, 0);
            for (p = 0; p < 7; ++p) {
                int xorig[] = { 0,4,0,2,0,1,0 };
                int yorig[] = { 0,0,4,0,2,0,1 };
                int xspc[] = { 8,8,4,4,2,2,1 };
                int yspc[] = { 8,8,8,4,4,2,2 };
                int i, j, x, y;
                // pass1_x[4] = 0, pass1_x[5] = 1, pass1_x[12] = 1
                x = (a->s->img_x - xorig[p] + xspc[p] - 1) / xspc[p];
                y = (a->s->img_y - yorig[p] + yspc[p] - 1) / yspc[p];
                if (x && y) {
                    png__uint32 img_len = ((((a->s->img_n * x * depth) + 7) >> 3) + 1) * y;
                    if (!png__create_png_image_raw(a, image_data, image_data_len, out_n, x, y, depth, color)) {
                        PNG_FREE(final);
                        return 0;
                    }
                    for (j = 0; j < y; ++j) {
                        for (i = 0; i < x; ++i) {
                            int out_y = j * yspc[p] + yorig[p];
                            int out_x = i * xspc[p] + xorig[p];
                            memcpy(final + out_y * a->s->img_x * out_bytes + out_x * out_bytes,
                                a->out + (j * x + i) * out_bytes, out_bytes);
                        }
                    }
                    PNG_FREE(a->out);
                    image_data += img_len;
                    image_data_len -= img_len;
                }
            }
            a->out = final;

            return 1;
        }

        static int png__compute_transparency(png__png* z, png_uc tc[3], int out_n)
        {
            png__context* s = z->s;
            png__uint32 i, pixel_count = s->img_x * s->img_y;
            png_uc* p = z->out;

            // compute color-based transparency, assuming we've
            // already got 255 as the alpha value in the output
            PNG_ASSERT(out_n == 2 || out_n == 4);

            if (out_n == 2) {
                for (i = 0; i < pixel_count; ++i) {
                    p[1] = (p[0] == tc[0] ? 0 : 255);
                    p += 2;
                }
            }
            else {
                for (i = 0; i < pixel_count; ++i) {
                    if (p[0] == tc[0] && p[1] == tc[1] && p[2] == tc[2])
                        p[3] = 0;
                    p += 4;
                }
            }
            return 1;
        }

        static int png__compute_transparency16(png__png* z, png__uint16 tc[3], int out_n)
        {
            png__context* s = z->s;
            png__uint32 i, pixel_count = s->img_x * s->img_y;
            png__uint16* p = (png__uint16*)z->out;

            // compute color-based transparency, assuming we've
            // already got 65535 as the alpha value in the output
            PNG_ASSERT(out_n == 2 || out_n == 4);

            if (out_n == 2) {
                for (i = 0; i < pixel_count; ++i) {
                    p[1] = (p[0] == tc[0] ? 0 : 65535);
                    p += 2;
                }
            }
            else {
                for (i = 0; i < pixel_count; ++i) {
                    if (p[0] == tc[0] && p[1] == tc[1] && p[2] == tc[2])
                        p[3] = 0;
                    p += 4;
                }
            }
            return 1;
        }

        static int png__expand_png_palette(png__png* a, png_uc* palette, int len, int pal_img_n)
        {
            png__uint32 i, pixel_count = a->s->img_x * a->s->img_y;
            png_uc* p, * temp_out, * orig = a->out;

            p = (png_uc*)png__malloc_mad2(pixel_count, pal_img_n, 0);
            if (p == NULL) return png__err("outofmem", "Out of memory");

            // between here and free(out) below, exitting would leak
            temp_out = p;

            if (pal_img_n == 3) {
                for (i = 0; i < pixel_count; ++i) {
                    int n = orig[i] * 4;
                    p[0] = palette[n];
                    p[1] = palette[n + 1];
                    p[2] = palette[n + 2];
                    p += 3;
                }
            }
            else {
                for (i = 0; i < pixel_count; ++i) {
                    int n = orig[i] * 4;
                    p[0] = palette[n];
                    p[1] = palette[n + 1];
                    p[2] = palette[n + 2];
                    p[3] = palette[n + 3];
                    p += 4;
                }
            }
            PNG_FREE(a->out);
            a->out = temp_out;

            PNG_NOTUSED(len);

            return 1;
        }

        static int png__unpremultiply_on_load = 0;
        static int png__de_iphone_flag = 0;

        STBIDEF void png_set_unpremultiply_on_load(int flag_true_if_should_unpremultiply)
        {
            png__unpremultiply_on_load = flag_true_if_should_unpremultiply;
        }

        STBIDEF void png_convert_iphone_png_to_rgb(int flag_true_if_should_convert)
        {
            png__de_iphone_flag = flag_true_if_should_convert;
        }

        static void png__de_iphone(png__png* z)
        {
            png__context* s = z->s;
            png__uint32 i, pixel_count = s->img_x * s->img_y;
            png_uc* p = z->out;

            if (s->img_out_n == 3) {  // convert bgr to rgb
                for (i = 0; i < pixel_count; ++i) {
                    png_uc t = p[0];
                    p[0] = p[2];
                    p[2] = t;
                    p += 3;
                }
            }
            else {
                PNG_ASSERT(s->img_out_n == 4);
                if (png__unpremultiply_on_load) {
                    // convert bgr to rgb and unpremultiply
                    for (i = 0; i < pixel_count; ++i) {
                        png_uc a = p[3];
                        png_uc t = p[0];
                        if (a) {
                            png_uc half = a / 2;
                            p[0] = (p[2] * 255 + half) / a;
                            p[1] = (p[1] * 255 + half) / a;
                            p[2] = (t * 255 + half) / a;
                        }
                        else {
                            p[0] = p[2];
                            p[2] = t;
                        }
                        p += 4;
                    }
                }
                else {
                    // convert bgr to rgb
                    for (i = 0; i < pixel_count; ++i) {
                        png_uc t = p[0];
                        p[0] = p[2];
                        p[2] = t;
                        p += 4;
                    }
                }
            }
        }

#define PNG__PNG_TYPE(a,b,c,d)  (((unsigned) (a) << 24) + ((unsigned) (b) << 16) + ((unsigned) (c) << 8) + (unsigned) (d))

        static int png__parse_png_file(png__png* z, int scan, int req_comp)
        {
            png_uc palette[1024], pal_img_n = 0;
            png_uc has_trans = 0, tc[3] = { 0 };
            png__uint16 tc16[3];
            png__uint32 ioff = 0, idata_limit = 0, i, pal_len = 0;
            int first = 1, k, interlace = 0, color = 0, is_iphone = 0;
            png__context* s = z->s;

            z->expanded = NULL;
            z->idata = NULL;
            z->out = NULL;

            if (!png__check_png_header(s)) return 0;

            if (scan == PNG__SCAN_type) return 1;

            for (;;) {
                png__pngchunk c = png__get_chunk_header(s);
                switch (c.type) {
                case PNG__PNG_TYPE('C', 'g', 'B', 'I'):
                    is_iphone = 1;
                    png__skip(s, c.length);
                    break;
                case PNG__PNG_TYPE('I', 'H', 'D', 'R'): {
                    int comp, filter;
                    if (!first) return png__err("multiple IHDR", "Corrupt PNG");
                    first = 0;
                    if (c.length != 13) return png__err("bad IHDR len", "Corrupt PNG");
                    s->img_x = png__get32be(s);
                    s->img_y = png__get32be(s);
                    if (s->img_y > PNG_MAX_DIMENSIONS) return png__err("too large", "Very large image (corrupt?)");
                    if (s->img_x > PNG_MAX_DIMENSIONS) return png__err("too large", "Very large image (corrupt?)");
                    z->depth = png__get8(s);  if (z->depth != 1 && z->depth != 2 && z->depth != 4 && z->depth != 8 && z->depth != 16)  return png__err("1/2/4/8/16-bit only", "PNG not supported: 1/2/4/8/16-bit only");
                    color = png__get8(s);  if (color > 6)         return png__err("bad ctype", "Corrupt PNG");
                    if (color == 3 && z->depth == 16)                  return png__err("bad ctype", "Corrupt PNG");
                    if (color == 3) pal_img_n = 3; else if (color & 1) return png__err("bad ctype", "Corrupt PNG");
                    comp = png__get8(s);  if (comp) return png__err("bad comp method", "Corrupt PNG");
                    filter = png__get8(s);  if (filter) return png__err("bad filter method", "Corrupt PNG");
                    interlace = png__get8(s); if (interlace > 1) return png__err("bad interlace method", "Corrupt PNG");
                    if (!s->img_x || !s->img_y) return png__err("0-pixel image", "Corrupt PNG");
                    if (!pal_img_n) {
                        s->img_n = (color & 2 ? 3 : 1) + (color & 4 ? 1 : 0);
                        if ((1 << 30) / s->img_x / s->img_n < s->img_y) return png__err("too large", "Image too large to decode");
                        if (scan == PNG__SCAN_header) return 1;
                    }
                    else {
                        // if paletted, then pal_n is our final components, and
                        // img_n is # components to decompress/filter.
                        s->img_n = 1;
                        if ((1 << 30) / s->img_x / 4 < s->img_y) return png__err("too large", "Corrupt PNG");
                        // if SCAN_header, have to scan to see if we have a tRNS
                    }
                    break;
                }

                case PNG__PNG_TYPE('P', 'L', 'T', 'E'): {
                    if (first) return png__err("first not IHDR", "Corrupt PNG");
                    if (c.length > 256 * 3) return png__err("invalid PLTE", "Corrupt PNG");
                    pal_len = c.length / 3;
                    if (pal_len * 3 != c.length) return png__err("invalid PLTE", "Corrupt PNG");
                    for (i = 0; i < pal_len; ++i) {
                        palette[i * 4 + 0] = png__get8(s);
                        palette[i * 4 + 1] = png__get8(s);
                        palette[i * 4 + 2] = png__get8(s);
                        palette[i * 4 + 3] = 255;
                    }
                    break;
                }

                case PNG__PNG_TYPE('t', 'R', 'N', 'S'): {
                    if (first) return png__err("first not IHDR", "Corrupt PNG");
                    if (z->idata) return png__err("tRNS after IDAT", "Corrupt PNG");
                    if (pal_img_n) {
                        if (scan == PNG__SCAN_header) { s->img_n = 4; return 1; }
                        if (pal_len == 0) return png__err("tRNS before PLTE", "Corrupt PNG");
                        if (c.length > pal_len) return png__err("bad tRNS len", "Corrupt PNG");
                        pal_img_n = 4;
                        for (i = 0; i < c.length; ++i)
                            palette[i * 4 + 3] = png__get8(s);
                    }
                    else {
                        if (!(s->img_n & 1)) return png__err("tRNS with alpha", "Corrupt PNG");
                        if (c.length != (png__uint32)s->img_n * 2) return png__err("bad tRNS len", "Corrupt PNG");
                        has_trans = 1;
                        if (z->depth == 16) {
                            for (k = 0; k < s->img_n; ++k) tc16[k] = (png__uint16)png__get16be(s); // copy the values as-is
                        }
                        else {
                            for (k = 0; k < s->img_n; ++k) tc[k] = (png_uc)(png__get16be(s) & 255) * png__depth_scale_table[z->depth]; // non 8-bit images will be larger
                        }
                    }
                    break;
                }

                case PNG__PNG_TYPE('I', 'D', 'A', 'T'): {
                    if (first) return png__err("first not IHDR", "Corrupt PNG");
                    if (pal_img_n && !pal_len) return png__err("no PLTE", "Corrupt PNG");
                    if (scan == PNG__SCAN_header) { s->img_n = pal_img_n; return 1; }
                    if ((int)(ioff + c.length) < (int)ioff) return 0;
                    if (ioff + c.length > idata_limit) {
                        png__uint32 idata_limit_old = idata_limit;
                        png_uc* p;
                        if (idata_limit == 0) idata_limit = c.length > 4096 ? c.length : 4096;
                        while (ioff + c.length > idata_limit)
                            idata_limit *= 2;
                        PNG_NOTUSED(idata_limit_old);
                        p = (png_uc*)PNG_REALLOC_SIZED(z->idata, idata_limit_old, idata_limit); if (p == NULL) return png__err("outofmem", "Out of memory");
                        z->idata = p;
                    }
                    if (!png__getn(s, z->idata + ioff, c.length)) return png__err("outofdata", "Corrupt PNG");
                    ioff += c.length;
                    break;
                }

                case PNG__PNG_TYPE('I', 'E', 'N', 'D'): {
                    png__uint32 raw_len, bpl;
                    if (first) return png__err("first not IHDR", "Corrupt PNG");
                    if (scan != PNG__SCAN_load) return 1;
                    if (z->idata == NULL) return png__err("no IDAT", "Corrupt PNG");
                    // initial guess for decoded data size to avoid unnecessary reallocs
                    bpl = (s->img_x * z->depth + 7) / 8; // bytes per line, per component
                    raw_len = bpl * s->img_y * s->img_n /* pixels */ + s->img_y /* filter mode per row */;
                    z->expanded = (png_uc*)png_zlib_decode_malloc_guesssize_headerflag((char*)z->idata, ioff, raw_len, (int*)&raw_len, !is_iphone);
                    if (z->expanded == NULL) return 0; // zlib should set error
                    PNG_FREE(z->idata); z->idata = NULL;
                    if ((req_comp == s->img_n + 1 && req_comp != 3 && !pal_img_n) || has_trans)
                        s->img_out_n = s->img_n + 1;
                    else
                        s->img_out_n = s->img_n;
                    if (!png__create_png_image(z, z->expanded, raw_len, s->img_out_n, z->depth, color, interlace)) return 0;
                    if (has_trans) {
                        if (z->depth == 16) {
                            if (!png__compute_transparency16(z, tc16, s->img_out_n)) return 0;
                        }
                        else {
                            if (!png__compute_transparency(z, tc, s->img_out_n)) return 0;
                        }
                    }
                    if (is_iphone && png__de_iphone_flag && s->img_out_n > 2)
                        png__de_iphone(z);
                    if (pal_img_n) {
                        // pal_img_n == 3 or 4
                        s->img_n = pal_img_n; // record the actual colors we had
                        s->img_out_n = pal_img_n;
                        if (req_comp >= 3) s->img_out_n = req_comp;
                        if (!png__expand_png_palette(z, palette, pal_len, s->img_out_n))
                            return 0;
                    }
                    else if (has_trans) {
                        // non-paletted image with tRNS -> source image has (constant) alpha
                        ++s->img_n;
                    }
                    PNG_FREE(z->expanded); z->expanded = NULL;
                    // end of PNG chunk, read and skip CRC
                    png__get32be(s);
                    return 1;
                }

                default:
                    // if critical, fail
                    if (first) return png__err("first not IHDR", "Corrupt PNG");
                    if ((c.type & (1 << 29)) == 0) {
#ifndef PNG_NO_FAILURE_STRINGS
                        // not threadsafe
                        static char invalid_chunk[] = "XXXX PNG chunk not known";
                        invalid_chunk[0] = PNG__BYTECAST(c.type >> 24);
                        invalid_chunk[1] = PNG__BYTECAST(c.type >> 16);
                        invalid_chunk[2] = PNG__BYTECAST(c.type >> 8);
                        invalid_chunk[3] = PNG__BYTECAST(c.type >> 0);
#endif
                        return png__err(invalid_chunk, "PNG not supported: unknown PNG chunk type");
                    }
                    png__skip(s, c.length);
                    break;
                }
                // end of PNG chunk, read and skip CRC
                png__get32be(s);
            }
        }

        static void* png__do_png(png__png* p, int* x, int* y, int* n, int req_comp, png__result_info* ri)
        {
            void* result = NULL;
            if (req_comp < 0 || req_comp > 4) return png__errpuc("bad req_comp", "Internal error");
            if (png__parse_png_file(p, PNG__SCAN_load, req_comp)) {
                if (p->depth <= 8)
                    ri->bits_per_channel = 8;
                else if (p->depth == 16)
                    ri->bits_per_channel = 16;
                else
                    return png__errpuc("bad bits_per_channel", "PNG not supported: unsupported color depth");
                result = p->out;
                p->out = NULL;
                if (req_comp && req_comp != p->s->img_out_n) {
                    if (ri->bits_per_channel == 8)
                        result = png__convert_format((unsigned char*)result, p->s->img_out_n, req_comp, p->s->img_x, p->s->img_y);
                    else
                        result = png__convert_format16((png__uint16*)result, p->s->img_out_n, req_comp, p->s->img_x, p->s->img_y);
                    p->s->img_out_n = req_comp;
                    if (result == NULL) return result;
                }
                *x = p->s->img_x;
                *y = p->s->img_y;
                if (n) *n = p->s->img_n;
            }
            PNG_FREE(p->out);      p->out = NULL;
            PNG_FREE(p->expanded); p->expanded = NULL;
            PNG_FREE(p->idata);    p->idata = NULL;

            return result;
        }

        static void* png__png_load(png__context* s, int* x, int* y, int* comp, int req_comp, png__result_info* ri)
        {
            png__png p;
            p.s = s;
            return png__do_png(&p, x, y, comp, req_comp, ri);
        }

        static int png__png_test(png__context* s)
        {
            int r;
            r = png__check_png_header(s);
            png__rewind(s);
            return r;
        }

        static int png__png_info_raw(png__png* p, int* x, int* y, int* comp)
        {
            if (!png__parse_png_file(p, PNG__SCAN_header, 0)) {
                png__rewind(p->s);
                return 0;
            }
            if (x) *x = p->s->img_x;
            if (y) *y = p->s->img_y;
            if (comp) *comp = p->s->img_n;
            return 1;
        }

        static int png__png_info(png__context* s, int* x, int* y, int* comp)
        {
            png__png p;
            p.s = s;
            return png__png_info_raw(&p, x, y, comp);
        }

        static int png__png_is16(png__context* s)
        {
            png__png p;
            p.s = s;
            if (!png__png_info_raw(&p, NULL, NULL, NULL))
                return 0;
            if (p.depth != 16) {
                png__rewind(p.s);
                return 0;
            }
            return 1;
        }

        static void* png__load_main(png__context* s, int* x, int* y, int* comp, int req_comp, png__result_info* ri, int bpc)
        {
            memset(ri, 0, sizeof(*ri)); // make sure it's initialized if we add new fields
            ri->bits_per_channel = 8; // default is 8 so most paths don't have to be changed
            ri->channel_order = PNG_ORDER_RGB; // all current input & output are this, but this is here so we can add BGR order
            ri->num_channels = 0;

            if (png__png_test(s))  return png__png_load(s, x, y, comp, req_comp, ri);

            return png__errpuc("unknown image type", "Image not of any known type, or corrupt");
        }

        static png_uc* png__convert_16_to_8(png__uint16* orig, int w, int h, int channels)
        {
            int i;
            int img_len = w * h * channels;
            png_uc* reduced;

            reduced = (png_uc*)png__malloc(img_len);
            if (reduced == NULL) return png__errpuc("outofmem", "Out of memory");

            for (i = 0; i < img_len; ++i)
                reduced[i] = (png_uc)((orig[i] >> 8) & 0xFF); // top half of each byte is sufficient approx of 16->8 bit scaling

            PNG_FREE(orig);
            return reduced;
        }

        static unsigned char* png__load_and_postprocess_8bit(png__context* s, int* x, int* y, int* comp, int req_comp)
        {
            png__result_info ri;
            void* result = png__load_main(s, x, y, comp, req_comp, &ri, 8);

            if (result == NULL)
                return NULL;

            // it is the responsibility of the loaders to make sure we get either 8 or 16 bit.
            PNG_ASSERT(ri.bits_per_channel == 8 || ri.bits_per_channel == 16);

            if (ri.bits_per_channel != 8) {
                result = png__convert_16_to_8((png__uint16*)result, *x, *y, req_comp == 0 ? *comp : req_comp);
                ri.bits_per_channel = 8;
            }

            // @TODO: move png__convert_format to here

            //if (png__vertically_flip_on_load) {
            //    int channels = req_comp ? req_comp : *comp;
            //    png__vertical_flip(result, *x, *y, channels * sizeof(png_uc));
            //}

            return (unsigned char*)result;
        }

        static void png__start_mem(png__context* s, png_uc const* buffer, int len)
        {
            s->io.read = NULL;
            s->read_from_callbacks = 0;
            s->callback_already_read = 0;
            s->img_buffer = s->img_buffer_original = (png_uc*)buffer;
            s->img_buffer_end = s->img_buffer_original_end = (png_uc*)buffer + len;
        }

        STBIDEF png_uc* png_load_from_memory(png_uc const* buffer, int len, int* x, int* y, int* comp, int req_comp)
        {
            png__context s;
            png__start_mem(&s, buffer, len);
            return png__load_and_postprocess_8bit(&s, x, y, comp, req_comp);
        }

        //------------------------------------------------------------------------

        static int png__stdio_read(void* user, char* data, int size)
        {
            InputMemoryStream* stream = (InputMemoryStream*)user;
            return (int)stream->Read(size, data);
        }

        static void png__stdio_skip(void* user, int n)
        {
            InputMemoryStream* stream = (InputMemoryStream*)user;
            stream->Skip(n);
        }

        static int png__stdio_eof(void* user)
        {
            InputMemoryStream* stream = (InputMemoryStream*)user;
            return stream->Pos() == stream->Size() ? 1 : 0;
        }


        //---------------------------------------------------------------------

        ImagePngLoader::ImagePngLoader(const ImageLoaderParam& param)
            : Base::ImagePngLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgb24;
        }

        bool ImagePngLoader::FromStream()
        {
            const int req_comp = 4;
            int x, y, comp;
            png__context s;
            s.io.eof = png__stdio_eof;
            s.io.read = png__stdio_read;
            s.io.skip = png__stdio_skip;
            s.io_user_data = &_stream;
            s.buflen = sizeof(s.buffer_start);
            s.read_from_callbacks = 1;
            s.callback_already_read = 0;
            s.img_buffer = s.img_buffer_original = s.buffer_start;
            png__refill_buffer(&s);
            s.img_buffer_original_end = s.img_buffer_end;
            png__result_info ri;
            uint8_t* data = (uint8_t*)png__png_load(&s, &x, &y, &comp, req_comp, &ri);
            if (data)
            {
                if (ri.bits_per_channel == 16)
                {
                    const uint16_t* src = (uint16_t*)data;
                    size_t size = x * y * req_comp;
                    uint8_t* dst = (uint8_t*)PNG_MALLOC(size);
                    for (size_t i = 0; i < size; ++i)
                        dst[i] = uint8_t(src[i] >> 8);
                    PNG_FREE(data);
                    data = dst;
                }
                size_t stride = 4 * x;
                _image.Recreate(x, y, (Image::Format)_param.format);
                if (x < A)
                {
                    switch (_param.format)
                    {
                    case SimdPixelFormatGray8:
                        Base::RgbaToGray(data, x, y, stride, _image.data, _image.stride);
                        break;
                    case SimdPixelFormatBgr24:
                        Base::BgraToRgb(data, x, y, stride, _image.data, _image.stride);
                        break;
                    case SimdPixelFormatBgra32:
                        Base::BgraToRgba(data, x, y, stride, _image.data, _image.stride);
                        break;
                    case SimdPixelFormatRgb24:
                        Base::BgraToBgr(data, x, y, stride, _image.data, _image.stride);
                        break;
                    case SimdPixelFormatRgba32:
                        Base::Copy(data, stride, x, y, 4, _image.data, _image.stride);
                        break;
                    default:
                        break;
                    }
                }
                else
                {
                    switch (_param.format)
                    {
                    case SimdPixelFormatGray8:
                        Sse2::RgbaToGray(data, x, y, stride, _image.data, _image.stride);
                        break;
                    case SimdPixelFormatBgr24:
                        Sse41::BgraToRgb(data, x, y, stride, _image.data, _image.stride);
                        break;
                    case SimdPixelFormatBgra32:
                        Sse41::BgraToRgba(data, x, y, stride, _image.data, _image.stride);
                        break;
                    case SimdPixelFormatRgb24:
                        Sse41::BgraToBgr(data, x, y, stride, _image.data, _image.stride);
                        break;
                    case SimdPixelFormatRgba32:
                        Base::Copy(data, stride, x, y, 4, _image.data, _image.stride);
                        break;
                    default:
                        break;
                    }
                }
                PNG_FREE(data);
                return true;
            }
            return false;
        }
    }
#endif
}

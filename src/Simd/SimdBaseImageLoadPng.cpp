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
#define PNG_MALLOC(sz)           malloc(sz)
#define PNG_REALLOC(p,newsz)     realloc(p,newsz)
#define PNG_FREE(p)              free(p)
#define PNG_REALLOC_SIZED(p,oldsz,newsz) PNG_REALLOC(p,newsz)

#ifdef _MSC_VER
#define PNG_NOTUSED(v)  (void)(v)
#else
#define PNG_NOTUSED(v)  (void)sizeof(v)
#endif

#define PNG__BYTECAST(x)  ((uint8_t) ((x) & 255))  // truncate int to byte without warnings

        SIMD_INLINE int PngError(const char* str, const char* stub)
        {
            std::cout << "PNG load error: " << str << ", " << stub << "!" << std::endl;
            return 0;
        }

        SIMD_INLINE uint8_t * PngErrorPtr(const char* str, const char* stub)
        {
            return (uint8_t*)(size_t)(PngError(str, stub) ? NULL : NULL);
        }

        static void* png__malloc(size_t size)
        {
            return PNG_MALLOC(size);
        }

        struct PngContext
        {
            uint32_t img_x, img_y;
            int img_n, img_out_n;

            InputMemoryStream * stream;
        };

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

        static uint8_t png__compute_y(int r, int g, int b)
        {
            return (uint8_t)(((r * 77) + (g * 150) + (29 * b)) >> 8);
        }

        static uint8_t* png__convert_format(uint8_t* data, int img_n, int req_comp, unsigned int x, unsigned int y)
        {
            int i, j;
            uint8_t* good;

            if (req_comp == img_n) 
                return data;
            assert(req_comp >= 1 && req_comp <= 4);

            good = (uint8_t*)png__malloc_mad3(req_comp, x, y, 0);
            if (good == NULL) 
            {
                PNG_FREE(data);
                return PngErrorPtr("outofmem", "Out of memory");
            }

            for (j = 0; j < (int)y; ++j) 
            {
                uint8_t* src = data + j * x * img_n;
                uint8_t* dest = good + j * x * req_comp;

#define PNG__COMBO(a,b)  ((a)*8+(b))
#define PNG__CASE(a,b)   case PNG__COMBO(a,b): for(i=x-1; i >= 0; --i, src += a, dest += b)
                // convert source image with img_n components to one with req_comp components;
                // avoid switch per pixel, so use switch per scanline and massive macros
                switch (PNG__COMBO(img_n, req_comp)) 
                {
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
                default: assert(0); PNG_FREE(data); PNG_FREE(good); return PngErrorPtr("unsupported", "Unsupported format conversion");
                }
#undef PNG__CASE
            }

            PNG_FREE(data);
            return good;
        }

        static uint16_t png__compute_y_16(int r, int g, int b)
        {
            return (uint16_t)(((r * 77) + (g * 150) + (29 * b)) >> 8);
        }

        static uint16_t* png__convert_format16(uint16_t* data, int img_n, int req_comp, unsigned int x, unsigned int y)
        {
            int i, j;
            uint16_t* good;

            if (req_comp == img_n) return data;
            assert(req_comp >= 1 && req_comp <= 4);

            good = (uint16_t*)png__malloc(req_comp * x * y * 2);
            if (good == NULL) {
                PNG_FREE(data);
                return (uint16_t*)PngErrorPtr("outofmem", "Out of memory");
            }

            for (j = 0; j < (int)y; ++j) {
                uint16_t* src = data + j * x * img_n;
                uint16_t* dest = good + j * x * req_comp;

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
                default: assert(0); PNG_FREE(data); PNG_FREE(good); return (uint16_t*)PngErrorPtr("unsupported", "Unsupported format conversion");
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
            uint16_t fast[1 << PNG__ZFAST_BITS];
            uint16_t firstcode[16];
            int maxcode[17];
            uint16_t firstsymbol[16];
            uint8_t  size[288];
            uint16_t value[288];
        } png__zhuffman;

        SIMD_INLINE static int png__bitreverse16(int n)
        {
            n = ((n & 0xAAAA) >> 1) | ((n & 0x5555) << 1);
            n = ((n & 0xCCCC) >> 2) | ((n & 0x3333) << 2);
            n = ((n & 0xF0F0) >> 4) | ((n & 0x0F0F) << 4);
            n = ((n & 0xFF00) >> 8) | ((n & 0x00FF) << 8);
            return n;
        }

        SIMD_INLINE static int png__bit_reverse(int v, int bits)
        {
            assert(bits <= 16);
            // to bit reverse n bits, reverse 16 and shift
            // e.g. 11 bits, bit reverse and shift away 5
            return png__bitreverse16(v) >> (16 - bits);
        }

        static int png__zbuild_huffman(png__zhuffman* z, const uint8_t* sizelist, int num)
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
                    return PngError("bad sizes", "Corrupt PNG");
            code = 0;
            for (i = 1; i < 16; ++i) {
                next_code[i] = code;
                z->firstcode[i] = (uint16_t)code;
                z->firstsymbol[i] = (uint16_t)k;
                code = (code + sizes[i]);
                if (sizes[i])
                    if (code - 1 >= (1 << i)) return PngError("bad codelengths", "Corrupt PNG");
                z->maxcode[i] = code << (16 - i); // preshift for inner loop
                code <<= 1;
                k += sizes[i];
            }
            z->maxcode[16] = 0x10000; // sentinel
            for (i = 0; i < num; ++i) {
                int s = sizelist[i];
                if (s) {
                    int c = next_code[s] - z->firstcode[s] + z->firstsymbol[s];
                    uint16_t fastv = (uint16_t)((s << 9) | i);
                    z->size[c] = (uint8_t)s;
                    z->value[c] = (uint16_t)i;
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
            uint8_t* zbuffer, * zbuffer_end;
            int num_bits;
            uint32_t code_buffer;

            char* zout;
            char* zout_start;
            char* zout_end;
            int   z_expandable;

            png__zhuffman z_length, z_distance;
        } png__zbuf;

        SIMD_INLINE static int png__zeof(png__zbuf* z)
        {
            return (z->zbuffer >= z->zbuffer_end);
        }

        SIMD_INLINE static uint8_t png__zget8(png__zbuf* z)
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

        SIMD_INLINE static unsigned int png__zreceive(png__zbuf* z, int n)
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

        SIMD_INLINE static int png__zhuffman_decode(png__zbuf* a, png__zhuffman* z)
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
            if (!z->z_expandable) return PngError("output buffer limit", "Corrupt PNG");
            cur = (unsigned int)(z->zout - z->zout_start);
            limit = old_limit = (unsigned)(z->zout_end - z->zout_start);
            if (UINT_MAX - cur < (unsigned)n) return PngError("outofmem", "Out of memory");
            while (cur + n > limit) {
                if (limit > UINT_MAX / 2) return PngError("outofmem", "Out of memory");
                limit *= 2;
            }
            q = (char*)PNG_REALLOC_SIZED(z->zout_start, old_limit, limit);
            PNG_NOTUSED(old_limit);
            if (q == NULL) return PngError("outofmem", "Out of memory");
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
            char* zout = a->zout;
            for (;;) {
                int z = png__zhuffman_decode(a, &a->z_length);
                if (z < 256) {
                    if (z < 0) return PngError("bad huffman code", "Corrupt PNG"); // error in huffman codes
                    if (zout >= a->zout_end) {
                        if (!png__zexpand(a, zout, 1)) return 0;
                        zout = a->zout;
                    }
                    *zout++ = (char)z;
                }
                else {
                    uint8_t* p;
                    int len, dist;
                    if (z == 256) {
                        a->zout = zout;
                        return 1;
                    }
                    z -= 257;
                    len = png__zlength_base[z];
                    if (png__zlength_extra[z]) len += png__zreceive(a, png__zlength_extra[z]);
                    z = png__zhuffman_decode(a, &a->z_distance);
                    if (z < 0) return PngError("bad huffman code", "Corrupt PNG");
                    dist = png__zdist_base[z];
                    if (png__zdist_extra[z]) dist += png__zreceive(a, png__zdist_extra[z]);
                    if (zout - a->zout_start < dist) return PngError("bad dist", "Corrupt PNG");
                    if (zout + len > a->zout_end) {
                        if (!png__zexpand(a, zout, len)) return 0;
                        zout = a->zout;
                    }
                    p = (uint8_t*)(zout - dist);
                    if (dist == 1) { // run of one byte; common in images.
                        uint8_t v = *p;
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
            static const uint8_t length_dezigzag[19] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };
            png__zhuffman z_codelength;
            uint8_t lencodes[286 + 32 + 137];//padding for maximum single op
            uint8_t codelength_sizes[19];
            int i, n;

            int hlit = png__zreceive(a, 5) + 257;
            int hdist = png__zreceive(a, 5) + 1;
            int hclen = png__zreceive(a, 4) + 4;
            int ntot = hlit + hdist;

            memset(codelength_sizes, 0, sizeof(codelength_sizes));
            for (i = 0; i < hclen; ++i) {
                int s = png__zreceive(a, 3);
                codelength_sizes[length_dezigzag[i]] = (uint8_t)s;
            }
            if (!png__zbuild_huffman(&z_codelength, codelength_sizes, 19)) return 0;

            n = 0;
            while (n < ntot) {
                int c = png__zhuffman_decode(a, &z_codelength);
                if (c < 0 || c >= 19) return PngError("bad codelengths", "Corrupt PNG");
                if (c < 16)
                    lencodes[n++] = (uint8_t)c;
                else {
                    uint8_t fill = 0;
                    if (c == 16) {
                        c = png__zreceive(a, 2) + 3;
                        if (n == 0) return PngError("bad codelengths", "Corrupt PNG");
                        fill = lencodes[n - 1];
                    }
                    else if (c == 17) {
                        c = png__zreceive(a, 3) + 3;
                    }
                    else if (c == 18) {
                        c = png__zreceive(a, 7) + 11;
                    }
                    else {
                        return PngError("bad codelengths", "Corrupt PNG");
                    }
                    if (ntot - n < c) return PngError("bad codelengths", "Corrupt PNG");
                    memset(lencodes + n, fill, c);
                    n += c;
                }
            }
            if (n != ntot) return PngError("bad codelengths", "Corrupt PNG");
            if (!png__zbuild_huffman(&a->z_length, lencodes, hlit)) return 0;
            if (!png__zbuild_huffman(&a->z_distance, lencodes + hlit, hdist)) return 0;
            return 1;
        }

        static int png__parse_uncompressed_block(png__zbuf* a)
        {
            uint8_t header[4];
            int len, nlen, k;
            if (a->num_bits & 7)
                png__zreceive(a, a->num_bits & 7); // discard
             // drain the bit-packed data into header
            k = 0;
            while (a->num_bits > 0) {
                header[k++] = (uint8_t)(a->code_buffer & 255); // suppress MSVC run-time check
                a->code_buffer >>= 8;
                a->num_bits -= 8;
            }
            if (a->num_bits < 0) return PngError("zlib corrupt", "Corrupt PNG");
            // now fill header the normal way
            while (k < 4)
                header[k++] = png__zget8(a);
            len = header[1] * 256 + header[0];
            nlen = header[3] * 256 + header[2];
            if (nlen != (len ^ 0xffff)) return PngError("zlib corrupt", "Corrupt PNG");
            if (a->zbuffer + len > a->zbuffer_end) return PngError("read past buffer", "Corrupt PNG");
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
            if (png__zeof(a)) return PngError("bad zlib header", "Corrupt PNG"); // zlib spec
            if ((cmf * 256 + flg) % 31 != 0) return PngError("bad zlib header", "Corrupt PNG"); // zlib spec
            if (flg & 32) return PngError("no preset dict", "Corrupt PNG"); // preset dictionary not allowed in png
            if (cm != 8) return PngError("bad compression", "Corrupt PNG"); // DEFLATE required for png
            // window = 1 << (8 + cinfo)... but who cares, we fully buffer output
            return 1;
        }

        static const uint8_t png__zdefault_length[288] =
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
        static const uint8_t png__zdefault_distance[32] =
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

        static char* png_zlib_decode_malloc_guesssize_headerflag(const char* buffer, int len, int initial_size, int* outlen, int parse_header)
        {
            png__zbuf a;
            char* p = (char*)png__malloc(initial_size);
            if (p == NULL) 
                return NULL;
            a.zbuffer = (uint8_t*)buffer;
            a.zbuffer_end = (uint8_t*)buffer + len;
            if (png__do_zlib(&a, p, initial_size, 1, parse_header)) 
            {
                if (outlen) 
                    *outlen = (int)(a.zout - a.zout_start);
                return a.zout_start;
            }
            else 
            {
                PNG_FREE(a.zout_start);
                return NULL;
            }
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
            PngContext* s;
            uint8_t * expanded, * out;
            uint8_t depth;
        } png__png;

        enum 
        {
            PNG__F_none = 0,
            PNG__F_sub = 1,
            PNG__F_up = 2,
            PNG__F_avg = 3,
            PNG__F_paeth = 4,
            // synthetic filters used for first scanline to avoid needing a dummy row of 0s
            PNG__F_avg_first,
            PNG__F_paeth_first
        };

        static uint8_t first_row_filter[5] =
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

        static const uint8_t png__depth_scale_table[9] = { 0, 0xff, 0x55, 0, 0x11, 0,0,0, 0x01 };

        // create the png data from post-deflated data
        static int png__create_png_image_raw(png__png* a, uint8_t* raw, uint32_t raw_len, int out_n, uint32_t x, uint32_t y, int depth, int color)
        {
            int bytes = (depth == 16 ? 2 : 1);
            PngContext* s = a->s;
            uint32_t i, j, stride = x * out_n * bytes;
            uint32_t img_len, img_width_bytes;
            int k;
            int img_n = s->img_n; // copy it into a local for later

            int output_bytes = out_n * bytes;
            int filter_bytes = img_n * bytes;
            int width = x;

            assert(out_n == s->img_n || out_n == s->img_n + 1);
            a->out = (uint8_t*)png__malloc_mad3(x, y, output_bytes, 0); // extra bytes to write off the end into
            if (!a->out) return PngError("outofmem", "Out of memory");

            if (!png__mad3sizes_valid(img_n, x, depth, 7)) return PngError("too large", "Corrupt PNG");
            img_width_bytes = (((img_n * x * depth) + 7) >> 3);
            img_len = (img_width_bytes + 1) * y;

            // we used to check for exact match between raw_len and img_len on non-interlaced PNGs,
            // but issue #276 reported a PNG in the wild that had extra data at the end (all zeros),
            // so just check for raw_len < img_len always.
            if (raw_len < img_len) return PngError("not enough pixels", "Corrupt PNG");

            for (j = 0; j < y; ++j) {
                uint8_t* cur = a->out + stride * j;
                uint8_t* prior;
                int filter = *raw++;

                if (filter > 4)
                    return PngError("invalid filter", "Corrupt PNG");

                if (depth < 8) {
                    if (img_width_bytes > x) return PngError("invalid width", "Corrupt PNG");
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
                    assert(img_n + 1 == out_n);
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
                    uint8_t* cur = a->out + stride * j;
                    uint8_t* in = a->out + stride * j + x * out_n - img_width_bytes;
                    // unpack 1/2/4-bit into a 8-bit buffer. allows us to keep the common 8-bit path optimal at minimal cost for 1/2/4-bit
                    // png guarante byte alignment, if width is not multiple of 8/4/2 we'll decode dummy trailing data that will be skipped in the later loop
                    uint8_t scale = (color == 0) ? png__depth_scale_table[depth] : 1; // scale grayscale values to 0..255 range

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
                            assert(img_n == 3);
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
            else if (depth == 16) 
            {
                // force the image data from big-endian to platform-native.
                // this is done in a separate pass due to the decoding relying
                // on the data being untouched, but could probably be done
                // per-line during decode if care is taken.
                uint8_t* cur = a->out;
                uint16_t* cur16 = (uint16_t*)cur;

                for (i = 0; i < x * y * out_n; ++i, cur16++, cur += 2)
                    *cur16 = (cur[0] << 8) | cur[1];
            }

            return 1;
        }

        static int png__create_png_image(png__png* a, uint8_t* image_data, uint32_t image_data_len, int out_n, int depth, int color, int interlaced)
        {
            int bytes = (depth == 16 ? 2 : 1);
            int out_bytes = out_n * bytes;
            uint8_t* final;
            int p;
            if (!interlaced)
                return png__create_png_image_raw(a, image_data, image_data_len, out_n, a->s->img_x, a->s->img_y, depth, color);

            // de-interlacing
            final = (uint8_t*)png__malloc_mad3(a->s->img_x, a->s->img_y, out_bytes, 0);
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
                    uint32_t img_len = ((((a->s->img_n * x * depth) + 7) >> 3) + 1) * y;
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

        static int png__compute_transparency(png__png* z, uint8_t tc[3], int out_n)
        {
            PngContext* s = z->s;
            uint32_t i, pixel_count = s->img_x * s->img_y;
            uint8_t* p = z->out;

            // compute color-based transparency, assuming we've
            // already got 255 as the alpha value in the output
            assert(out_n == 2 || out_n == 4);

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

        static int png__compute_transparency16(png__png* z, uint16_t tc[3], int out_n)
        {
            PngContext* s = z->s;
            uint32_t i, pixel_count = s->img_x * s->img_y;
            uint16_t* p = (uint16_t*)z->out;

            // compute color-based transparency, assuming we've
            // already got 65535 as the alpha value in the output
            assert(out_n == 2 || out_n == 4);

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

        static int png__expand_png_palette(png__png* a, uint8_t* palette, int len, int pal_img_n)
        {
            uint32_t i, pixel_count = a->s->img_x * a->s->img_y;
            uint8_t* p, * temp_out, * orig = a->out;

            p = (uint8_t*)png__malloc_mad2(pixel_count, pal_img_n, 0);
            if (p == NULL) return PngError("outofmem", "Out of memory");

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

        //---------------------------------------------------------------------

        ImagePngLoader::ImagePngLoader(const ImageLoaderParam& param)
            : ImageLoader(param)
            , _toAny8(NULL)
            , _toBgra8(NULL)
            , _toAny16(NULL)
            , _toBgra16(NULL)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgba32;
        }

        void ImagePngLoader::SetConverters()
        {
            _bgrToBgra = Base::BgrToBgra;
        }

        SIMD_INLINE constexpr uint32_t ChunkType(char a, char b, char c, char d)
        {
            return ((uint32_t(a) << 24) + (uint32_t(b) << 16) + (uint32_t(c) << 8) + uint32_t(d));
        }

        bool ImagePngLoader::FromStream()
        {
            const int req_comp = 4;
            PngContext context;
            context.stream = &_stream;
            png__png p;
            p.s = &context;
            png__png* z = &p;

            PngContext* s = z->s;

            z->expanded = NULL;
            z->out = NULL;

            if (!ParseFile())
                return false;

            s->img_x = _width;
            s->img_y = _height;
            z->depth = _depth;
            s->img_n = _channels;

            uint32_t isize = 0;
            char * idata = (char*)MergedData(isize);
            if (idata == NULL) 
                return false;

            uint32_t bpl = (s->img_x * z->depth + 7) / 8;
            uint32_t raw_len = bpl * s->img_y * s->img_n + s->img_y;
            z->expanded = (uint8_t*)png_zlib_decode_malloc_guesssize_headerflag(idata, isize, raw_len, (int*)&raw_len, _iPhone ? 0 : 1);
            if (z->expanded == NULL) 
                return false;
            if ((req_comp == s->img_n + 1 && req_comp != 3 && !_paletteChannels) || _hasTrans)
                s->img_out_n = s->img_n + 1;
            else
                s->img_out_n = s->img_n;
            if (!png__create_png_image(z, z->expanded, raw_len, s->img_out_n, z->depth, _color, _interlace)) 
                return 0;
            if (_hasTrans) 
            {
                if (z->depth == 16)
                {
                    if (!png__compute_transparency16(z, _tc16, s->img_out_n))
                        return false;
                }
                else
                {
                    if (!png__compute_transparency(z, _tc, s->img_out_n))
                        return false;
                }
            }
            if (_paletteChannels)
            {
                s->img_n = _paletteChannels; // record the actual colors we had
                s->img_out_n = _paletteChannels;
                if (req_comp >= 3) s->img_out_n = req_comp;
                if (!png__expand_png_palette(z, _palette.data, (int)_palette.size, s->img_out_n))
                    return false;
            }
            else if (_hasTrans)
                ++s->img_n;
            PNG_FREE(z->expanded); z->expanded = NULL;

            if (!(p.depth <= 8 || p.depth == 16))
                return false;
            uint8_t* data = p.out;
            p.out = NULL;
            if (req_comp && req_comp != p.s->img_out_n)
            {
                if (p.depth <= 8)
                    data = png__convert_format((uint8_t*)data, p.s->img_out_n, req_comp, p.s->img_x, p.s->img_y);
                else
                    data = (uint8_t*)png__convert_format16((uint16_t*)data, p.s->img_out_n, req_comp, p.s->img_x, p.s->img_y);
                p.s->img_out_n = req_comp;
                if (data == NULL)
                    return false;
            }
            if (p.depth == 16)
            {
                size_t size = context.img_x * context.img_y * context.img_n;
                const uint16_t* src = (uint16_t*)data;
                uint8_t* dst = (uint8_t*)PNG_MALLOC(size);
                for (size_t i = 0; i < size; ++i)
                    dst[i] = uint8_t(src[i] >> 8);
                PNG_FREE(data);
                data = dst;
            }
            PNG_FREE(p.out);
            PNG_FREE(p.expanded);
            if (data)
            {
                size_t stride = 4 * context.img_x;
                _image.Recreate(context.img_x, context.img_y, (Image::Format)_param.format);
                switch (_param.format)
                {
                case SimdPixelFormatGray8:
                    Base::RgbaToGray(data, context.img_x, context.img_y, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatBgr24:
                    Base::BgraToRgb(data, context.img_x, context.img_y, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatBgra32:
                    Base::BgraToRgba(data, context.img_x, context.img_y, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatRgb24:
                    Base::BgraToBgr(data, context.img_x, context.img_y, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatRgba32:
                    Base::Copy(data, stride, context.img_x, context.img_y, 4, _image.data, _image.stride);
                    break;
                default: 
                    break;
                }
                PNG_FREE(data);
                return true;
            }
            return false;
        }

        bool ImagePngLoader::ParseFile()
        {
            _first = true, _iPhone = false, _hasTrans = false;
            if (!CheckHeader())
                return false;
            for (bool run = true; run;)
            {
                Chunk chunk;
                if (!ReadChunk(chunk))
                    return 0;
                if (chunk.type == ChunkType('C', 'g', 'B', 'I'))
                {
                    _iPhone = true;
                    _stream.Skip(chunk.size);
                }
                else if (chunk.type == ChunkType('I', 'H', 'D', 'R'))
                {
                    if (!ReadHeader(chunk))
                        return false;
                    SetConverters();
                }
                else if (chunk.type == ChunkType('P', 'L', 'T', 'E'))
                {
                    if (!ReadPalette(chunk))
                        return false;
                }
                else if (chunk.type == ChunkType('t', 'R', 'N', 'S'))
                {
                    if (!ReadTransparency(chunk))
                        return false;
                }
                else if (chunk.type == ChunkType('I', 'D', 'A', 'T'))
                {
                    if (!ReadData(chunk))
                        return false;
                }
                else if (chunk.type == ChunkType('I', 'E', 'N', 'D'))
                {
                    if (_first)
                        return false;
                    run = false;
                }
                else
                {
                    if (_first || (chunk.type & (1 << 29)) == 0)
                        return false;
                    _stream.Skip(chunk.size);
                }
                uint32_t crc32;
                if (!_stream.ReadBe32u(crc32))
                    return false;
            }
            return true;
        }

        bool ImagePngLoader::CheckHeader()
        {
            const size_t size = 8;
            const uint8_t control[size] = { 137, 80, 78, 71, 13, 10, 26, 10 };
            uint8_t buffer[size];
            return _stream.Read(size, buffer) == size && memcmp(buffer, control, size) == 0;
        }

        SIMD_INLINE bool ImagePngLoader::ReadChunk(Chunk& chunk)
        {
            if (_stream.ReadBe32u(chunk.size) && _stream.ReadBe32u(chunk.type))
            {
                chunk.offs = (uint32_t)_stream.Pos();
                return true;
            }
            return false;
        }

        bool ImagePngLoader::ReadHeader(const Chunk& chunk)
        {
            const int MAX_SIZE = 1 << 24;
            if (!_first)
                return false;
            _first = false;
            if (!(chunk.size == 13 && _stream.CanRead(13)))
                return false;
            uint8_t comp, filter;
            if (!(_stream.ReadBe32u(_width) && _stream.ReadBe32u(_height) &&
                _stream.Read8u(_depth) && _stream.Read8u(_color) && _stream.Read8u(comp) &&
                _stream.Read8u(filter) && _stream.Read8u(_interlace)))
                return false;
            if (_width == 0 || _width > MAX_SIZE || _height == 0 || _height > MAX_SIZE)
                return false;
            if (_depth != 1 && _depth != 2 && _depth != 4 && _depth != 8 && _depth != 16)
                return false;
            if (_color > 6 || (_color == 3 && _depth == 16))
                return false;
            _paletteChannels = 0;
            if (_color == 3)
                _paletteChannels = 3;
            else if (_color & 1)
                return false;
            if (comp != 0 || filter != 0 || _interlace > 1)
                return false;
            if (!_paletteChannels)
            {
                _channels = (_color & 2 ? 3 : 1) + (_color & 4 ? 1 : 0);
                if ((1 << 30) / _width / _channels < _height)
                    return false;
            }
            else
            {
                _channels = 1;
                if ((1 << 30) / _width / 4 < _height)
                    return false;
            }
            return true;
        }

        bool ImagePngLoader::ReadPalette(const Chunk& chunk)
        {
            if (_first || chunk.size > 256 * 3)
                return false;
            size_t length = chunk.size / 3;
            if (length * 3 != chunk.size)
                return false;
            if (_stream.CanRead(chunk.size))
            {
                _palette.Resize(length * 4);
                _bgrToBgra(_stream.Current(), length, 1, length, _palette.data, _palette.size, 0xFF);
                _stream.Skip(chunk.size);
                return true;
            }
            else
                return false;
        }

        bool ImagePngLoader::ReadTransparency(const Chunk& chunk)
        {
            if (_first)
                return false;
            if (_idats.size())
                return false;
            if (_paletteChannels)
            {
                if (_palette.size == 0 || chunk.size > _palette.size || !_stream.CanRead(chunk.size))
                    return false;
                _paletteChannels = 4;
                for (size_t i = 0; i < chunk.size; ++i)
                    _palette.data[i * 4 + 3] = _stream.Current()[i];
                _stream.Skip(chunk.size);
            }
            else
            {
                if (!(_channels & 1) || chunk.size != _channels * 2)
                    return false;
                _hasTrans = true;
                for (size_t k = 0; k < _channels; ++k)
                    if (!_stream.ReadBe16u(_tc16[k]))
                        return false;
                if (_depth != 16)
                {
                    for (size_t k = 0; k < _channels; ++k)
                        _tc[k] = uint8_t(_tc16[k]) * png__depth_scale_table[_depth];
                }
            }
            return true;
        }

        bool ImagePngLoader::ReadData(const Chunk& chunk)
        {
            if (_first)
                return false;
            if (_paletteChannels && !_palette.size)
                return false;
            if (!_stream.CanRead(chunk.size))
                return false;
            _idats.push_back(chunk);
            _stream.Skip(chunk.size);
            return true;
        }

        uint8_t* ImagePngLoader::MergedData(uint32_t& size)
        {
            if (_idats.empty())
            {
                size = 0;
                return NULL;
            }
            else if (_idats.size() == 1)
            {
                size = _idats[0].size;
                return (uint8_t*)_stream.Data() + _idats[0].offs;
            }
            else
            {
                size = 0;
                for (size_t i = 0; i < _idats.size(); ++i)
                    size += _idats[i].size;
                _idat.Resize(size);
                for (size_t i = 0, offset = 0; i < _idats.size(); ++i)
                {
                    memcpy(_idat.data + offset, _stream.Data() + _idats[i].offs, _idats[i].size);
                    offset += _idats[i].size;
                }
                return _idat.data;
            }
        }
    }
}

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
#include "Simd/SimdImageSavePng.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
        SIMD_INLINE int PngError(const char* str, const char* stub)
        {
            std::cout << "PNG load error: " << str << ", " << stub << "!" << std::endl;
            return 0;
        }

        SIMD_INLINE uint8_t* PngErrorPtr(const char* str, const char* stub)
        {
            return (uint8_t*)(size_t)(PngError(str, stub) ? NULL : NULL);
        }

        namespace Zlib
        {
            const size_t ZFAST_BITS = 9;
            const size_t ZFAST_SIZE = 1 << ZFAST_BITS;
            const size_t ZFAST_MASK = ZFAST_SIZE - 1;

            struct Zhuffman
            {
                uint16_t fast[ZFAST_SIZE];
                uint16_t firstCode[16];
                int maxCode[17];
                uint16_t firstSymbol[16];
                uint8_t  size[288];
                uint16_t value[288];

                bool Build(const uint8_t* sizelist, int num)
                {
                    int i, k = 0;
                    int code, nextCode[16], sizes[17];

                    memset(sizes, 0, sizeof(sizes));
                    memset(fast, 0, sizeof(fast));
                    for (i = 0; i < num; ++i)
                        ++sizes[sizelist[i]];
                    sizes[0] = 0;
                    for (i = 1; i < 16; ++i)
                        if (sizes[i] > (1 << i))
                            return PngError("bad sizes", "Corrupt PNG");
                    code = 0;
                    for (i = 1; i < 16; ++i)
                    {
                        nextCode[i] = code;
                        firstCode[i] = (uint16_t)code;
                        firstSymbol[i] = (uint16_t)k;
                        code = (code + sizes[i]);
                        if (sizes[i] && code - 1 >= (1 << i))
                            return PngError("bad codelengths", "Corrupt PNG");
                        maxCode[i] = code << (16 - i); // preshift for inner loop
                        code <<= 1;
                        k += sizes[i];
                    }
                    maxCode[16] = 0x10000; // sentinel
                    for (i = 0; i < num; ++i)
                    {
                        int s = sizelist[i];
                        if (s)
                        {
                            int c = nextCode[s] - firstCode[s] + firstSymbol[s];
                            uint16_t fastv = (uint16_t)((s << 9) | i);
                            size[c] = (uint8_t)s;
                            value[c] = (uint16_t)i;
                            if (s <= (int)ZFAST_BITS)
                            {
                                int j = ZlibBitRev(nextCode[s], s);
                                while (j < (1 << ZFAST_BITS))
                                {
                                    fast[j] = fastv;
                                    j += (1 << s);
                                }
                            }
                            ++nextCode[s];
                        }
                    }
                    return 1;
                }
            };

            static SIMD_INLINE int BitRev16(int n)
            {
                n = ((n & 0xAAAA) >> 1) | ((n & 0x5555) << 1);
                n = ((n & 0xCCCC) >> 2) | ((n & 0x3333) << 2);
                n = ((n & 0xF0F0) >> 4) | ((n & 0x0F0F) << 4);
                n = ((n & 0xFF00) >> 8) | ((n & 0x00FF) << 8);
                return n;
            }

            static SIMD_INLINE int ZhuffmanDecode(InputMemoryStream& is, const Zhuffman& z)
            {
                int b, s;
                if (is.BitCount() < 16)
                {
                    if (is.Eof())
                        return -1;
                    is.FillBits();
                }
                b = z.fast[is.BitBuffer() & ZFAST_MASK];
                if (b)
                {
                    s = b >> 9;
                    is.BitBuffer() >>= s;
                    is.BitCount() -= s;
                    return b & 511;
                }
                else
                {
                    int k;
                    k = BitRev16((int)is.BitBuffer());
                    for (s = ZFAST_BITS + 1; k >= z.maxCode[s]; ++s);
                    if (s >= 16)
                        return -1;
                    b = (k >> (16 - s)) - z.firstCode[s] + z.firstSymbol[s];
                    if (b >= sizeof(z.size) || z.size[b] != s)
                        return -1;
                    is.BitBuffer() >>= s;
                    is.BitCount() -= s;
                    return z.value[b];
                }
            }

            static int ParseHuffmanBlock(InputMemoryStream& is, const Zhuffman& zLength, const Zhuffman& zDistance, OutputMemoryStream& os)
            {
                static const int zlengthBase[31] = { 3,4,5,6,7,8,9,10,11,13, 15,17,19,23,27,31,35,43,51,59, 67,83,99,115,131,163,195,227,258,0,0 };
                static const int zlengthExtra[31] = { 0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0,0,0 };
                static const int zdistBase[32] = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193, 257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577,0,0 };
                static const int zdistExtra[32] = { 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };

                SIMD_PERF_FUNC();

                uint8_t* beg = os.Data(), * dst = os.Current(), * end = beg + os.Capacity();
                for (;;)
                {
                    int z = ZhuffmanDecode(is, zLength);
                    if (z < 256)
                    {
                        if (z < 0)
                            return PngError("bad huffman code", "Corrupt PNG");
                        if (dst >= end)
                        {
                            os.Reserve(end - beg + 1);
                            beg = os.Data();
                            dst = os.Current();
                            end = beg + os.Capacity();
                        }
                        *dst++ = (uint8_t)z;
                    }
                    else
                    {
                        int len, dist;
                        if (z == 256)
                        {
                            os.Seek(dst - beg);
                            return 1;
                        }
                        z -= 257;
                        len = zlengthBase[z];
                        if (zlengthExtra[z])
                            len += (int)is.ReadBits(zlengthExtra[z]);
                        z = ZhuffmanDecode(is, zDistance);
                        if (z < 0)
                            return PngError("bad huffman code", "Corrupt PNG");
                        dist = zdistBase[z];
                        if (zdistExtra[z])
                            dist += (int)is.ReadBits(zdistExtra[z]);
                        if (dst - beg < dist)
                            return PngError("bad dist", "Corrupt PNG");
                        if (dst + len > end)
                        {
                            os.Reserve(dst - beg + len);
                            beg = os.Data();
                            dst = os.Current();
                            end = beg + os.Capacity();
                        }
                        if (dist == 1)
                        {
                            uint8_t val = dst[-dist];
                            if (len < 16)
                            {
                                while (len--)
                                    *dst++ = val;
                            }
                            else
                            {
                                memset(dst, val, len);
                                dst += len;
                            }
                        }
                        else
                        {
                            uint8_t* src = dst - dist;
                            if (dist < len || len < 16)
                            {
                                while(len--)
                                    *dst++ = *src++;
                            }
                            else
                            {
                                memcpy(dst, src, len);
                                dst += len;
                            }                        
                        }
                    }
                }
            }

            static int ComputeHuffmanCodes(InputMemoryStream& is, Zhuffman& zLength, Zhuffman& zDistance)
            {
                static const uint8_t length_dezigzag[19] = { 16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15 };
                Zhuffman z_codelength;
                uint8_t lencodes[286 + 32 + 137];
                uint8_t codelength_sizes[19];
                int i, n;

                int hlit = (int)is.ReadBits(5) + 257;
                int hdist = (int)is.ReadBits(5) + 1;
                int hclen = (int)is.ReadBits(4) + 4;
                int ntot = hlit + hdist;

                memset(codelength_sizes, 0, sizeof(codelength_sizes));
                for (i = 0; i < hclen; ++i)
                {
                    int s = (int)is.ReadBits(3);
                    codelength_sizes[length_dezigzag[i]] = (uint8_t)s;
                }
                if (!z_codelength.Build(codelength_sizes, 19))
                    return 0;
                n = 0;
                while (n < ntot)
                {
                    int c = ZhuffmanDecode(is, z_codelength);
                    if (c < 0 || c >= 19)
                        return PngError("bad codelengths", "Corrupt PNG");
                    if (c < 16)
                        lencodes[n++] = (uint8_t)c;
                    else
                    {
                        uint8_t fill = 0;
                        if (c == 16)
                        {
                            c = (int)is.ReadBits(2) + 3;
                            if (n == 0) return PngError("bad codelengths", "Corrupt PNG");
                            fill = lencodes[n - 1];
                        }
                        else if (c == 17)
                            c = (int)is.ReadBits(3) + 3;
                        else if (c == 18)
                            c = (int)is.ReadBits(7) + 11;
                        else
                            return PngError("bad codelengths", "Corrupt PNG");
                        if (ntot - n < c)
                            return PngError("bad codelengths", "Corrupt PNG");
                        memset(lencodes + n, fill, c);
                        n += c;
                    }
                }
                if (n != ntot)
                    return PngError("bad codelengths", "Corrupt PNG");
                if (!zLength.Build(lencodes, hlit))
                    return 0;
                if (!zDistance.Build(lencodes + hlit, hdist))
                    return 0;
                return 1;
            }

            static int ParseUncompressedBlock(InputMemoryStream& is, OutputMemoryStream& os)
            {
                is.ClearBits();
                uint16_t len, nlen;
                if (!is.Read16u(len) || !is.Read16u(nlen) || nlen != (len ^ 0xffff))
                    return PngError("zlib corrupt", "Corrupt PNG");
                if (!os.Write(is, len))
                    return PngError("read past buffer", "Corrupt PNG");
                return 1;
            }

            static int ParseHeader(InputMemoryStream& is)
            {
                uint8_t cmf, flg;
                if (!(is.Read8u(cmf) && is.Read8u(flg)))
                    return PngError("bad zlib header", "Corrupt PNG");
                if ((int(cmf) * 256 + flg) % 31 != 0)
                    return PngError("bad zlib header", "Corrupt PNG");
                if (flg & 32)
                    return PngError("no preset dict", "Corrupt PNG");
                if ((cmf & 15) != 8)
                    return PngError("bad compression", "Corrupt PNG");
                return 1;
            }

            bool Decode(InputMemoryStream& is, OutputMemoryStream& os, bool parseHeader)
            {
                static const uint8_t ZdefaultLength[288] = {
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
                static const uint8_t ZdefaultDistance[32] = {
                   5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5
                };

                Zhuffman zLength, zDistance;
                int final, type;
                if (parseHeader)
                {
                    if (!ParseHeader(is))
                        return false;
                }
                do
                {
                    final = (int)is.ReadBits(1);
                    type = (int)is.ReadBits(2);
                    if (type == 0)
                    {
                        if (!ParseUncompressedBlock(is, os))
                            return false;
                    }
                    else if (type == 3)
                        return false;
                    else
                    {
                        if (type == 1)
                        {
                            if (!zLength.Build(ZdefaultLength, 288))
                                return false;
                            if (!zDistance.Build(ZdefaultDistance, 32))
                                return false;
                        }
                        else
                        {
                            if (!ComputeHuffmanCodes(is, zLength, zDistance))
                                return false;
                        }
                        if (!ParseHuffmanBlock(is, zLength, zDistance, os))
                            return false;
                    }
                } while (!final);
                return true;
            }
        }

#define PNG_MALLOC(sz)           malloc(sz)
#define PNG_FREE(p)              free(p)

#define PNG__BYTECAST(x)  ((uint8_t) ((x) & 255))  // truncate int to byte without warnings

        static void* png__malloc(size_t size)
        {
            return PNG_MALLOC(size);
        }

        struct PngContext
        {
            uint32_t width, height;
            int channels, img_out_n;
        };

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

            good = (uint8_t*)png__malloc(req_comp * x * y * 1);
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

            if (req_comp == img_n)
                return data;
            assert(req_comp >= 1 && req_comp <= 4);

            good = (uint16_t*)png__malloc(req_comp * x * y * 2);
            if (good == NULL)
            {
                PNG_FREE(data);
                return (uint16_t*)PngErrorPtr("outofmem", "Out of memory");
            }

            for (j = 0; j < (int)y; ++j)
            {
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

        struct Png
        {
            PngContext* s;
            uint8_t * out;
            uint8_t depth;
        };

        enum 
        {
            PNG__F_none = 0,
            PNG__F_sub = 1,
            PNG__F_up = 2,
            PNG__F_avg = 3,
            PNG__F_paeth = 4,
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
        static int png__create_png_image_raw(Png* a, uint8_t* raw, uint32_t raw_len, int out_n, uint32_t x, uint32_t y, int depth, int color)
        {
            int bytes = (depth == 16 ? 2 : 1);
            PngContext* s = a->s;
            uint32_t i, j, stride = x * out_n * bytes;
            uint32_t img_len, img_width_bytes;
            int k;
            int img_n = s->channels; // copy it into a local for later

            int output_bytes = out_n * bytes;
            int filter_bytes = img_n * bytes;
            int width = x;

            assert(out_n == s->channels || out_n == s->channels + 1);
            a->out = (uint8_t*)png__malloc(x*y*output_bytes); // extra bytes to write off the end into
            if (!a->out) return PngError("outofmem", "Out of memory");

            if (!png__malloc(img_n* x*depth + 7)) return PngError("too large", "Corrupt PNG");
            img_width_bytes = (((img_n * x * depth) + 7) >> 3);
            img_len = (img_width_bytes + 1) * y;

            // we used to check for exact match between raw_len and img_len on non-interlaced PNGs,
            // but issue #276 reported a PNG in the wild that had extra data at the end (all zeros),
            // so just check for raw_len < img_len always.
            if (raw_len < img_len) 
                return PngError("not enough pixels", "Corrupt PNG");

            for (j = 0; j < y; ++j) 
            {
                uint8_t* cur = a->out + stride * j;
                uint8_t* prior;
                int filter = *raw++;

                if (filter > 4)
                    return PngError("invalid filter", "Corrupt PNG");

                if (depth < 8) 
                {
                    if (img_width_bytes > x) 
                        return PngError("invalid width", "Corrupt PNG");
                    cur += x * out_n - img_width_bytes; // store output to the rightmost img_len bytes, so we can decode in place
                    filter_bytes = 1;
                    width = img_width_bytes;
                }
                prior = cur - stride; // bugfix: need to compute this after 'cur +=' computation above

                // if first row, use special filter that doesn't sample previous row
                if (j == 0) filter = first_row_filter[filter];

                // handle first byte explicitly
                for (k = 0; k < filter_bytes; ++k) 
                {
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

                if (depth == 8) 
                {
                    if (img_n != out_n)
                        cur[img_n] = 255; // first pixel
                    raw += img_n;
                    cur += out_n;
                    prior += out_n;
                }
                else if (depth == 16) 
                {
                    if (img_n != out_n) 
                    {
                        cur[filter_bytes] = 255; // first pixel top byte
                        cur[filter_bytes + 1] = 255; // first pixel bottom byte
                    }
                    raw += filter_bytes;
                    cur += output_bytes;
                    prior += output_bytes;
                }
                else 
                {
                    raw += 1;
                    cur += 1;
                    prior += 1;
                }

                // this is a little gross, so that we don't switch per-pixel or per-component
                if (depth < 8 || img_n == out_n) 
                {
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
                else 
                {
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
                    if (depth == 16) 
                    {
                        cur = a->out + stride * j; // start at the beginning of the row again
                        for (i = 0; i < x; ++i, cur += output_bytes) 
                            cur[filter_bytes + 1] = 255;
                    }
                }
            }

            // we make a separate pass to expand bits to pixels; for performance,
            // this could run two scanlines behind the above code, so it won't
            // intefere with filtering but will still be in the cache.
            if (depth < 8)
            {
                for (j = 0; j < y; ++j)
                {
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

                    if (depth == 4) 
                    {
                        for (k = x * img_n; k >= 2; k -= 2, ++in) 
                        {
                            *cur++ = scale * ((*in >> 4));
                            *cur++ = scale * ((*in) & 0x0f);
                        }
                        if (k > 0) 
                            *cur++ = scale * ((*in >> 4));
                    }
                    else if (depth == 2) 
                    {
                        for (k = x * img_n; k >= 4; k -= 4, ++in) 
                        {
                            *cur++ = scale * ((*in >> 6));
                            *cur++ = scale * ((*in >> 4) & 0x03);
                            *cur++ = scale * ((*in >> 2) & 0x03);
                            *cur++ = scale * ((*in) & 0x03);
                        }
                        if (k > 0) 
                            *cur++ = scale * ((*in >> 6));
                        if (k > 1) 
                            *cur++ = scale * ((*in >> 4) & 0x03);
                        if (k > 2) 
                            *cur++ = scale * ((*in >> 2) & 0x03);
                    }
                    else if (depth == 1)
                    {
                        for (k = x * img_n; k >= 8; k -= 8, ++in) 
                        {
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
                    if (img_n != out_n) 
                    {
                        int q;
                        // insert alpha = 255
                        cur = a->out + stride * j;
                        if (img_n == 1) 
                        {
                            for (q = x - 1; q >= 0; --q)
                            {
                                cur[q * 2 + 1] = 255;
                                cur[q * 2 + 0] = cur[q];
                            }
                        }
                        else
                        {
                            assert(img_n == 3);
                            for (q = x - 1; q >= 0; --q) 
                            {
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

        static int png__create_png_image(Png* a, uint8_t* image_data, uint32_t image_data_len, int out_n, int depth, int color, int interlaced)
        {
            int bytes = (depth == 16 ? 2 : 1);
            int out_bytes = out_n * bytes;
            uint8_t* final;
            int p;
            if (!interlaced)
                return png__create_png_image_raw(a, image_data, image_data_len, out_n, a->s->width, a->s->height, depth, color);

            // de-interlacing
            final = (uint8_t*)png__malloc(a->s->width * a->s->height * out_bytes);
            for (p = 0; p < 7; ++p) 
            {
                int xorig[] = { 0,4,0,2,0,1,0 };
                int yorig[] = { 0,0,4,0,2,0,1 };
                int xspc[] = { 8,8,4,4,2,2,1 };
                int yspc[] = { 8,8,8,4,4,2,2 };
                int i, j, x, y;
                // pass1_x[4] = 0, pass1_x[5] = 1, pass1_x[12] = 1
                x = (a->s->width - xorig[p] + xspc[p] - 1) / xspc[p];
                y = (a->s->height - yorig[p] + yspc[p] - 1) / yspc[p];
                if (x && y) 
                {
                    uint32_t img_len = ((((a->s->channels * x * depth) + 7) >> 3) + 1) * y;
                    if (!png__create_png_image_raw(a, image_data, image_data_len, out_n, x, y, depth, color))
                    {
                        PNG_FREE(final);
                        return 0;
                    }
                    for (j = 0; j < y; ++j) 
                    {
                        for (i = 0; i < x; ++i) 
                        {
                            int out_y = j * yspc[p] + yorig[p];
                            int out_x = i * xspc[p] + xorig[p];
                            memcpy(final + out_y * a->s->width * out_bytes + out_x * out_bytes,
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

        static int png__compute_transparency(Png* z, uint8_t tc[3], int out_n)
        {
            PngContext* s = z->s;
            uint32_t i, pixel_count = s->width * s->height;
            uint8_t* p = z->out;

            // compute color-based transparency, assuming we've
            // already got 255 as the alpha value in the output
            assert(out_n == 2 || out_n == 4);

            if (out_n == 2) 
            {
                for (i = 0; i < pixel_count; ++i) 
                {
                    p[1] = (p[0] == tc[0] ? 0 : 255);
                    p += 2;
                }
            }
            else 
            {
                for (i = 0; i < pixel_count; ++i) 
                {
                    if (p[0] == tc[0] && p[1] == tc[1] && p[2] == tc[2])
                        p[3] = 0;
                    p += 4;
                }
            }
            return 1;
        }

        static int png__compute_transparency16(Png* z, uint16_t tc[3], int out_n)
        {
            PngContext* s = z->s;
            uint32_t i, pixel_count = s->width * s->height;
            uint16_t* p = (uint16_t*)z->out;

            // compute color-based transparency, assuming we've
            // already got 65535 as the alpha value in the output
            assert(out_n == 2 || out_n == 4);

            if (out_n == 2) 
            {
                for (i = 0; i < pixel_count; ++i)
                {
                    p[1] = (p[0] == tc[0] ? 0 : 65535);
                    p += 2;
                }
            }
            else 
            {
                for (i = 0; i < pixel_count; ++i)
                {
                    if (p[0] == tc[0] && p[1] == tc[1] && p[2] == tc[2])
                        p[3] = 0;
                    p += 4;
                }
            }
            return 1;
        }

        static int png__expand_png_palette(Png* a, uint8_t* palette, int len, int pal_img_n)
        {
            uint32_t i, pixel_count = a->s->width * a->s->height;
            uint8_t* p, * temp_out, * orig = a->out;

            p = (uint8_t*)png__malloc(pixel_count*pal_img_n);
            if (p == NULL) 
                return PngError("outofmem", "Out of memory");

            // between here and free(out) below, exitting would leak
            temp_out = p;

            if (pal_img_n == 3) 
            {
                for (i = 0; i < pixel_count; ++i) 
                {
                    int n = orig[i] * 4;
                    p[0] = palette[n];
                    p[1] = palette[n + 1];
                    p[2] = palette[n + 2];
                    p += 3;
                }
            }
            else 
            {
                for (i = 0; i < pixel_count; ++i) 
                {
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
            Png p;
            p.s = &context;
            Png* z = &p;

            PngContext* s = z->s;

            z->out = NULL;

            if (!ParseFile())
                return false;

            s->width = _width;
            s->height = _height;
            s->channels = _channels;
            z->depth = _depth;

            InputMemoryStream zSrc = MergedDataStream();
            OutputMemoryStream zDst(AlignHi(size_t(_width) * _depth, 8) * _height * _channels + _height);
            if(!Zlib::Decode(zSrc, zDst, !_iPhone))
                return false;

            if ((req_comp == s->channels + 1 && req_comp != 3 && !_paletteChannels) || _hasTrans)
                s->img_out_n = s->channels + 1;
            else
                s->img_out_n = s->channels;
            if (!png__create_png_image(z, zDst.Data(), (int)zDst.Size(), s->img_out_n, z->depth, _color, _interlace))
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
                s->channels = _paletteChannels; // record the actual colors we had
                s->img_out_n = _paletteChannels;
                if (req_comp >= 3) 
                    s->img_out_n = req_comp;
                if (!png__expand_png_palette(z, _palette.data, (int)_palette.size, s->img_out_n))
                    return false;
            }
            else if (_hasTrans)
                ++s->channels;

            if (!(p.depth <= 8 || p.depth == 16))
                return false;
            uint8_t* data = p.out;
            p.out = NULL;
            if (req_comp && req_comp != p.s->img_out_n)
            {
                if (p.depth <= 8)
                    data = png__convert_format((uint8_t*)data, p.s->img_out_n, req_comp, _width, _height);
                else
                    data = (uint8_t*)png__convert_format16((uint16_t*)data, p.s->img_out_n, req_comp, _width, _height);
                p.s->img_out_n = req_comp;
                if (data == NULL)
                    return false;
            }
            if (p.depth == 16)
            {
                size_t size = context.width * context.height * req_comp;
                const uint16_t* src = (uint16_t*)data;
                uint8_t* dst = (uint8_t*)PNG_MALLOC(size);
                for (size_t i = 0; i < size; ++i)
                    dst[i] = uint8_t(src[i] >> 8);
                PNG_FREE(data);
                data = dst;
            }
            PNG_FREE(p.out);
            if (data)
            {
                size_t stride = 4 * context.width;
                _image.Recreate(context.width, context.height, (Image::Format)_param.format);
                switch (_param.format)
                {
                case SimdPixelFormatGray8:
                    Base::RgbaToGray(data, context.width, context.height, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatBgr24:
                    Base::BgraToRgb(data, context.width, context.height, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatBgra32:
                    Base::BgraToRgba(data, context.width, context.height, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatRgb24:
                    Base::BgraToBgr(data, context.width, context.height, stride, _image.data, _image.stride);
                    break;
                case SimdPixelFormatRgba32:
                    Base::Copy(data, stride, context.width, context.height, 4, _image.data, _image.stride);
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
            return _idats.size() != 0;
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

        InputMemoryStream ImagePngLoader::MergedDataStream()
        {
            if (_idats.size() == 1)
                return InputMemoryStream((uint8_t*)_stream.Data() + _idats[0].offs, _idats[0].size);
            else
            {
                size_t size = 0;
                for (size_t i = 0; i < _idats.size(); ++i)
                    size += _idats[i].size;
                _idat.Resize(size);
                for (size_t i = 0, offset = 0; i < _idats.size(); ++i)
                {
                    memcpy(_idat.data + offset, _stream.Data() + _idats[i].offs, _idats[i].size);
                    offset += _idats[i].size;
                }
                return InputMemoryStream(_idat.data, _idat.size);
            }
        }
    }
}

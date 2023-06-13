/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar,
*               2022-2022 Fabien Spindler.
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
#include "Simd/SimdImageLoadPng.h"
#include "Simd/SimdImageSavePng.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"

namespace Simd
{
    namespace Base
    {
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
                            return CorruptPngError("bad sizes");
                    code = 0;
                    for (i = 1; i < 16; ++i)
                    {
                        nextCode[i] = code;
                        firstCode[i] = (uint16_t)code;
                        firstSymbol[i] = (uint16_t)k;
                        code = (code + sizes[i]);
                        if (sizes[i] && code - 1 >= (1 << i))
                            return CorruptPngError("bad codelengths");
                        maxCode[i] = code << (16 - i);
                        code <<= 1;
                        k += sizes[i];
                    }
                    maxCode[16] = 0x10000;
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
                            return CorruptPngError("bad huffman code");
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
                            return CorruptPngError("bad huffman code");
                        dist = zdistBase[z];
                        if (zdistExtra[z])
                            dist += (int)is.ReadBits(zdistExtra[z]);
                        if (dst - beg < dist)
                            return CorruptPngError("bad dist");
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
                        return CorruptPngError("bad codelengths");
                    if (c < 16)
                        lencodes[n++] = (uint8_t)c;
                    else
                    {
                        uint8_t fill = 0;
                        if (c == 16)
                        {
                            c = (int)is.ReadBits(2) + 3;
                            if (n == 0) return CorruptPngError("bad codelengths");
                            fill = lencodes[n - 1];
                        }
                        else if (c == 17)
                            c = (int)is.ReadBits(3) + 3;
                        else if (c == 18)
                            c = (int)is.ReadBits(7) + 11;
                        else
                            return CorruptPngError("bad codelengths");
                        if (ntot - n < c)
                            return CorruptPngError("bad codelengths");
                        memset(lencodes + n, fill, c);
                        n += c;
                    }
                }
                if (n != ntot)
                    return CorruptPngError("bad codelengths");
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
                    return CorruptPngError("zlib corrupt");
                if (!os.Write(is, len))
                    return CorruptPngError("read past buffer");
                return 1;
            }

            static int ParseHeader(InputMemoryStream& is)
            {
                uint8_t cmf, flg;
                if (!(is.Read8u(cmf) && is.Read8u(flg)))
                    return CorruptPngError("bad zlib header");
                if ((int(cmf) * 256 + flg) % 31 != 0)
                    return CorruptPngError("bad zlib header");
                if (flg & 32)
                    return CorruptPngError("no preset dict");
                if ((cmf & 15) != 8)
                    return CorruptPngError("bad compression");
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

        //-------------------------------------------------------------------------------------------------

        static const uint8_t DepthScaleTable[9] = { 0, 0xff, 0x55, 0, 0x11, 0,0,0, 0x01 };

        static void DecodeLine0(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
                memcpy(dst, curr, width * srcN);
            else
            {
                for (int x = 0; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i];
                    for (; i < dstN; ++i)
                        dst[i] = 0xFF;
                    curr += srcN;
                    dst += dstN;
                }
            }
        }

        static void DecodeLine1(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                for (int i = 0; i < srcN; ++i)
                    dst[i] = curr[i];
                for (int i = srcN, n = srcN * width; i < n; ++i)
                    dst[i] = curr[i] + dst[i - dstN];
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i];
                for (; i < dstN; ++i)
                    dst[i] = 0xFF;
                curr += srcN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + dst[i - dstN];
                    for (; i < dstN; ++i)
                        dst[i] = 0xFF;
                    curr += srcN;
                    dst += dstN;
                }
            }
        }

        static void DecodeLine2(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                for (int i = 0, n = srcN * width; i < n; ++i)
                    dst[i] = curr[i] + prev[i];
            }
            else
            {
                for (int x = 0; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + prev[i];
                    for (; i < dstN; ++i)
                        dst[i] = 0xFF;
                    curr += srcN;
                    prev += dstN;
                    dst += dstN;
                }
            }
        }

        static void DecodeLine3(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                for (int i = 0; i < srcN; ++i)
                    dst[i] = curr[i] + (prev[i] >> 1);
                for (int i = srcN, n = srcN * width; i < n; ++i)
                    dst[i] = curr[i] + ((prev[i] + dst[i - dstN]) >> 1);
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i] + (prev[i] >> 1);
                for (; i < dstN; ++i)
                    dst[i] = 0xFF;
                curr += srcN;
                prev += dstN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + ((prev[i] + dst[i - dstN]) >> 1);
                    for (; i < dstN; ++i)
                        dst[i] = 0xFF;
                    curr += srcN;
                    prev += dstN;                    
                    dst += dstN;
                }
            }
        }

        static void DecodeLine4(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                for (int i = 0; i < srcN; ++i)
                    dst[i] = curr[i] + Paeth(0, prev[i], 0);
                for (int i = srcN, n = srcN * width; i < n; ++i)
                    dst[i] = curr[i] + Paeth(dst[i - dstN], prev[i], prev[i - dstN]);
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i] + Paeth(0, prev[i], 0);
                for (; i < dstN; ++i)
                    dst[i] = 0xFF;
                curr += srcN;
                prev += dstN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + Paeth(dst[i - dstN], prev[i], prev[i - dstN]);
                    for (; i < dstN; ++i)
                        dst[i] = 0xFF;
                    curr += srcN;
                    prev += dstN;
                    dst += dstN;
                }
            }
        }

        static void DecodeLine5(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                for (int i = 0; i < srcN; ++i)
                    dst[i] = curr[i];
                for (int i = srcN, n = srcN * width; i < n; ++i)
                    dst[i] = curr[i] + (dst[i - dstN] >> 1);
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i];
                for (; i < dstN; ++i)
                    dst[i] = 0xFF;
                curr += srcN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + (dst[i - dstN] >> 1);;
                    for (; i < dstN; ++i)
                        dst[i] = 0xFF;
                    curr += srcN;
                    dst += dstN;
                }
            }
        }

        static void DecodeLine6(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                for (int i = 0; i < srcN; ++i)
                    dst[i] = curr[i];
                for (int i = srcN, n = srcN * width; i < n; ++i)
                    dst[i] = curr[i] + Paeth(dst[i - dstN], 0, 0);
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i];
                for (; i < dstN; ++i)
                    dst[i] = 0xFF;
                curr += srcN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + Paeth(dst[i - dstN], 0, 0);
                    for (; i < dstN; ++i)
                        dst[i] = 0xFF;
                    curr += srcN;
                    dst += dstN;
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<class T> void ComputeTransparency(T * dst, size_t size, size_t outN, T tc[3])
        {
            if (outN == 2)
            {
                for (size_t i = 0; i < size; ++i)
                {
                    dst[1] = (dst[0] == tc[0] ? 0 : std::numeric_limits<T>::max());
                    dst += 2;
                }
            }
            else if (outN == 4)
            {
                for (size_t i = 0; i < size; ++i)
                {
                    if (dst[0] == tc[0] && dst[1] == tc[1] && dst[2] == tc[2])
                        dst[3] = 0;
                    dst += 4;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        static void ExpandPalette(const uint8_t* src, size_t size, int outN, const uint8_t* palette, uint8_t* dst)
        {
            if (outN == 3)
            {
                for (size_t i = 0; i < size; ++i)
                {
                    int n = src[i] * 4;
                    dst[0] = palette[n];
                    dst[1] = palette[n + 1];
                    dst[2] = palette[n + 2];
                    dst += 3;
                }
            }
            else if (outN == 4)
            {
                for (size_t i = 0; i < size; ++i)
                {
                    int n = src[i] * 4;
                    dst[0] = palette[n];
                    dst[1] = palette[n + 1];
                    dst[2] = palette[n + 2];
                    dst[3] = palette[n + 3];
                    dst += 4;
                }
            }
            else
                assert(0);
        }

        //-------------------------------------------------------------------------------------------------

        template<class T> SIMD_INLINE uint8_t Convert(int r, int g, int b)
        {
            return (uint8_t)(((r * 77) + (g * 150) + (29 * b)) >> (sizeof(T) * 8));
        };

        template<class T> uint8_t Convert(int g);

        template<> SIMD_INLINE uint8_t Convert<uint8_t>(int g)
        {
            return (uint8_t)(g);
        };

        template<> SIMD_INLINE uint8_t Convert<uint16_t>(int g)
        {
            return (uint8_t)(g >> 8);
        };

        template<class T, int channels, SimdPixelFormatType format> SIMD_INLINE void ConvertPixel(const T* src, uint8_t* dst)
        {
            if ((channels == 1 || channels == 2) && format == SimdPixelFormatGray8) 
            { 
                dst[0] = Convert<T>(src[0]); 
            }
            else if ((channels == 1 || channels == 2) && (format == SimdPixelFormatBgr24 || format == SimdPixelFormatRgb24))
            {
                uint8_t gray = Convert<T>(src[0]);
                dst[0] = gray;
                dst[1] = gray;
                dst[2] = gray;
            }
            else if (channels == 1 && (format == SimdPixelFormatBgra32 || format == SimdPixelFormatRgba32))
            {
                uint8_t gray = Convert<T>(src[0]);
                dst[0] = Convert<T>(src[0]);
                dst[1] = gray;
                dst[2] = gray;
                dst[3] = 0xFF;
            }
            else if (channels == 2 && (format == SimdPixelFormatBgra32 || format == SimdPixelFormatRgba32))
            {
                uint8_t gray = Convert<T>(src[0]);
                dst[0] = Convert<T>(src[0]);
                dst[1] = gray;
                dst[2] = gray;
                dst[3] = Convert<T>(src[1]);
            }
            else if ((channels == 3 || channels == 4) && format == SimdPixelFormatGray8)
            {
                dst[0] = Convert<T>(src[0], src[1], src[2]);
            }
            else if ((channels == 3 || channels == 4) && format == SimdPixelFormatBgr24)
            {
                dst[0] = Convert<T>(src[2]);
                dst[1] = Convert<T>(src[1]);
                dst[2] = Convert<T>(src[0]);
            }
            else if ((channels == 3 || channels == 4) && format == SimdPixelFormatRgb24)
            {
                dst[0] = Convert<T>(src[0]);
                dst[1] = Convert<T>(src[1]);
                dst[2] = Convert<T>(src[2]);
            }
            else if (channels == 3 && format == SimdPixelFormatBgra32)
            {
                dst[0] = Convert<T>(src[2]);
                dst[1] = Convert<T>(src[1]);
                dst[2] = Convert<T>(src[0]);
                dst[3] = 0xFF;
            }
            else if (channels == 3 && format == SimdPixelFormatRgba32)
            {
                dst[0] = Convert<T>(src[0]);
                dst[1] = Convert<T>(src[1]);
                dst[2] = Convert<T>(src[2]);
                dst[3] = 0xFF;
            }
            else if (channels == 4 && format == SimdPixelFormatBgra32)
            {
                dst[0] = Convert<T>(src[2]);
                dst[1] = Convert<T>(src[1]);
                dst[2] = Convert<T>(src[0]);
                dst[3] = Convert<T>(src[3]);
            }
            else if (channels == 4 && format == SimdPixelFormatRgba32)
            {
                dst[0] = Convert<T>(src[0]);
                dst[1] = Convert<T>(src[1]);
                dst[2] = Convert<T>(src[2]);
                dst[3] = Convert<T>(src[3]);
            }
            else 
                assert(0);
        }

        template<class T, int channels, SimdPixelFormatType format> void ConvertFormat(const T* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            typedef Simd::View<Simd::Allocator> Image;
            size_t count = Image::ChannelCount((Image::Format)format);
            size_t srcGap = srcStride - channels * width;
            size_t dstGap = dstStride - count * width;
            for (size_t y = 0; y < height; ++y)
            {
                for (size_t x = 0; x < width; ++x)
                {
                    ConvertPixel<T, channels, format>(src, dst);
                    src += channels;
                    dst += count;
                }
                src += srcGap;
                dst += dstGap;
            }
        }

        template<class T, int channels> ImagePngLoader::ConverterPtr GetConverter(SimdPixelFormatType format)
        {
            switch (format)
            {
            case SimdPixelFormatGray8: return (ImagePngLoader::ConverterPtr)ConvertFormat<T, channels, SimdPixelFormatGray8>;
            case SimdPixelFormatBgr24: return (ImagePngLoader::ConverterPtr)ConvertFormat<T, channels, SimdPixelFormatBgr24>;
            case SimdPixelFormatBgra32: return (ImagePngLoader::ConverterPtr)ConvertFormat<T, channels, SimdPixelFormatBgra32>;
            case SimdPixelFormatRgb24: return (ImagePngLoader::ConverterPtr)ConvertFormat<T, channels, SimdPixelFormatRgb24>;
            case SimdPixelFormatRgba32: return (ImagePngLoader::ConverterPtr)ConvertFormat<T, channels, SimdPixelFormatRgba32>;
            default:
                assert(0);
                return NULL;
            }
        }

        template<class T> ImagePngLoader::ConverterPtr GetConverter(int channels, SimdPixelFormatType format)
        {
            switch (channels)
            {
            case 1: return GetConverter<T, 1>(format);
            case 2: return GetConverter<T, 2>(format);
            case 3: return GetConverter<T, 3>(format);
            case 4: return GetConverter<T, 4>(format);
            default:
                assert(0);
                return NULL;
            }
        }

        static ImagePngLoader::ConverterPtr GetConverter(int depth, int channels, SimdPixelFormatType format)
        {
            if (depth <= 8)
                return GetConverter<uint8_t>(channels, format);
            else if(depth == 16)
                return GetConverter<uint16_t>(channels, format);
            assert(0);
            return NULL;
        }

        //-------------------------------------------------------------------------------------------------

        ImagePngLoader::ImagePngLoader(const ImageLoaderParam& param)
            : ImageLoader(param)
            , _converter(NULL)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgba32;
            _decodeLine[0] = Base::DecodeLine0;
            _decodeLine[1] = Base::DecodeLine1;
            _decodeLine[2] = Base::DecodeLine2;
            _decodeLine[3] = Base::DecodeLine3;
            _decodeLine[4] = Base::DecodeLine4;
            _decodeLine[5] = Base::DecodeLine5;
            _decodeLine[6] = Base::DecodeLine6;
            _expandPalette = Base::ExpandPalette;
        }

        void ImagePngLoader::SetConverter()
        {
            _converter = GetConverter(_depth, _outN, _param.format);
        }

#ifdef SIMD_CPP_2011_ENABLE
        SIMD_INLINE constexpr uint32_t ChunkType(char a, char b, char c, char d)
#else
        SIMD_INLINE uint32_t ChunkType(char a, char b, char c, char d)
#endif
        {
            return ((uint32_t(a) << 24) + (uint32_t(b) << 16) + (uint32_t(c) << 8) + uint32_t(d));
        }

        bool ImagePngLoader::FromStream()
        {
            SIMD_PERF_FUNC();

            if (!ParseFile())
                return false;

            InputMemoryStream zSrc = MergedDataStream();
            OutputMemoryStream zDst(AlignHi(size_t(_width) * _depth, 8) * _height * _channels + _height);
            if(!Zlib::Decode(zSrc, zDst, !_iPhone))
                return false;

            if (!CreateImage(zDst.Data(), zDst.Size()))
                return false;

            if (_hasTrans) 
            {
                if (_depth == 16)
                    ComputeTransparency((uint16_t*)_buffer.data, _width * _height, _outN, _tc16);
                else
                    ComputeTransparency(_buffer.data, _width * _height, _outN, _tc);
            }

            ExpandPalette();

            ConvertImage();

            return true;
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
            int reqN = 4;
            if (Image::ChannelCount((Image::Format)_param.format) == _channels && _depth != 16)
                reqN = _channels;
            else
                reqN = 4;
            if ((reqN == _channels + 1 && reqN != 3 && !_paletteChannels) || _hasTrans)
                _outN = _channels + 1;
            else
                _outN = _channels;
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
                BgrToBgra(_stream.Current(), length, 1, length, _palette.data, _palette.size, 0xFF);
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
                        _tc[k] = uint8_t(_tc16[k]) * DepthScaleTable[_depth];
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

        bool ImagePngLoader::CreateImage(const uint8_t* data, size_t size)
        {
            SIMD_PERF_FUNC();

            int outS = _outN * (_depth == 16 ? 2 : 1);
            if (!_interlace)
                return CreateImageRaw(data, (int)size, _width, _height);
            Array8u buf(_width * _height * outS);
            for (int p = 0; p < 7; ++p)
            {
                static const int xorig[] = { 0,4,0,2,0,1,0 };
                static const int yorig[] = { 0,0,4,0,2,0,1 };
                static const int xspc[] = { 8,8,4,4,2,2,1 };
                static const int yspc[] = { 8,8,8,4,4,2,2 };
                int i, j, x, y;
                x = (_width - xorig[p] + xspc[p] - 1) / xspc[p];
                y = (_height - yorig[p] + yspc[p] - 1) / yspc[p];
                if (x && y)
                {
                    uint32_t img_len = ((((_channels * x * _depth) + 7) >> 3) + 1) * y;
                    if (!CreateImageRaw(data, (int)size, x, y))
                        return false;
                    for (j = 0; j < y; ++j)
                    {
                        for (i = 0; i < x; ++i)
                        {
                            int out_y = j * yspc[p] + yorig[p];
                            int out_x = i * xspc[p] + xorig[p];
                            memcpy(buf.data + out_y * _width * outS + out_x * outS, _buffer.data + (j * x + i) * outS, outS);
                        }
                    }
                    data += img_len;
                    size -= img_len;
                }
            }
            _buffer.Swap(buf);
            return true;
        }

        bool ImagePngLoader::CreateImageRaw(const uint8_t* data, uint32_t size, uint32_t width, uint32_t height)
        {
            static const uint8_t FirstRowFilter[5] = { 0, 1, 0, 5, 6 };
            int bytes = (_depth == 16 ? 2 : 1);
            uint32_t i, j, stride = width * _outN * bytes;
            uint32_t img_len, img_width_bytes;
            int k;
            int width_ = width;

            int output_bytes = _outN * bytes;
            int filter_bytes = _channels * bytes;

            assert(_outN == _channels || _outN == _channels + 1);

            _buffer.Resize(width * height * output_bytes);
            if (_buffer.Empty())
                return PngLoadError("outofmem", "Out of memory");

            img_width_bytes = (_channels * width * _depth + 7) >> 3;
            img_len = (img_width_bytes + 1) * height;

            if (size < img_len)
                return CorruptPngError("not enough pixels");

            for (j = 0; j < height; ++j)
            {
                uint8_t* cur = _buffer.data + stride * j;
                uint8_t* prior;
                int filter = *data++;

                if (filter > 4)
                    return CorruptPngError("invalid filter");

                if (_depth < 8)
                {
                    if (img_width_bytes > width)
                        return CorruptPngError("invalid width");
                    cur += width * _outN - img_width_bytes; // store output to the rightmost img_len bytes, so we can decode in place
                    filter_bytes = 1;
                    width_ = img_width_bytes;
                }
                prior = cur - stride;
                if (j == 0)
                    filter = FirstRowFilter[filter];

                int size = (_depth < 8 || _channels == _outN ? width_ : width);
                int dstN = _depth < 8 || _channels == _outN ? filter_bytes : output_bytes;
                _decodeLine[filter](data, cur - stride, size, filter_bytes, dstN, cur);
                data += size * filter_bytes;
            }
            if (_depth < 8)
            {
                for (j = 0; j < height; ++j)
                {
                    uint8_t* cur = _buffer.data + stride * j;
                    const uint8_t* in = _buffer.data + stride * j + width * _outN - img_width_bytes;
                    uint8_t scale = (_color == 0) ? DepthScaleTable[_depth] : 1;
                    if (_depth == 4)
                    {
                        for (k = width * _channels; k >= 2; k -= 2, ++in)
                        {
                            *cur++ = scale * ((*in >> 4));
                            *cur++ = scale * ((*in) & 0x0f);
                        }
                        if (k > 0)
                            *cur++ = scale * ((*in >> 4));
                    }
                    else if (_depth == 2)
                    {
                        for (k = width * _channels; k >= 4; k -= 4, ++in)
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
                    else if (_depth == 1)
                    {
                        for (k = width * _channels; k >= 8; k -= 8, ++in)
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
                    if (_channels != _outN)
                    {
                        int q;
                        cur = _buffer.data + stride * j;
                        if (_channels == 1)
                        {
                            for (q = width - 1; q >= 0; --q)
                            {
                                cur[q * 2 + 1] = 255;
                                cur[q * 2 + 0] = cur[q];
                            }
                        }
                        else
                        {
                            assert(_channels == 3);
                            for (q = width - 1; q >= 0; --q)
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
            else if (_depth == 16)
            {
                uint8_t* cur = _buffer.data;
                uint16_t* cur16 = (uint16_t*)cur;
                for (i = 0; i < width * height * _outN; ++i, cur16++, cur += 2)
                    *cur16 = (cur[0] << 8) | cur[1];
            }
            return 1;
        }

        void ImagePngLoader::ExpandPalette()
        {
            if (_paletteChannels)
            {
                _outN = Max(_paletteChannels, _outN);
                Array8u buf(_width * _height * _outN);
                _expandPalette(_buffer.data, _width * _height, _outN, _palette.data, buf.data);
                _buffer.Swap(buf);
            }
        }

        void ImagePngLoader::ConvertImage()
        {
            SIMD_PERF_FUNC();
            SetConverter();
            _image.Recreate(_width, _height, (Image::Format)_param.format);
            _converter(_buffer.data, _width, _height, _width * _outN, _image.data, _image.stride);
        }
    }
}

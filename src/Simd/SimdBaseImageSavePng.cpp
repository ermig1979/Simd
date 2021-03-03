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
#include "Simd/SimdMemory.h"
#include "Simd/SimdImageSave.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdPerformance.h"

namespace Simd
{
    namespace Base
    {
#ifndef PNG_MALLOC
#define PNG_MALLOC(sz)        malloc(sz)
#define PNG_REALLOC(p,newsz)  realloc(p,newsz)
#define PNG_FREE(p)           free(p)
#endif

#ifndef PNG_REALLOC_SIZED
#define PNG_REALLOC_SIZED(p,oldsz,newsz) PNG_REALLOC(p,newsz)
#endif

#ifndef PNG_MEMMOVE
#define PNG_MEMMOVE(a,b,sz) memmove(a,b,sz)
#endif

#define PNG_UCHAR(x) (unsigned char) ((x) & 0xff)

#define png__sbraw(a) ((int *) (void *) (a) - 2)
#define png__sbm(a)   png__sbraw(a)[0]
#define png__sbn(a)   png__sbraw(a)[1]

#define png__sbneedgrow(a,n)  ((a)==0 || png__sbn(a)+n >= png__sbm(a))
#define png__sbmaybegrow(a,n) (png__sbneedgrow(a,(n)) ? png__sbgrow(a,n) : 0)
#define png__sbgrow(a,n)  png__sbgrowf((void **) &(a), (n), sizeof(*(a)))

#define png__sbpush(a, v)      (png__sbmaybegrow(a,1), (a)[png__sbn(a)++] = (v))
#define png__sbcount(a)        ((a) ? png__sbn(a) : 0)
#define png__sbfree(a)         ((a) ? PNG_FREE(png__sbraw(a)),0 : 0)

        static void* png__sbgrowf(void** arr, int increment, int itemsize)
        {
            int m = *arr ? 2 * png__sbm(*arr) + increment : increment + 1;
            void* p = PNG_REALLOC_SIZED(*arr ? png__sbraw(*arr) : 0, *arr ? (png__sbm(*arr) * itemsize + sizeof(int) * 2) : 0, itemsize * m + sizeof(int) * 2);
            assert(p);
            if (p) {
                if (!*arr) ((int*)p)[1] = 0;
                *arr = (void*)((int*)p + 2);
                png__sbm(*arr) = m;
            }
            return *arr;
        }

        static unsigned char* png__zlib_flushf(unsigned char* data, unsigned int* bitbuffer, int* bitcount)
        {
            while (*bitcount >= 8) {
                png__sbpush(data, PNG_UCHAR(*bitbuffer));
                *bitbuffer >>= 8;
                *bitcount -= 8;
            }
            return data;
        }

        SIMD_INLINE int ZlibBitrev(int code, int codebits)
        {
            int res = 0;
            while (codebits--)
            {
                res = (res << 1) | (code & 1);
                code >>= 1;
            }
            return res;
        }

        SIMD_INLINE int ZlibCount(const uint8_t* a, const uint8_t* b, int limit)
        {
            int i = 0;
            for (; i < limit && i < 258; ++i)
                if (a[i] != b[i]) break;
            return i;
        }

        SIMD_INLINE uint32_t ZlibHash(const uint8_t* data)
        {
            uint32_t hash = data[0] + (data[1] << 8) + (data[2] << 16);
            hash ^= hash << 3;
            hash += hash >> 5;
            hash ^= hash << 4;
            hash += hash >> 17;
            hash ^= hash << 25;
            hash += hash >> 6;
            return hash;
        }

#define png__zlib_flush() (out = png__zlib_flushf(out, &bitbuf, &bitcount))
#define png__zlib_add(code,codebits) \
      (bitbuf |= (code) << bitcount, bitcount += (codebits), png__zlib_flush())
#define png__zlib_huffa(b,c)  png__zlib_add(ZlibBitrev(b,c),c)
        // default huffman tables
#define png__zlib_huff1(n)  png__zlib_huffa(0x30 + (n), 8)
#define png__zlib_huff2(n)  png__zlib_huffa(0x190 + (n)-144, 9)
#define png__zlib_huff3(n)  png__zlib_huffa(0 + (n)-256,7)
#define png__zlib_huff4(n)  png__zlib_huffa(0xc0 + (n)-280,8)
#define png__zlib_huff(n)  ((n) <= 143 ? png__zlib_huff1(n) : (n) <= 255 ? png__zlib_huff2(n) : (n) <= 279 ? png__zlib_huff3(n) : png__zlib_huff4(n))
#define png__zlib_huffb(n) ((n) <= 143 ? png__zlib_huff1(n) : png__zlib_huff2(n))

        static uint8_t* ZlibCompress(uint8_t* data, int data_len, int* out_len, int quality)
        {
            const int ZHASH = 16384;
            const int basket = quality * 2;
            typedef Array32i HashTable;

            static uint16_t lengthc[] = { 3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258, 259 };
            static uint8_t  lengtheb[] = { 0,0,0,0,0,0,0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,  4,  5,  5,  5,  5,  0 };
            static uint16_t distc[] = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577, 32768 };
            static uint8_t  disteb[] = { 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };
            uint32_t bitbuf = 0;
            int i, j, bitcount = 0;
            unsigned char* out = NULL;
            if (quality < 5)
                quality = 5;
            HashTable hashTable(ZHASH * basket);
            memset(hashTable.data, -1, hashTable.RawSize());

            png__sbpush(out, 0x78);   // DEFLATE 32K window
            png__sbpush(out, 0x5e);   // FLEVEL = 1
            png__zlib_add(1, 1);  // BFINAL = 1
            png__zlib_add(1, 2);  // BTYPE = 1 -- fixed huffman

            i = 0;
            while (i < data_len - 3)
            {
                // hash next 3 bytes of data to be compressed
                int h = ZlibHash(data + i) & (ZHASH - 1), best = 3;
                uint8_t* bestLoc = 0;
                int* hList = hashTable.data + h * basket;
                for (j = 0; hList[j] != -1 && j < basket; ++j)
                {
                    if (hList[j] > i - 32768)
                    {
                        int d = ZlibCount(data + hList[j], data + i, data_len - i);
                        if (d >= best)
                        {
                            best = d;
                            bestLoc = data + hList[j];
                        }
                    }
                }
                if (j == basket)
                {
                    memmove(hList, hList + quality, quality * sizeof(int));
                    memset(hList + quality, -1, quality * sizeof(int));
                    j = quality;
                }
                hList[j] = i;

                if (bestLoc)
                {
                    h = ZlibHash(data + i + 1) & (ZHASH - 1);
                    int* hList = hashTable.data + h * basket;
                    for (j = 0; hList[j] != -1 && j < basket; ++j)
                    {
                        if (hList[j] > i - 32767)
                        {
                            int e = ZlibCount(data + hList[j], data + i + 1, data_len - i - 1);
                            if (e > best)
                            {
                                bestLoc = NULL;
                                break;
                            }
                        }
                    }
                }

                if (bestLoc)
                {
                    int d = (int)(data + i - bestLoc); // distance back
                    assert(d <= 32767 && best <= 258);
                    for (j = 0; best > lengthc[j + 1] - 1; ++j);
                    png__zlib_huff(j + 257);
                    if (lengtheb[j]) png__zlib_add(best - lengthc[j], lengtheb[j]);
                    for (j = 0; d > distc[j + 1] - 1; ++j);
                    png__zlib_add(ZlibBitrev(j, 5), 5);
                    if (disteb[j]) png__zlib_add(d - distc[j], disteb[j]);
                    i += best;
                }
                else
                {
                    png__zlib_huffb(data[i]);
                    ++i;
                }
            }
            // write out final bytes
            for (; i < data_len; ++i)
                png__zlib_huffb(data[i]);
            png__zlib_huff(256); // end of block
            // pad with 0 bits to byte boundary
            while (bitcount)
                png__zlib_add(0, 1);

            {
                // compute adler32 on input
                unsigned int s1 = 1, s2 = 0;
                int blocklen = (int)(data_len % 5552);
                j = 0;
                while (j < data_len)
                {
                    for (i = 0; i < blocklen; ++i) { s1 += data[j + i]; s2 += s1; }
                    s1 %= 65521; s2 %= 65521;
                    j += blocklen;
                    blocklen = 5552;
                }
                png__sbpush(out, PNG_UCHAR(s2 >> 8));
                png__sbpush(out, PNG_UCHAR(s2));
                png__sbpush(out, PNG_UCHAR(s1 >> 8));
                png__sbpush(out, PNG_UCHAR(s1));
            }
            *out_len = png__sbn(out);
            // make returned pointer freeable
            PNG_MEMMOVE(png__sbraw(out), out, *out_len);
            return (unsigned char*)png__sbraw(out);
        }

        SIMD_INLINE uint8_t Paeth(int a, int b, int c)
        {
            int p = a + b - c, pa = abs(p - a), pb = abs(p - b), pc = abs(p - c);
            if (pa <= pb && pa <= pc)
                return uint8_t(a);
            if (pb <= pc)
                return uint8_t(b);
            return uint8_t(c);
        }

        static uint32_t EncodeLine(const uint8_t* src, size_t stride, size_t n, size_t size, int type, int8_t* dst)
        {
            if (type == 0)
                memcpy(dst, src, size);
            else
            {
                for (size_t i = 0; i < n; ++i)
                {
                    switch (type)
                    {
                    case 1: dst[i] = src[i]; break;
                    case 2: dst[i] = src[i] - src[i - stride]; break;
                    case 3: dst[i] = src[i] - (src[i - stride] >> 1); break;
                    case 4: dst[i] = (int8_t)(src[i] - src[i - stride]); break;
                    case 5: dst[i] = src[i]; break;
                    case 6: dst[i] = src[i]; break;
                    }
                }
                switch (type)
                {
                case 1: for (size_t i = n; i < size; ++i) dst[i] = src[i] - src[i - n]; break;
                case 2: for (size_t i = n; i < size; ++i) dst[i] = src[i] - src[i - stride]; break;
                case 3: for (size_t i = n; i < size; ++i) dst[i] = src[i] - ((src[i - n] + src[i - stride]) >> 1); break;
                case 4: for (size_t i = n; i < size; ++i) dst[i] = src[i] - Paeth(src[i - n], src[i - stride], src[i - stride - n]); break;
                case 5: for (size_t i = n; i < size; ++i) dst[i] = src[i] - (src[i - n] >> 1); break;
                case 6: for (size_t i = n; i < size; ++i) dst[i] = src[i] - src[i - n]; break;
                }
            }
            uint32_t sum = 0;
            for (size_t i = 0; i < size; ++i)
                sum += ::abs(dst[i]);
            return sum;
        }

        ImagePngSaver::ImagePngSaver(const ImageSaverParam& param)
            : ImageSaver(param)
            , _channels(0)
            , _size(0)
            , _convert(NULL)
            , _encode(NULL)
        {
            switch (_param.format)
            {
            case SimdPixelFormatGray8:
                _channels = 1;
                break;
            case SimdPixelFormatBgr24:
                _channels = 3;
                break;
            case SimdPixelFormatBgra32:
                _channels = 4;
                break;
            case SimdPixelFormatRgb24:
                _channels = 3;
                break;
            }
            _size = _param.width * _channels;
            if (_param.format == SimdPixelFormatRgb24)
            {
                _convert = Base::BgrToRgb;
                _bgr.Resize(_param.height * _size);
            }
            _filt.Resize((_size + 1) * _param.height);
            _line.Resize(_size * FILTERS);
            _encode = Base::EncodeLine;
        }

        bool ImagePngSaver::ToStream(const uint8_t* src, size_t stride)
        {
            if (_param.format == SimdPixelFormatRgb24)
            {
                _convert(src, _param.width, _param.height, stride, _bgr.data, _size);
                src = _bgr.data;
                stride = _size;
            }
            for (size_t row = 0; row < _param.height; ++row)
            {
                int bestFilter = 0, bestSum = INT_MAX;
                for (int filter = 0; filter < FILTERS; filter++)
                {
                    static const int TYPES[] = { 0, 1, 0, 5, 6, 0, 1, 2, 3, 4 };
                    int type = TYPES[filter + (row ? 1 : 0) * FILTERS];
                    int sum = _encode(src + stride * row, stride, _channels, _size, type, _line.data + _size * filter);
                    if (sum < bestSum)
                    {
                        bestSum = sum;
                        bestFilter = filter;
                    }
                }
                _filt[row * (_size + 1)] = (uint8_t)bestFilter;
                PNG_MEMMOVE(_filt.data + row * (_size + 1) + 1, _line.data + _size * bestFilter, _size);
            }
            int zlen;
            uint8_t* zlib = ZlibCompress(_filt.data, _filt.size, &zlen, COMPRESSION);
            if (zlib)
            {
                WriteToStream(zlib, zlen);
                PNG_FREE(zlib);
                return true;
            }
            return false;
        }

        SIMD_INLINE void WriteCrc32(OutputMemoryStream& stream, size_t size)
        {
            stream.WriteBe32(Base::Crc32(stream.Current() - size - 4, size + 4));
        }

        void ImagePngSaver::WriteToStream(const uint8_t* zlib, size_t zlen)
        {
            const uint8_t SIGNATURE[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
            const int8_t CTYPE[5] = { -1, 0, 4, 2, 6 };
            _stream.Reserve(8 + 12 + 13 + 12 + zlen + 12);
            _stream.Write(SIGNATURE, 8);
            _stream.WriteBe32(13);
            _stream.Write("IHDR", 4);
            _stream.WriteBe32(_param.width);
            _stream.WriteBe32(_param.height);
            _stream.Write<uint8_t>(8);
            _stream.Write<uint8_t>(CTYPE[_channels]);
            _stream.Write<uint8_t>(0);
            _stream.Write<uint8_t>(0);
            _stream.Write<uint8_t>(0);
            WriteCrc32(_stream, 13);
            _stream.WriteBe32(zlen);
            _stream.Write("IDAT", 4);
            _stream.Write(zlib, zlen);
            WriteCrc32(_stream, zlen);
            _stream.WriteBe32(0);
            _stream.Write("IEND", 4);
            WriteCrc32(_stream, 0);
        }
    }
}

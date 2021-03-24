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
        SIMD_INLINE int ZlibBitRev(int bits, int count)
        {
            int rev = 0;
            while (count--)
            {
                rev = (rev << 1) | (bits & 1);
                bits >>= 1;
            }
            return rev;
        }

        SIMD_INLINE int ZlibCount(const uint8_t* a, const uint8_t* b, int limit)
        {
            int i = 0;
            for (; i < limit && i < 258; ++i)
                if (a[i] != b[i]) 
                    break;
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

        SIMD_INLINE void ZlibHuffA(int bits, int count, OutputMemoryStream& stream)
        {
            stream.WriteBits(ZlibBitRev(bits, count), count);
        }

        SIMD_INLINE void ZlibHuff1(int bits, OutputMemoryStream& stream)
        {
            ZlibHuffA(0x30 + bits, 8, stream);
        }

        SIMD_INLINE void ZlibHuff2(int bits, OutputMemoryStream& stream)
        {
            ZlibHuffA(0x190 + bits - 144, 9, stream);
        }

        SIMD_INLINE void ZlibHuff3(int bits, OutputMemoryStream& stream)
        {
            ZlibHuffA(0 + bits - 256, 7, stream);
        }

        SIMD_INLINE void ZlibHuff4(int bits, OutputMemoryStream& stream)
        {
            ZlibHuffA(0xc0 + bits - 280, 8, stream);
        }

        SIMD_INLINE void ZlibHuff(int bits, OutputMemoryStream& stream)
        {
            if (bits <= 143)
                ZlibHuff1(bits, stream);
            else if(bits <= 255)
                ZlibHuff2(bits, stream);
            else if (bits <= 279)
                ZlibHuff3(bits, stream);
            else
                ZlibHuff4(bits, stream);
        }

        SIMD_INLINE void ZlibHuffB(int bits, OutputMemoryStream& stream)
        {
            if (bits <= 143)
                ZlibHuff1(bits, stream);
            else
                ZlibHuff2(bits, stream);
        }

        static void ZlibCompress(uint8_t* data, int size, int quality, OutputMemoryStream& stream)
        {
            const int ZHASH = 16384;
            const int basket = quality * 2;
            typedef Array32i HashTable;

            static uint16_t lengthc[] = { 3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258, 259 };
            static uint8_t  lengtheb[] = { 0,0,0,0,0,0,0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,  4,  5,  5,  5,  5,  0 };
            static uint16_t distc[] = { 1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577, 32768 };
            static uint8_t  disteb[] = { 0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13 };
            if (quality < 5)
                quality = 5;
            HashTable hashTable(ZHASH * basket);
            memset(hashTable.data, -1, hashTable.RawSize());

            stream.Write(uint8_t(0x78));
            stream.Write(uint8_t(0x5e));
            stream.WriteBits(1, 1);
            stream.WriteBits(1, 2);

            int i = 0, j;
            while (i < size - 3)
            {
                int h = ZlibHash(data + i) & (ZHASH - 1), best = 3;
                uint8_t* bestLoc = 0;
                int* hList = hashTable.data + h * basket;
                for (j = 0; hList[j] != -1 && j < basket; ++j)
                {
                    if (hList[j] > i - 32768)
                    {
                        int d = ZlibCount(data + hList[j], data + i, size - i);
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
                            int e = ZlibCount(data + hList[j], data + i + 1, size - i - 1);
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
                    int d = (int)(data + i - bestLoc);
                    assert(d <= 32767 && best <= 258);
                    for (j = 0; best > lengthc[j + 1] - 1; ++j);
                    ZlibHuff(j + 257, stream);
                    if (lengtheb[j])
                        stream.WriteBits(best - lengthc[j], lengtheb[j]);
                    for (j = 0; d > distc[j + 1] - 1; ++j);
                    stream.WriteBits(ZlibBitRev(j, 5), 5);
                    if (disteb[j])
                        stream.WriteBits(d - distc[j], disteb[j]);
                    i += best;
                }
                else
                {
                    ZlibHuffB(data[i], stream);
                    ++i;
                }
            }
            for (; i < size; ++i)
                ZlibHuffB(data[i], stream);
            ZlibHuff(256, stream);
            stream.FlushBits(true);

            unsigned int s1 = 1, s2 = 0;
            int blockSize = (int)(size % 5552);
            j = 0;
            while (j < size)
            {
                for (i = 0; i < blockSize; ++i)
                { 
                    s1 += data[j + i]; 
                    s2 += s1; 
                }
                s1 %= 65521; 
                s2 %= 65521;
                j += blockSize;
                blockSize = 5552;
            }
            stream.Write(uint8_t(s2 >> 8));
            stream.Write(uint8_t(s2));
            stream.Write(uint8_t(s1 >> 8));
            stream.Write(uint8_t(s1));
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
            _compress = Base::ZlibCompress;
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
                memcpy(_filt.data + row * (_size + 1) + 1, _line.data + _size * bestFilter, _size);
            }
            OutputMemoryStream zlib;
            _compress(_filt.data, (int)_filt.size, COMPRESSION, zlib);
            WriteToStream(zlib.Data(), zlib.Size());
            return true;
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
            _stream.WriteBe32((uint32_t)_param.width);
            _stream.WriteBe32((uint32_t)_param.height);
            _stream.Write<uint8_t>(8);
            _stream.Write<uint8_t>(CTYPE[_channels]);
            _stream.Write<uint8_t>(0);
            _stream.Write<uint8_t>(0);
            _stream.Write<uint8_t>(0);
            WriteCrc32(_stream, 13);
            _stream.WriteBe32((uint32_t)zlen);
            _stream.Write("IDAT", 4);
            _stream.Write(zlib, zlen);
            WriteCrc32(_stream, zlen);
            _stream.WriteBe32(0);
            _stream.Write("IEND", 4);
            WriteCrc32(_stream, 0);
        }
    }
}

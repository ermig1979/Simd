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
#include "Simd/SimdMemory.h"
#include "Simd/SimdImageSave.h"
#include "Simd/SimdImageSavePng.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
    namespace Base
    {
        const uint16_t ZlibLenC[30] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258, 259 };
        const uint8_t  ZlibLenEb[29] = { 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4,  4,  5,  5,  5,  5,  0 };
        const uint16_t ZlibDistC[31] = { 1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577, 32768 };
        const uint8_t  ZlibDistEb[30] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13 };

#if defined(SIMD_PNG_ZLIB_BIT_REV_TABLE)
        int ZlibBitRevTable[512];
        static bool ZlibBitRevTableInit()
        {
            for (int i = 0; i < 512; i++)
            {
                int rev = 0, val = i;
                for (size_t b = 0; b < 9; b++)
                {
                    rev = (rev << 1) | (val & 1);
                    val >>= 1;
                }
                ZlibBitRevTable[i] = rev;
            }
            return true;
        }
        bool ZlibBitRevTableInited = ZlibBitRevTableInit();

#endif

        uint32_t ZlibAdler32(uint8_t* data, int size)
        {
            uint32_t lo = 1, hi = 0;
            for (int b = 0, n = (int)(size % 5552); b < size;)
            {
                for (int i = 0; i < n; ++i)
                {
                    lo += data[b + i];
                    hi += lo;
                }
                lo %= 65521;
                hi %= 65521;
                b += n;
                n = 5552;
            }
            return (hi << 16) | lo;
        }

        void ZlibCompress(uint8_t* data, int size, int quality, OutputMemoryStream& stream)
        {
            const int ZHASH = 16384;
            if (quality < 5)
                quality = 5;
            const int basket = quality * 2;
            Array32i hashTable(ZHASH * basket);
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
                    memcpy(hList, hList + quality, quality * sizeof(int));
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
                    for (j = 0; best > Base::ZlibLenC[j + 1] - 1; ++j);
                    Base::ZlibHuff(j + 257, stream);
                    if (Base::ZlibLenEb[j])
                        stream.WriteBits(best - Base::ZlibLenC[j], Base::ZlibLenEb[j]);
                    for (j = 0; d > Base::ZlibDistC[j + 1] - 1; ++j);
                    stream.WriteBits(Base::ZlibBitRev(j, 5), 5);
                    if (Base::ZlibDistEb[j])
                        stream.WriteBits(d - Base::ZlibDistC[j], Base::ZlibDistEb[j]);
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
            stream.FlushBits();
            stream.WriteBe32u(ZlibAdler32(data, size));
        }

        uint32_t EncodeLine0(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            uint32_t sum = 0;
            for (size_t i = 0; i < size; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine1(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            uint32_t sum = 0;
            for (size_t i = 0; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            for (size_t i = n; i < size; ++i)
            {
                dst[i] = src[i] - src[i - n];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine2(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            uint32_t sum = 0;
            for (size_t i = 0; i < n; ++i)
            {
                dst[i] = src[i] - src[i - stride];
                sum += ::abs(dst[i]);
            }
            for (size_t i = n; i < size; ++i)
            {
                dst[i] = src[i] - src[i - stride];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine3(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            uint32_t sum = 0;
            for (size_t i = 0; i < n; ++i)
            {
                dst[i] = src[i] - (src[i - stride] >> 1);
                sum += ::abs(dst[i]);
            }
            for (size_t i = n; i < size; ++i)
            {
                dst[i] = src[i] - ((src[i - n] + src[i - stride]) >> 1);
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine4(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            uint32_t sum = 0;
            for (size_t i = 0; i < n; ++i)
            {
                dst[i] = (int8_t)(src[i] - src[i - stride]);
                sum += ::abs(dst[i]);
            }
            for (size_t i = n; i < size; ++i)
            {
                dst[i] = src[i] - Paeth(src[i - n], src[i - stride], src[i - stride - n]);
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine5(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            uint32_t sum = 0;
            for (size_t i = 0; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            for (size_t i = n; i < size; ++i)
            {
                dst[i] = src[i] - (src[i - n] >> 1);
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine6(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            uint32_t sum = 0;
            for (size_t i = 0; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            for (size_t i = n; i < size; ++i)
            {
                dst[i] = src[i] - src[i - n];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        ImagePngSaver::ImagePngSaver(const ImageSaverParam& param)
            : ImageSaver(param)
            , _channels(0)
            , _size(0)
            , _convert(NULL)
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
            case SimdPixelFormatRgba32:
                _channels = 4;
                break;
            default: 
                break;
            }
            _size = _param.width * _channels;
            if (_param.format == SimdPixelFormatBgr24)
            {
                _convert = Base::BgrToRgb;
                _buff.Resize(_param.height * _size);
            }
            else if (_param.format == SimdPixelFormatBgra32)
            {
                _convert = Base::BgraToRgba;
                _buff.Resize(_param.height * _size);
            }
            _filt.Resize((_size + 1) * _param.height);
            _line.Resize(_size * FILTERS);
            _encode[0] = Base::EncodeLine0;
            _encode[1] = Base::EncodeLine1;
            _encode[2] = Base::EncodeLine2;
            _encode[3] = Base::EncodeLine3;
            _encode[4] = Base::EncodeLine4;
            _encode[5] = Base::EncodeLine5;
            _encode[6] = Base::EncodeLine6;
            _compress = Base::ZlibCompress;
        }

        bool ImagePngSaver::ToStream(const uint8_t* src, size_t stride)
        {
            if (_convert)
            {
                _convert(src, _param.width, _param.height, stride, _buff.data, _size);
                src = _buff.data;
                stride = _size;
            }
            for (size_t row = 0; row < _param.height; ++row)
            {
                int bestFilter = 0, bestSum = INT_MAX;
                for (int filter = 0; filter < FILTERS; filter++)
                {
                    static const int TYPES[] = { 0, 1, 0, 5, 6, 0, 1, 2, 3, 4 };
                    int type = TYPES[filter + (row ? 1 : 0) * FILTERS];
                    int sum = _encode[type](src + stride * row, stride, _channels, _size, _line.data + _size * filter);
                    if (sum < bestSum)
                    {
                        bestSum = sum;
                        bestFilter = filter;
                    }
                }
                _filt[row * (_size + 1)] = (uint8_t)bestFilter;
                memcpy(_filt.data + row * (_size + 1) + 1, _line.data + _size * bestFilter, _size);
            }
            OutputMemoryStream zlib(Simd::Min(_param.width * _param.height, Base::AlgCacheL1()));
            _compress(_filt.data, (int)_filt.size, COMPRESSION, zlib);
            WriteToStream(zlib.Data(), zlib.Size());
            return true;
        }

        SIMD_INLINE void WriteCrc32(OutputMemoryStream& stream, size_t size)
        {
            stream.WriteBe32u(Base::Crc32(stream.Current() - size - 4, size + 4));
        }

        void ImagePngSaver::WriteToStream(const uint8_t* zlib, size_t zlen)
        {
            const uint8_t SIGNATURE[8] = { 137, 80, 78, 71, 13, 10, 26, 10 };
            const int8_t CTYPE[5] = { -1, 0, 4, 2, 6 };
            _stream.Reserve(8 + 12 + 13 + 12 + zlen + 12);
            _stream.Write(SIGNATURE, 8);
            _stream.WriteBe32u(13);
            _stream.Write("IHDR", 4);
            _stream.WriteBe32u((uint32_t)_param.width);
            _stream.WriteBe32u((uint32_t)_param.height);
            _stream.Write8u(8);
            _stream.Write8u(CTYPE[_channels]);
            _stream.Write8u(0);
            _stream.Write8u(0);
            _stream.Write8u(0);
            WriteCrc32(_stream, 13);
            _stream.WriteBe32u((uint32_t)zlen);
            _stream.Write("IDAT", 4);
            _stream.Write(zlib, zlen);
            WriteCrc32(_stream, zlen);
            _stream.WriteBe32u(0);
            _stream.Write("IEND", 4);
            WriteCrc32(_stream, 0);
        }
    }
}

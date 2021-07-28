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
#include "Simd/SimdImageSavePng.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdExtract.h"

namespace Simd
{        
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        uint32_t ZlibAdler32(uint8_t* data, int size)
        {
            __m128i _i0 = _mm_setr_epi32(0, -1, -2, -3), _4 = _mm_set1_epi32(4);
            uint32_t lo = 1, hi = 0;
            for (int b = 0, n = (int)(size % 5552); b < size;)
            {
                int n4 = n & (~3), i = 0;
                __m128i _i = _mm_add_epi32(_i0, _mm_set1_epi32(n));
                __m128i _l = _mm_setzero_si128(), _h = _mm_setzero_si128();
                for (; i < n4; i += 4)
                {
                    __m128i d = _mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(data + b + i)));
                    _l = _mm_add_epi32(_l, d);
                    _h = _mm_add_epi32(_h, _mm_mullo_epi32(d, _i));
                    _i = _mm_sub_epi32(_i, _4);
                }
                int l = Sse2::ExtractInt32Sum(_l), h = Sse2::ExtractInt32Sum(_h);
                for (; i < n; ++i)
                {
                    l += data[b + i];
                    h += data[b + i]*(n - i);
                }
                hi = (hi + h + lo*n) % 65521;
                lo = (lo + l) % 65521;
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
                int h = Base::ZlibHash(data + i) & (ZHASH - 1), best = 3;
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
                    h = Base::ZlibHash(data + i + 1) & (ZHASH - 1);
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
                    Base::ZlibHuffB(data[i], stream);
                    ++i;
                }
            }
            for (; i < size; ++i)
                Base::ZlibHuffB(data[i], stream);
            Base::ZlibHuff(256, stream);
            stream.FlushBits();
            stream.WriteBe32u(ZlibAdler32(data, size));
        }

        uint32_t EncodeLine0(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            size_t i = 0, sizeA = AlignLo(size, A);
            __m128i _sum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i _src = _mm_loadu_si128((__m128i*)(src + i));
                _mm_storeu_si128((__m128i*)(dst + i), _src);
                _sum = _mm_add_epi32(_sum, _mm_sad_epu8(_mm_setzero_si128(), _mm_abs_epi8(_src)));
            }
            uint32_t sum = Sse2::ExtractInt32Sum(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine1(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            size_t i = 0, sizeA = AlignLo(size - n, A) + n;
            uint32_t sum = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            __m128i _sum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i _src0 = _mm_loadu_si128((__m128i*)(src + i));
                __m128i _src1 = _mm_loadu_si128((__m128i*)(src + i - n));
                __m128i _dst = _mm_sub_epi8(_src0, _src1);
                _mm_storeu_si128((__m128i*)(dst + i), _dst);
                _sum = _mm_add_epi32(_sum, _mm_sad_epu8(_mm_setzero_si128(), _mm_abs_epi8(_dst)));
            }
            sum += Sse2::ExtractInt32Sum(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i] - src[i - n];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine2(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            size_t i = 0, sizeA = AlignLo(size - n, A) + n;
            uint32_t sum = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i] - src[i - stride];
                sum += ::abs(dst[i]);
            }
            __m128i _sum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i _src0 = _mm_loadu_si128((__m128i*)(src + i));
                __m128i _src1 = _mm_loadu_si128((__m128i*)(src + i - stride));
                __m128i _dst = _mm_sub_epi8(_src0, _src1);
                _mm_storeu_si128((__m128i*)(dst + i), _dst);
                _sum = _mm_add_epi32(_sum, _mm_sad_epu8(_mm_setzero_si128(), _mm_abs_epi8(_dst)));
            }
            sum += Sse2::ExtractInt32Sum(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i] - src[i - stride];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine3(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            size_t i = 0, sizeA = AlignLo(size - n, A) + n;
            uint32_t sum = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i] - (src[i - stride] >> 1);
                sum += ::abs(dst[i]);
            }
            __m128i _sum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i _src0 = _mm_loadu_si128((__m128i*)(src + i));
                __m128i _src1 = _mm_loadu_si128((__m128i*)(src + i - n));
                __m128i _src2 = _mm_loadu_si128((__m128i*)(src + i - stride));
                __m128i lo = _mm_srli_epi16(_mm_add_epi16(UnpackU8<0>(_src1), UnpackU8<0>(_src2)), 1);
                __m128i hi = _mm_srli_epi16(_mm_add_epi16(UnpackU8<1>(_src1), UnpackU8<1>(_src2)), 1);
                __m128i _dst = _mm_sub_epi8(_src0, _mm_packus_epi16(lo, hi));
                _mm_storeu_si128((__m128i*)(dst + i), _dst);
                _sum = _mm_add_epi32(_sum, _mm_sad_epu8(_mm_setzero_si128(), _mm_abs_epi8(_dst)));
            }
            sum += Sse2::ExtractInt32Sum(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i] - ((src[i - n] + src[i - stride]) >> 1);
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        SIMD_INLINE __m128i Paeth(__m128i a, __m128i b, __m128i c)
        {
            __m128i p = _mm_sub_epi16(_mm_add_epi16(a, b), c);
            __m128i pa = _mm_abs_epi16(_mm_sub_epi16(p, a));
            __m128i pb = _mm_abs_epi16(_mm_sub_epi16(p, b));
            __m128i pc = _mm_abs_epi16(_mm_sub_epi16(p, c));
            __m128i mbc = _mm_or_si128(_mm_cmpgt_epi16(pa, pb), _mm_cmpgt_epi16(pa, pc));
            __m128i mc = _mm_cmpgt_epi16(pb, pc);
            return _mm_blendv_epi8(a, _mm_blendv_epi8(b, c, mc), mbc);
        }

        uint32_t EncodeLine4(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            size_t i = 0, sizeA = AlignLo(size - n, A) + n;
            uint32_t sum = 0;
            for (; i < n; ++i)
            {
                dst[i] = (int8_t)(src[i] - src[i - stride]);
                sum += ::abs(dst[i]);
            }
            __m128i _sum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i _src0 = _mm_loadu_si128((__m128i*)(src + i));
                __m128i _src1 = _mm_loadu_si128((__m128i*)(src + i - n));
                __m128i _src2 = _mm_loadu_si128((__m128i*)(src + i - stride));
                __m128i _src3 = _mm_loadu_si128((__m128i*)(src + i - stride - n));
                __m128i lo = Paeth(UnpackU8<0>(_src1), UnpackU8<0>(_src2), UnpackU8<0>(_src3));
                __m128i hi = Paeth(UnpackU8<1>(_src1), UnpackU8<1>(_src2), UnpackU8<1>(_src3));
                __m128i _dst = _mm_sub_epi8(_src0, _mm_packus_epi16(lo, hi));
                _mm_storeu_si128((__m128i*)(dst + i), _dst);
                _sum = _mm_add_epi32(_sum, _mm_sad_epu8(_mm_setzero_si128(), _mm_abs_epi8(_dst)));
            }
            sum += Sse2::ExtractInt32Sum(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i] - Base::Paeth(src[i - n], src[i - stride], src[i - stride - n]);
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine5(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            size_t i = 0, sizeA = AlignLo(size - n, A) + n;
            uint32_t sum = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            __m128i _sum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i _src0 = _mm_loadu_si128((__m128i*)(src + i));
                __m128i _src1 = _mm_loadu_si128((__m128i*)(src + i - n));
                __m128i lo = _mm_srli_epi16(UnpackU8<0>(_src1), 1);
                __m128i hi = _mm_srli_epi16(UnpackU8<1>(_src1), 1);
                __m128i _dst = _mm_sub_epi8(_src0, _mm_packus_epi16(lo, hi));
                _mm_storeu_si128((__m128i*)(dst + i), _dst);
                _sum = _mm_add_epi32(_sum, _mm_sad_epu8(_mm_setzero_si128(), _mm_abs_epi8(_dst)));
            }
            sum += Sse2::ExtractInt32Sum(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i] - (src[i - n] >> 1);
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        uint32_t EncodeLine6(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            size_t i = 0, sizeA = AlignLo(size - n, A) + n;
            uint32_t sum = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            __m128i _sum = _mm_setzero_si128();
            for (; i < sizeA; i += A)
            {
                __m128i _src0 = _mm_loadu_si128((__m128i*)(src + i));
                __m128i _src1 = _mm_loadu_si128((__m128i*)(src + i - n));
                __m128i _dst = _mm_sub_epi8(_src0, _src1);
                _mm_storeu_si128((__m128i*)(dst + i), _dst);
                _sum = _mm_add_epi32(_sum, _mm_sad_epu8(_mm_setzero_si128(), _mm_abs_epi8(_dst)));
            }
            sum += Sse2::ExtractInt32Sum(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i] - src[i - n];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        ImagePngSaver::ImagePngSaver(const ImageSaverParam& param)
            : Base::ImagePngSaver(param)
        {
            if (_param.format == SimdPixelFormatBgr24)
                _convert = Sse41::BgrToRgb;
            else if (_param.format == SimdPixelFormatBgra32)
                _convert = Sse41::BgraToRgba;
            _encode[0] = Sse41::EncodeLine0;
            _encode[1] = Sse41::EncodeLine1;
            _encode[2] = Sse41::EncodeLine2;
            _encode[3] = Sse41::EncodeLine3;
            _encode[4] = Sse41::EncodeLine4;
            _encode[5] = Sse41::EncodeLine5;
            _encode[6] = Sse41::EncodeLine6;
            _compress = Sse41::ZlibCompress;
        }
    }
#endif// SIMD_SSE41_ENABLE
}

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
#include "Simd/SimdAvx2.h"
#include "Simd/SimdExtract.h"

namespace Simd
{        
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        static uint32_t ZlibAdler32(uint8_t* data, int size)
        {
            __m256i _i0 = _mm256_setr_epi32(0, -1, -2, -3, -4, -5, -6, -7), _8 = _mm256_set1_epi32(8);
            uint32_t lo = 1, hi = 0;
            for (int b = 0, n = (int)(size % 5552); b < size;)
            {
                int n8 = n & (~7), i = 0;
                __m256i _i = _mm256_add_epi32(_i0, _mm256_set1_epi32(n));
                __m256i _l = _mm256_setzero_si256(), _h = _mm256_setzero_si256();
                for (; i < n8; i += 8)
                {
                    __m256i d = _mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(data + b + i)));
                    _l = _mm256_add_epi32(_l, d);
                    _h = _mm256_add_epi32(_h, _mm256_mullo_epi32(d, _i));
                    _i = _mm256_sub_epi32(_i, _8);
                }
                int l = Avx2::ExtractSum<uint32_t>(_l), h = Avx2::ExtractSum<uint32_t>(_h);
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
                        int d = Avx2::ZlibCount(data + hList[j], data + i, size - i);
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
                            int e = Avx2::ZlibCount(data + hList[j], data + i + 1, size - i - 1);
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
            __m256i _sum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i _src = _mm256_loadu_si256((__m256i*)(src + i));
                _mm256_storeu_si256((__m256i*)(dst + i), _src);
                _sum = _mm256_add_epi32(_sum, _mm256_sad_epu8(_mm256_setzero_si256(), _mm256_abs_epi8(_src)));
            }
            uint32_t sum = Avx2::ExtractSum<uint32_t>(_sum);
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
            __m256i _sum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i _src0 = _mm256_loadu_si256((__m256i*)(src + i));
                __m256i _src1 = _mm256_loadu_si256((__m256i*)(src + i - n));
                __m256i _dst = _mm256_sub_epi8(_src0, _src1);
                _mm256_storeu_si256((__m256i*)(dst + i), _dst);
                _sum = _mm256_add_epi32(_sum, _mm256_sad_epu8(_mm256_setzero_si256(), _mm256_abs_epi8(_dst)));
            }
            sum += Avx2::ExtractSum<uint32_t>(_sum);
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
            __m256i _sum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i _src0 = _mm256_loadu_si256((__m256i*)(src + i));
                __m256i _src1 = _mm256_loadu_si256((__m256i*)(src + i - stride));
                __m256i _dst = _mm256_sub_epi8(_src0, _src1);
                _mm256_storeu_si256((__m256i*)(dst + i), _dst);
                _sum = _mm256_add_epi32(_sum, _mm256_sad_epu8(_mm256_setzero_si256(), _mm256_abs_epi8(_dst)));
            }
            sum += Avx2::ExtractSum<uint32_t>(_sum);
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
            __m256i _sum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i _src0 = _mm256_loadu_si256((__m256i*)(src + i));
                __m256i _src1 = _mm256_loadu_si256((__m256i*)(src + i - n));
                __m256i _src2 = _mm256_loadu_si256((__m256i*)(src + i - stride));
                __m256i lo = _mm256_srli_epi16(_mm256_add_epi16(UnpackU8<0>(_src1), UnpackU8<0>(_src2)), 1);
                __m256i hi = _mm256_srli_epi16(_mm256_add_epi16(UnpackU8<1>(_src1), UnpackU8<1>(_src2)), 1);
                __m256i _dst = _mm256_sub_epi8(_src0, _mm256_packus_epi16(lo, hi));
                _mm256_storeu_si256((__m256i*)(dst + i), _dst);
                _sum = _mm256_add_epi32(_sum, _mm256_sad_epu8(_mm256_setzero_si256(), _mm256_abs_epi8(_dst)));
            }
            sum += Avx2::ExtractSum<uint32_t>(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i] - ((src[i - n] + src[i - stride]) >> 1);
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        SIMD_INLINE __m256i Paeth(__m256i a, __m256i b, __m256i c)
        {
            __m256i p = _mm256_sub_epi16(_mm256_add_epi16(a, b), c);
            __m256i pa = _mm256_abs_epi16(_mm256_sub_epi16(p, a));
            __m256i pb = _mm256_abs_epi16(_mm256_sub_epi16(p, b));
            __m256i pc = _mm256_abs_epi16(_mm256_sub_epi16(p, c));
            __m256i mbc = _mm256_or_si256(_mm256_cmpgt_epi16(pa, pb), _mm256_cmpgt_epi16(pa, pc));
            __m256i mc = _mm256_cmpgt_epi16(pb, pc);
            return _mm256_blendv_epi8(a, _mm256_blendv_epi8(b, c, mc), mbc);
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
            __m256i _sum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i _src0 = _mm256_loadu_si256((__m256i*)(src + i));
                __m256i _src1 = _mm256_loadu_si256((__m256i*)(src + i - n));
                __m256i _src2 = _mm256_loadu_si256((__m256i*)(src + i - stride));
                __m256i _src3 = _mm256_loadu_si256((__m256i*)(src + i - stride - n));
                __m256i lo = Paeth(UnpackU8<0>(_src1), UnpackU8<0>(_src2), UnpackU8<0>(_src3));
                __m256i hi = Paeth(UnpackU8<1>(_src1), UnpackU8<1>(_src2), UnpackU8<1>(_src3));
                __m256i _dst = _mm256_sub_epi8(_src0, _mm256_packus_epi16(lo, hi));
                _mm256_storeu_si256((__m256i*)(dst + i), _dst);
                _sum = _mm256_add_epi32(_sum, _mm256_sad_epu8(_mm256_setzero_si256(), _mm256_abs_epi8(_dst)));
            }
            sum += Avx2::ExtractSum<uint32_t>(_sum);
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
            __m256i _sum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i _src0 = _mm256_loadu_si256((__m256i*)(src + i));
                __m256i _src1 = _mm256_loadu_si256((__m256i*)(src + i - n));
                __m256i lo = _mm256_srli_epi16(UnpackU8<0>(_src1), 1);
                __m256i hi = _mm256_srli_epi16(UnpackU8<1>(_src1), 1);
                __m256i _dst = _mm256_sub_epi8(_src0, _mm256_packus_epi16(lo, hi));
                _mm256_storeu_si256((__m256i*)(dst + i), _dst);
                _sum = _mm256_add_epi32(_sum, _mm256_sad_epu8(_mm256_setzero_si256(), _mm256_abs_epi8(_dst)));
            }
            sum += Avx2::ExtractSum<uint32_t>(_sum);
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
            __m256i _sum = _mm256_setzero_si256();
            for (; i < sizeA; i += A)
            {
                __m256i _src0 = _mm256_loadu_si256((__m256i*)(src + i));
                __m256i _src1 = _mm256_loadu_si256((__m256i*)(src + i - n));
                __m256i _dst = _mm256_sub_epi8(_src0, _src1);
                _mm256_storeu_si256((__m256i*)(dst + i), _dst);
                _sum = _mm256_add_epi32(_sum, _mm256_sad_epu8(_mm256_setzero_si256(), _mm256_abs_epi8(_dst)));
            }
            sum += Avx2::ExtractSum<uint32_t>(_sum);
            for (; i < size; ++i)
            {
                dst[i] = src[i] - src[i - n];
                sum += ::abs(dst[i]);
            }
            return sum;
        }

        ImagePngSaver::ImagePngSaver(const ImageSaverParam& param)
            : Sse41::ImagePngSaver(param)
        {
            if (_param.format == SimdPixelFormatBgr24)
                _convert = Avx2::BgrToRgb;
            else if (_param.format == SimdPixelFormatBgra32)
                _convert = Avx2::BgraToRgba;
            _encode[0] = Avx2::EncodeLine0;
            _encode[1] = Avx2::EncodeLine1;
            _encode[2] = Avx2::EncodeLine2;
            _encode[3] = Avx2::EncodeLine3;
            _encode[4] = Avx2::EncodeLine4;
            _encode[5] = Avx2::EncodeLine5;
            _encode[6] = Avx2::EncodeLine6;
            _compress = Avx2::ZlibCompress;
        }
    }
#endif// SIMD_AVX2_ENABLE
}

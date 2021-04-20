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
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdExtract.h"

namespace Simd
{        
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        static uint32_t ZlibAdler32(uint8_t* data, int size)
        {
            __m512i _i0 = _mm512_setr_epi32(0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12, -13, -14, -15), _16 = _mm512_set1_epi32(16);
            uint32_t lo = 1, hi = 0;
            for (int b = 0, n = (int)(size % 5552); b < size;)
            {
                int n16 = n & (~15), i = 0;
                __m512i _i = _mm512_add_epi32(_i0, _mm512_set1_epi32(n));
                __m512i _l = _mm512_setzero_si512(), _h = _mm512_setzero_si512();
                for (; i < n16; i += 16)
                {
                    __m512i d = _mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(data + b + i)));
                    _l = _mm512_add_epi32(_l, d);
                    _h = _mm512_add_epi32(_h, _mm512_mullo_epi32(d, _i));
                    _i = _mm512_sub_epi32(_i, _16);
                }
                if (i < n)
                {
                    __mmask16 t = TailMask16(n - i);
                    __m512i d = _mm512_cvtepu8_epi32(_mm_maskz_loadu_epi8(t, (__m128i*)(data + b + i)));
                    _l = _mm512_add_epi32(_l, d);
                    _h = _mm512_add_epi32(_h, _mm512_mullo_epi32(d, _i));
                    _i = _mm512_sub_epi32(_i, _16);
                }
                int l = Avx512bw::ExtractSum<uint32_t>(_l), h = Avx512bw::ExtractSum<uint32_t>(_h);
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
            __m512i _sum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i _src = _mm512_loadu_si512((__m512i*)(src + i));
                _mm512_storeu_si512((__m512i*)(dst + i), _src);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_src)));
            }
            if (i < size)
            {
                __mmask64 tail = TailMask64(size - i);
                __m512i _src = _mm512_maskz_loadu_epi8(tail, src + i);
                _mm512_mask_storeu_epi8(dst + i, tail, _src);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_src)));
            }
            return Avx512bw::ExtractSum<uint32_t>(_sum);
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
            __m512i _sum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i _src0 = _mm512_loadu_si512((__m512i*)(src + i));
                __m512i _src1 = _mm512_loadu_si512((__m512i*)(src + i - n));
                __m512i _dst = _mm512_sub_epi8(_src0, _src1);
                _mm512_storeu_si512((__m512i*)(dst + i), _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            if (i < size)
            {
                __mmask64 tail = TailMask64(size - i);
                __m512i _src0 = _mm512_maskz_loadu_epi8(tail, src + i);
                __m512i _src1 = _mm512_maskz_loadu_epi8(tail, src + i - n);
                __m512i _dst = _mm512_sub_epi8(_src0, _src1);
                _mm512_mask_storeu_epi8(dst + i, tail, _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            return sum + Avx512bw::ExtractSum<uint32_t>(_sum);
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
            __m512i _sum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i _src0 = _mm512_loadu_si512((__m512i*)(src + i));
                __m512i _src1 = _mm512_loadu_si512((__m512i*)(src + i - stride));
                __m512i _dst = _mm512_sub_epi8(_src0, _src1);
                _mm512_storeu_si512((__m512i*)(dst + i), _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            if (i < size)
            {
                __mmask64 tail = TailMask64(size - i);
                __m512i _src0 = _mm512_maskz_loadu_epi8(tail, src + i);
                __m512i _src1 = _mm512_maskz_loadu_epi8(tail, src + i - stride);
                __m512i _dst = _mm512_sub_epi8(_src0, _src1);
                _mm512_mask_storeu_epi8(dst + i, tail, _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            return sum + Avx512bw::ExtractSum<uint32_t>(_sum);
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
            __m512i _sum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i _src0 = _mm512_loadu_si512((__m512i*)(src + i));
                __m512i _src1 = _mm512_loadu_si512((__m512i*)(src + i - n));
                __m512i _src2 = _mm512_loadu_si512((__m512i*)(src + i - stride));
                __m512i lo = _mm512_srli_epi16(_mm512_add_epi16(UnpackU8<0>(_src1), UnpackU8<0>(_src2)), 1);
                __m512i hi = _mm512_srli_epi16(_mm512_add_epi16(UnpackU8<1>(_src1), UnpackU8<1>(_src2)), 1);
                __m512i _dst = _mm512_sub_epi8(_src0, _mm512_packus_epi16(lo, hi));
                _mm512_storeu_si512((__m512i*)(dst + i), _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            if (i < size)
            {
                __mmask64 tail = TailMask64(size - i);
                __m512i _src0 = _mm512_maskz_loadu_epi8(tail, src + i);
                __m512i _src1 = _mm512_maskz_loadu_epi8(tail, src + i - n);
                __m512i _src2 = _mm512_maskz_loadu_epi8(tail, src + i - stride);
                __m512i lo = _mm512_srli_epi16(_mm512_add_epi16(UnpackU8<0>(_src1), UnpackU8<0>(_src2)), 1);
                __m512i hi = _mm512_srli_epi16(_mm512_add_epi16(UnpackU8<1>(_src1), UnpackU8<1>(_src2)), 1);
                __m512i _dst = _mm512_sub_epi8(_src0, _mm512_packus_epi16(lo, hi));
                _mm512_mask_storeu_epi8(dst + i, tail, _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            return sum + Avx512bw::ExtractSum<uint32_t>(_sum);
        }
        
        SIMD_INLINE __m512i Paeth(__m512i a, __m512i b, __m512i c)
        {
            __m512i p = _mm512_sub_epi16(_mm512_add_epi16(a, b), c);
            __m512i pa = _mm512_abs_epi16(_mm512_sub_epi16(p, a));
            __m512i pb = _mm512_abs_epi16(_mm512_sub_epi16(p, b));
            __m512i pc = _mm512_abs_epi16(_mm512_sub_epi16(p, c));
            __mmask32 mbc = _mm512_cmpgt_epi16_mask(pa, pb) | _mm512_cmpgt_epi16_mask(pa, pc);
            __mmask32 mc = _mm512_cmpgt_epi16_mask(pb, pc);
            return _mm512_mask_mov_epi16(a, mbc, _mm512_mask_mov_epi16(b, mc, c));
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
            __m512i _sum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i _src0 = _mm512_loadu_si512((__m512i*)(src + i));
                __m512i _src1 = _mm512_loadu_si512((__m512i*)(src + i - n));
                __m512i _src2 = _mm512_loadu_si512((__m512i*)(src + i - stride));
                __m512i _src3 = _mm512_loadu_si512((__m512i*)(src + i - stride - n));
                __m512i lo = Paeth(UnpackU8<0>(_src1), UnpackU8<0>(_src2), UnpackU8<0>(_src3));
                __m512i hi = Paeth(UnpackU8<1>(_src1), UnpackU8<1>(_src2), UnpackU8<1>(_src3));
                __m512i _dst = _mm512_sub_epi8(_src0, _mm512_packus_epi16(lo, hi));
                _mm512_storeu_si512((__m512i*)(dst + i), _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            if (i < size)
            {
                __mmask64 tail = TailMask64(size - i);
                __m512i _src0 = _mm512_maskz_loadu_epi8(tail, src + i);
                __m512i _src1 = _mm512_maskz_loadu_epi8(tail, src + i - n);
                __m512i _src2 = _mm512_maskz_loadu_epi8(tail, src + i - stride);
                __m512i _src3 = _mm512_maskz_loadu_epi8(tail, src + i - stride - n);
                __m512i lo = Paeth(UnpackU8<0>(_src1), UnpackU8<0>(_src2), UnpackU8<0>(_src3));
                __m512i hi = Paeth(UnpackU8<1>(_src1), UnpackU8<1>(_src2), UnpackU8<1>(_src3));
                __m512i _dst = _mm512_sub_epi8(_src0, _mm512_packus_epi16(lo, hi));
                _mm512_mask_storeu_epi8(dst + i, tail, _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            return sum + Avx512bw::ExtractSum<uint32_t>(_sum);
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
            __m512i _sum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i _src0 = _mm512_loadu_si512((__m512i*)(src + i));
                __m512i _src1 = _mm512_loadu_si512((__m512i*)(src + i - n));
                __m512i lo = _mm512_srli_epi16(UnpackU8<0>(_src1), 1);
                __m512i hi = _mm512_srli_epi16(UnpackU8<1>(_src1), 1);
                __m512i _dst = _mm512_sub_epi8(_src0, _mm512_packus_epi16(lo, hi));
                _mm512_storeu_si512((__m512i*)(dst + i), _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            if (i < size)
            {
                __mmask64 tail = TailMask64(size - i);
                __m512i _src0 = _mm512_maskz_loadu_epi8(tail, src + i);
                __m512i _src1 = _mm512_maskz_loadu_epi8(tail, src + i - n);
                __m512i lo = _mm512_srli_epi16(UnpackU8<0>(_src1), 1);
                __m512i hi = _mm512_srli_epi16(UnpackU8<1>(_src1), 1);
                __m512i _dst = _mm512_sub_epi8(_src0, _mm512_packus_epi16(lo, hi));
                _mm512_mask_storeu_epi8(dst + i, tail, _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            return sum + Avx512bw::ExtractSum<uint32_t>(_sum);
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
            __m512i _sum = _mm512_setzero_si512();
            for (; i < sizeA; i += A)
            {
                __m512i _src0 = _mm512_loadu_si512((__m512i*)(src + i));
                __m512i _src1 = _mm512_loadu_si512((__m512i*)(src + i - n));
                __m512i _dst = _mm512_sub_epi8(_src0, _src1);
                _mm512_storeu_si512((__m512i*)(dst + i), _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            if (i < size)
            {
                __mmask64 tail = TailMask64(size - i);
                __m512i _src0 = _mm512_maskz_loadu_epi8(tail, src + i);
                __m512i _src1 = _mm512_maskz_loadu_epi8(tail, src + i - n);
                __m512i _dst = _mm512_sub_epi8(_src0, _src1);
                _mm512_mask_storeu_epi8(dst + i, tail, _dst);
                _sum = _mm512_add_epi32(_sum, _mm512_sad_epu8(_mm512_setzero_si512(), _mm512_abs_epi8(_dst)));
            }
            return sum + Avx512bw::ExtractSum<uint32_t>(_sum);
        }

        ImagePngSaver::ImagePngSaver(const ImageSaverParam& param)
            : Avx2::ImagePngSaver(param)
        {
            if (_param.format == SimdPixelFormatBgr24)
                _convert = Avx512bw::BgrToRgb;
            else if (_param.format == SimdPixelFormatBgra32)
                _convert = Avx512bw::BgraToRgba;
            _encode[0] = Avx512bw::EncodeLine0;
            _encode[1] = Avx512bw::EncodeLine1;
            _encode[2] = Avx512bw::EncodeLine2;
            _encode[3] = Avx512bw::EncodeLine3;
            _encode[4] = Avx512bw::EncodeLine4;
            _encode[5] = Avx512bw::EncodeLine5;
            _encode[6] = Avx512bw::EncodeLine6;
            _compress = Avx512bw::ZlibCompress;
        }
    }
#endif// SIMD_AVX512BW_ENABLE
}

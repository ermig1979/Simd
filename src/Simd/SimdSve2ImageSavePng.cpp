/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2026 Yermalayeu Ihar.
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
#include "Simd/SimdSve2.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE void AddAbsSum(const svint8_t& value, const svbool_t& mask, const svuint8_t& ones, const svuint8_t& zero, svuint32_t& sum)
        {
            svuint8_t abs8 = svreinterpret_u8(svabs_s8_x(mask, value));
            sum = svdot_u32(sum, svsel_u8(mask, abs8, zero), ones);
        }

        SIMD_INLINE svuint8_t PackU16(const svuint16_t& lo, const svuint16_t& hi)
        {
            return svuzp1_u8(svreinterpret_u8(lo), svreinterpret_u8(hi));
        }

        uint32_t ZlibAdler32(uint8_t* data, int size)
        {
            const size_t N = svcntw();
            uint32_t lo = 1, hi = 0;
            for (int b = 0, n = (int)(size % 5552); b < size;)
            {
                svuint32_t _l = svdup_n_u32(0), _h = svdup_n_u32(0);
                int i = 0;
                for (; i < n; i += (int)N)
                {
                    svbool_t mask = svwhilelt_b32((uint64_t)i, (uint64_t)n);
                    svuint32_t d = svld1ub_u32(mask, data + b + i);
                    svuint32_t w = svreinterpret_u32(svindex_s32((int32_t)(n - i), -1));
                    _l = svadd_u32_m(mask, _l, d);
                    _h = svmla_u32_m(mask, _h, d, w);
                }
                uint32_t l = svaddv_u32(svptrue_b32(), _l);
                uint32_t h = svaddv_u32(svptrue_b32(), _h);
                hi = (hi + h + lo * n) % 65521;
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
            const size_t A = svcntb();
            const size_t sizeA = AlignLo(size, A);
            const svbool_t body = svptrue_b8();
            const svuint8_t ones = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            svuint32_t sum = svdup_n_u32(0);
            size_t i = 0;
            for (; i < sizeA; i += A)
            {
                svint8_t _src = svreinterpret_s8(svld1_u8(body, src + i));
                svst1_s8(body, dst + i, _src);
                AddAbsSum(_src, body, ones, zero, sum);
            }
            if (sizeA < size)
            {
                svbool_t tail = svwhilelt_b8(sizeA, size);
                svint8_t _src = svreinterpret_s8(svld1_u8(tail, src + i));
                svst1_s8(tail, dst + i, _src);
                AddAbsSum(_src, tail, ones, zero, sum);
            }
            return svaddv_u32(svptrue_b32(), sum);
        }

        uint32_t EncodeLine1(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            const size_t A = svcntb();
            const size_t sizeA = AlignLo(size - n, A) + n;
            const svbool_t body = svptrue_b8();
            const svuint8_t ones = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            uint32_t sum = 0;
            size_t i = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            svuint32_t _sum = svdup_n_u32(0);
            for (; i < sizeA; i += A)
            {
                svint8_t _dst = svsub_s8_x(body, svreinterpret_s8(svld1_u8(body, src + i)), svreinterpret_s8(svld1_u8(body, src + i - n)));
                svst1_s8(body, dst + i, _dst);
                AddAbsSum(_dst, body, ones, zero, _sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b8(i, size);
                svint8_t _dst = svsub_s8_x(tail, svreinterpret_s8(svld1_u8(tail, src + i)), svreinterpret_s8(svld1_u8(tail, src + i - n)));
                svst1_s8(tail, dst + i, _dst);
                AddAbsSum(_dst, tail, ones, zero, _sum);
            }
            return sum + svaddv_u32(svptrue_b32(), _sum);
        }

        uint32_t EncodeLine2(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            const size_t A = svcntb();
            const size_t sizeA = AlignLo(size - n, A) + n;
            const svbool_t body = svptrue_b8();
            const svuint8_t ones = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            uint32_t sum = 0;
            size_t i = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i] - src[i - stride];
                sum += ::abs(dst[i]);
            }
            svuint32_t _sum = svdup_n_u32(0);
            for (; i < sizeA; i += A)
            {
                svint8_t _dst = svsub_s8_x(body, svreinterpret_s8(svld1_u8(body, src + i)), svreinterpret_s8(svld1_u8(body, src + i - stride)));
                svst1_s8(body, dst + i, _dst);
                AddAbsSum(_dst, body, ones, zero, _sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b8(i, size);
                svint8_t _dst = svsub_s8_x(tail, svreinterpret_s8(svld1_u8(tail, src + i)), svreinterpret_s8(svld1_u8(tail, src + i - stride)));
                svst1_s8(tail, dst + i, _dst);
                AddAbsSum(_dst, tail, ones, zero, _sum);
            }
            return sum + svaddv_u32(svptrue_b32(), _sum);
        }

        uint32_t EncodeLine3(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            const size_t A = svcntb();
            const size_t sizeA = AlignLo(size - n, A) + n;
            const svbool_t body = svptrue_b8();
            const svuint8_t ones = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            uint32_t sum = 0;
            size_t i = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i] - (src[i - stride] >> 1);
                sum += ::abs(dst[i]);
            }
            svuint32_t _sum = svdup_n_u32(0);
            for (; i < sizeA; i += A)
            {
                svuint8_t _src0 = svld1_u8(body, src + i);
                svuint8_t avg = svhadd_u8_x(body, svld1_u8(body, src + i - n), svld1_u8(body, src + i - stride));
                svint8_t _dst = svreinterpret_s8(svsub_u8_x(body, _src0, avg));
                svst1_s8(body, dst + i, _dst);
                AddAbsSum(_dst, body, ones, zero, _sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b8(i, size);
                svuint8_t _src0 = svld1_u8(tail, src + i);
                svuint8_t avg = svhadd_u8_x(tail, svld1_u8(tail, src + i - n), svld1_u8(tail, src + i - stride));
                svint8_t _dst = svreinterpret_s8(svsub_u8_x(tail, _src0, avg));
                svst1_s8(tail, dst + i, _dst);
                AddAbsSum(_dst, tail, ones, zero, _sum);
            }
            return sum + svaddv_u32(svptrue_b32(), _sum);
        }

        SIMD_INLINE svuint16_t Paeth(const svuint16_t& a, const svuint16_t& b, const svuint16_t& c, const svbool_t& mask)
        {
            svint16_t _a = svreinterpret_s16(a);
            svint16_t _b = svreinterpret_s16(b);
            svint16_t _c = svreinterpret_s16(c);
            svint16_t p = svsub_s16_x(mask, svadd_s16_x(mask, _a, _b), _c);
            svint16_t pa = svabs_s16_x(mask, svsub_s16_x(mask, p, _a));
            svint16_t pb = svabs_s16_x(mask, svsub_s16_x(mask, p, _b));
            svint16_t pc = svabs_s16_x(mask, svsub_s16_x(mask, p, _c));
            svbool_t mbc = svorr_b_z(mask, svcmpgt_s16(mask, pa, pb), svcmpgt_s16(mask, pa, pc));
            svbool_t mc = svcmpgt_s16(mask, pb, pc);
            return svsel_u16(mbc, svsel_u16(mc, c, b), a);
        }

        uint32_t EncodeLine4(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            const size_t A = svcntb();
            const size_t sizeA = AlignLo(size - n, A) + n;
            const svbool_t body = svptrue_b8();
            const svbool_t body16 = svptrue_b16();
            const svuint8_t ones = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            uint32_t sum = 0;
            size_t i = 0;
            for (; i < n; ++i)
            {
                dst[i] = (int8_t)(src[i] - src[i - stride]);
                sum += ::abs(dst[i]);
            }
            svuint32_t _sum = svdup_n_u32(0);
            for (; i < sizeA; i += A)
            {
                svuint8_t _src0 = svld1_u8(body, src + i);
                svuint8_t _src1 = svld1_u8(body, src + i - n);
                svuint8_t _src2 = svld1_u8(body, src + i - stride);
                svuint8_t _src3 = svld1_u8(body, src + i - stride - n);
                svuint16_t lo = Paeth(svunpklo_u16(_src1), svunpklo_u16(_src2), svunpklo_u16(_src3), body16);
                svuint16_t hi = Paeth(svunpkhi_u16(_src1), svunpkhi_u16(_src2), svunpkhi_u16(_src3), body16);
                svint8_t _dst = svreinterpret_s8(svsub_u8_x(body, _src0, PackU16(lo, hi)));
                svst1_s8(body, dst + i, _dst);
                AddAbsSum(_dst, body, ones, zero, _sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b8(i, size);
                svuint8_t _src0 = svld1_u8(tail, src + i);
                svuint8_t _src1 = svld1_u8(tail, src + i - n);
                svuint8_t _src2 = svld1_u8(tail, src + i - stride);
                svuint8_t _src3 = svld1_u8(tail, src + i - stride - n);
                svuint16_t lo = Paeth(svunpklo_u16(_src1), svunpklo_u16(_src2), svunpklo_u16(_src3), body16);
                svuint16_t hi = Paeth(svunpkhi_u16(_src1), svunpkhi_u16(_src2), svunpkhi_u16(_src3), body16);
                svint8_t _dst = svreinterpret_s8(svsub_u8_x(tail, _src0, PackU16(lo, hi)));
                svst1_s8(tail, dst + i, _dst);
                AddAbsSum(_dst, tail, ones, zero, _sum);
            }
            return sum + svaddv_u32(svptrue_b32(), _sum);
        }

        uint32_t EncodeLine5(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            const size_t A = svcntb();
            const size_t sizeA = AlignLo(size - n, A) + n;
            const svbool_t body = svptrue_b8();
            const svuint8_t ones = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            uint32_t sum = 0;
            size_t i = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            svuint32_t _sum = svdup_n_u32(0);
            for (; i < sizeA; i += A)
            {
                svuint8_t _src0 = svld1_u8(body, src + i);
                svuint8_t half = svlsr_n_u8_x(body, svld1_u8(body, src + i - n), 1);
                svint8_t _dst = svreinterpret_s8(svsub_u8_x(body, _src0, half));
                svst1_s8(body, dst + i, _dst);
                AddAbsSum(_dst, body, ones, zero, _sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b8(i, size);
                svuint8_t _src0 = svld1_u8(tail, src + i);
                svuint8_t half = svlsr_n_u8_x(tail, svld1_u8(tail, src + i - n), 1);
                svint8_t _dst = svreinterpret_s8(svsub_u8_x(tail, _src0, half));
                svst1_s8(tail, dst + i, _dst);
                AddAbsSum(_dst, tail, ones, zero, _sum);
            }
            return sum + svaddv_u32(svptrue_b32(), _sum);
        }

        uint32_t EncodeLine6(const uint8_t* src, size_t stride, size_t n, size_t size, int8_t* dst)
        {
            const size_t A = svcntb();
            const size_t sizeA = AlignLo(size - n, A) + n;
            const svbool_t body = svptrue_b8();
            const svuint8_t ones = svdup_n_u8(1);
            const svuint8_t zero = svdup_n_u8(0);
            uint32_t sum = 0;
            size_t i = 0;
            for (; i < n; ++i)
            {
                dst[i] = src[i];
                sum += ::abs(dst[i]);
            }
            svuint32_t _sum = svdup_n_u32(0);
            for (; i < sizeA; i += A)
            {
                svint8_t _dst = svsub_s8_x(body, svreinterpret_s8(svld1_u8(body, src + i)), svreinterpret_s8(svld1_u8(body, src + i - n)));
                svst1_s8(body, dst + i, _dst);
                AddAbsSum(_dst, body, ones, zero, _sum);
            }
            if (i < size)
            {
                svbool_t tail = svwhilelt_b8(i, size);
                svint8_t _dst = svsub_s8_x(tail, svreinterpret_s8(svld1_u8(tail, src + i)), svreinterpret_s8(svld1_u8(tail, src + i - n)));
                svst1_s8(tail, dst + i, _dst);
                AddAbsSum(_dst, tail, ones, zero, _sum);
            }
            return sum + svaddv_u32(svptrue_b32(), _sum);
        }

        ImagePngSaver::ImagePngSaver(const ImageSaverParam& param)
            : Neon::ImagePngSaver(param)
        {
            if (_param.format == SimdPixelFormatBgr24)
                _convert = Sve2::BgrToRgb;
            else if (_param.format == SimdPixelFormatBgra32)
                _convert = Sve2::BgraToRgba;
            _encode[0] = Sve2::EncodeLine0;
            _encode[1] = Sve2::EncodeLine1;
            _encode[2] = Sve2::EncodeLine2;
            _encode[3] = Sve2::EncodeLine3;
            _encode[4] = Sve2::EncodeLine4;
            _encode[5] = Sve2::EncodeLine5;
            _encode[6] = Sve2::EncodeLine6;
            _compress = Sve2::ZlibCompress;
        }
    }
#endif
}

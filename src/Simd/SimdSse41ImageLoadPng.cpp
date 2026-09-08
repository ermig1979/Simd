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
#include "Simd/SimdImageLoad.h"
#include "Simd/SimdImageSavePng.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSse41.h"
#include "Simd/SimdMemory.h"
#include "Simd/SimdUnpack.h"

namespace Simd
{
#if defined(SIMD_SSE41_ENABLE) 
    namespace Sse41
    {
        template<size_t pixelSize> SIMD_INLINE void Copy(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            Base::Copy(src, srcStride, width, height, pixelSize, dst, dstStride);
        }

        SIMD_INLINE void GrayToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            GrayToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }

        SIMD_INLINE void BgrToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            BgrToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }


        SIMD_INLINE void RgbToBgra(const uint8_t* src, size_t width, size_t height, size_t srcStride, uint8_t* dst, size_t dstStride)
        {
            RgbToBgra(src, width, height, srcStride, dst, dstStride, 0xFF);
        }

        //-------------------------------------------------------------------------------------------------

        SIMD_INLINE void FillExtra(uint8_t* dst, int srcN, int dstN)
        {
            for (int i = srcN; i < dstN; ++i)
                dst[i] = 0xFF;
        }

        template<int n> SIMD_INLINE __m128i PrefixSum(__m128i x);

        template<> SIMD_INLINE __m128i PrefixSum<1>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 1));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<2>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 2));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<3>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 3));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 6));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 12));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<4>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 4));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 8));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<6>(__m128i x)
        {
            x = _mm_add_epi8(x, _mm_slli_si128(x, 6));
            x = _mm_add_epi8(x, _mm_slli_si128(x, 12));
            return x;
        }

        template<> SIMD_INLINE __m128i PrefixSum<8>(__m128i x)
        {
            return _mm_add_epi8(x, _mm_slli_si128(x, 8));
        }

        template<int n> SIMD_INLINE __m128i FirstN()
        {
            return _mm_srli_si128(K_INV_ZERO, 16 - n);
        }

        template<int step> SIMD_INLINE void StoreStep(uint8_t* dst, __m128i x);

        template<> SIMD_INLINE void StoreStep<8>(uint8_t* dst, __m128i x)
        {
            _mm_storel_epi64((__m128i*)dst, x);
        }

        template<> SIMD_INLINE void StoreStep<12>(uint8_t* dst, __m128i x)
        {
            _mm_storel_epi64((__m128i*)dst, x);
            *(uint32_t*)(dst + 8) = _mm_extract_epi32(x, 2);
        }

        template<> SIMD_INLINE void StoreStep<15>(uint8_t* dst, __m128i x)
        {
            StoreStep<12>(dst, x);
            *(uint16_t*)(dst + 12) = (uint16_t)_mm_extract_epi16(x, 6);
            dst[14] = (uint8_t)_mm_extract_epi8(x, 14);
        }

        template<> SIMD_INLINE void StoreStep<16>(uint8_t* dst, __m128i x)
        {
            _mm_storeu_si128((__m128i*)dst, x);
        }

        template<int n, int step> SIMD_INLINE __m128i LastPixel(__m128i x)
        {
            return _mm_and_si128(_mm_srli_si128(x, step - n), FirstN<n>());
        }

        template<int n> SIMD_INLINE __m128i KeepFirst(__m128i first, __m128i rest)
        {
            return _mm_blendv_epi8(rest, first, FirstN<n>());
        }

        template<int n, int step> void DecodeSubEq(const uint8_t* curr, int width, uint8_t* dst)
        {
            int size = width * n, i = 0;
            __m128i prev = _mm_setzero_si128();
            for (; i + (int)A <= size; i += step)
            {
                __m128i x = _mm_add_epi8(_mm_loadu_si128((__m128i*)(curr + i)), prev);
                x = PrefixSum<n>(x);
                StoreStep<step>(dst + i, x);
                prev = LastPixel<n, step>(x);
            }
            if (i == 0)
            {
                for (; i < n && i < size; ++i)
                    dst[i] = curr[i];
            }
            for (; i < size; ++i)
                dst[i] = curr[i] + dst[i - n];
        }

        SIMD_INLINE __m128i Average(__m128i a, __m128i b)
        {
            return _mm_sub_epi8(_mm_avg_epu8(a, b), _mm_and_si128(_mm_xor_si128(a, b), K8_01));
        }

        template<int n, int s, int step> struct AvgShift
        {
            static SIMD_INLINE __m128i Run(__m128i x, __m128i curr, __m128i prev)
            {
                x = KeepFirst<s>(x, _mm_add_epi8(curr, Average(prev, _mm_slli_si128(x, n))));
                return AvgShift<n, s + n, step>::Run(x, curr, prev);
            }
        };

        template<int n, int step> struct AvgShift<n, step, step>
        {
            static SIMD_INLINE __m128i Run(__m128i x, __m128i, __m128i)
            {
                return x;
            }
        };

        template<int n, int step> SIMD_INLINE __m128i PrefixAvg(__m128i curr, __m128i prev, __m128i left)
        {
            return AvgShift<n, n, step>::Run(_mm_add_epi8(curr, Average(prev, left)), curr, prev);
        }

        SIMD_INLINE __m128i PaethPredictor(__m128i a, __m128i b, __m128i c)
        {
            __m128i p = _mm_sub_epi16(_mm_add_epi16(a, b), c);
            __m128i pa = _mm_abs_epi16(_mm_sub_epi16(p, a));
            __m128i pb = _mm_abs_epi16(_mm_sub_epi16(p, b));
            __m128i pc = _mm_abs_epi16(_mm_sub_epi16(p, c));
            __m128i mbc = _mm_or_si128(_mm_cmpgt_epi16(pa, pb), _mm_cmpgt_epi16(pa, pc));
            __m128i mc = _mm_cmpgt_epi16(pb, pc);
            return _mm_blendv_epi8(a, _mm_blendv_epi8(b, c, mc), mbc);
        }

        SIMD_INLINE __m128i PaethPack(__m128i a, __m128i b, __m128i c)
        {
            return _mm_packus_epi16(PaethPredictor(UnpackU8<0>(a), UnpackU8<0>(b), UnpackU8<0>(c)), K_ZERO);
        }

        template<int n, int s, int step> struct PaethShift
        {
            static SIMD_INLINE void Run(__m128i curr, __m128i prev, __m128i & a, __m128i & c, __m128i & x)
            {
                __m128i _curr = _mm_srli_si128(curr, s);
                __m128i _prev = _mm_srli_si128(prev, s);
                __m128i d = _mm_add_epi8(_curr, PaethPack(a, _prev, c));
                x = _mm_or_si128(x, _mm_slli_si128(_mm_and_si128(d, FirstN<n>()), s));
                a = d;
                c = _prev;
                PaethShift<n, s + n, step>::Run(curr, prev, a, c, x);
            }
        };

        template<int n, int step> struct PaethShift<n, step, step>
        {
            static SIMD_INLINE void Run(__m128i, __m128i, __m128i &, __m128i &, __m128i &)
            {
            }
        };

        template<int n, int step> SIMD_INLINE __m128i PrefixPaeth(__m128i curr, __m128i prev, __m128i & a, __m128i & c)
        {
            __m128i x = _mm_setzero_si128();
            PaethShift<n, 0, step>::Run(curr, prev, a, c, x);
            return x;
        }

        template<int n, int step> void DecodeAvgEq(const uint8_t* curr, const uint8_t* prev, int width, uint8_t* dst)
        {
            int size = width * n, i = 0;
            __m128i left = _mm_setzero_si128();
            for (; i + (int)A <= size; i += step)
            {
                __m128i _curr = _mm_loadu_si128((__m128i*)(curr + i));
                __m128i _prev = _mm_loadu_si128((__m128i*)(prev + i));
                __m128i x = PrefixAvg<n, step>(_curr, _prev, left);
                StoreStep<step>(dst + i, x);
                left = LastPixel<n, step>(x);
            }
            if (i == 0)
            {
                for (; i < n && i < size; ++i)
                    dst[i] = curr[i] + (prev[i] >> 1);
            }
            for (; i < size; ++i)
                dst[i] = curr[i] + ((prev[i] + dst[i - n]) >> 1);
        }

        template<int n, int step> void DecodePaethEq(const uint8_t* curr, const uint8_t* prev, int width, uint8_t* dst)
        {
            int size = width * n, i = 0;
            __m128i a = _mm_setzero_si128();
            __m128i c = _mm_setzero_si128();
            for (; i + (int)A <= size; i += step)
            {
                __m128i _curr = _mm_loadu_si128((__m128i*)(curr + i));
                __m128i _prev = _mm_loadu_si128((__m128i*)(prev + i));
                __m128i x = PrefixPaeth<n, step>(_curr, _prev, a, c);
                StoreStep<step>(dst + i, x);
            }
            if (i == 0)
            {
                for (; i < n && i < size; ++i)
                    dst[i] = curr[i] + Base::Paeth(0, prev[i], 0);
            }
            for (; i < size; ++i)
                dst[i] = curr[i] + Base::Paeth(dst[i - n], prev[i], prev[i - n]);
        }

        template<int n, int step> void DecodeAvgFirstEq(const uint8_t* curr, int width, uint8_t* dst)
        {
            int size = width * n, i = 0;
            __m128i left = _mm_setzero_si128();
            for (; i + (int)A <= size; i += step)
            {
                __m128i _curr = _mm_loadu_si128((__m128i*)(curr + i));
                __m128i x = PrefixAvg<n, step>(_curr, K_ZERO, left);
                StoreStep<step>(dst + i, x);
                left = LastPixel<n, step>(x);
            }
            if (i == 0)
            {
                for (; i < n && i < size; ++i)
                    dst[i] = curr[i];
            }
            for (; i < size; ++i)
                dst[i] = curr[i] + (dst[i - n] >> 1);
        }

        //-------------------------------------------------------------------------------------------------

        void DecodeLine0(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                int size = width * srcN, i = 0, sizeA = (int)AlignLo(size, A);
                for (; i < sizeA; i += (int)A)
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_loadu_si128((__m128i*)(curr + i)));
                for (; i < size; ++i)
                    dst[i] = curr[i];
            }
            else
            {
                for (int x = 0; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i];
                    FillExtra(dst, srcN, dstN);
                    curr += srcN;
                    dst += dstN;
                }
            }
        }

        void DecodeLine1(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                switch (srcN)
                {
                case 1: DecodeSubEq<1, 16>(curr, width, dst); break;
                case 2: DecodeSubEq<2, 16>(curr, width, dst); break;
                case 3: DecodeSubEq<3, 15>(curr, width, dst); break;
                case 4: DecodeSubEq<4, 16>(curr, width, dst); break;
                case 6: DecodeSubEq<6, 12>(curr, width, dst); break;
                case 8: DecodeSubEq<8, 16>(curr, width, dst); break;
                default:
                    for (int i = 0; i < srcN; ++i)
                        dst[i] = curr[i];
                    for (int i = srcN, n = srcN * width; i < n; ++i)
                        dst[i] = curr[i] + dst[i - dstN];
                    break;
                }
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i];
                FillExtra(dst, srcN, dstN);
                curr += srcN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + dst[i - dstN];
                    FillExtra(dst, srcN, dstN);
                    curr += srcN;
                    dst += dstN;
                }
            }
        }

        void DecodeLine2(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                int size = width * srcN, i = 0, sizeA = (int)AlignLo(size, A);
                for (; i < sizeA; i += (int)A)
                {
                    __m128i _curr = _mm_loadu_si128((__m128i*)(curr + i));
                    __m128i _prev = _mm_loadu_si128((__m128i*)(prev + i));
                    _mm_storeu_si128((__m128i*)(dst + i), _mm_add_epi8(_curr, _prev));
                }
                for (; i < size; ++i)
                    dst[i] = curr[i] + prev[i];
            }
            else
            {
                for (int x = 0; x < width; ++x)
                {
                    int i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + prev[i];
                    FillExtra(dst, srcN, dstN);
                    curr += srcN;
                    prev += dstN;
                    dst += dstN;
                }
            }
        }

        void DecodeLine3(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                switch (srcN)
                {
                case 1: DecodeAvgEq<1, 16>(curr, prev, width, dst); break;
                case 2: DecodeAvgEq<2, 16>(curr, prev, width, dst); break;
                case 3: DecodeAvgEq<3, 15>(curr, prev, width, dst); break;
                case 4: DecodeAvgEq<4, 16>(curr, prev, width, dst); break;
                case 6: DecodeAvgEq<6, 12>(curr, prev, width, dst); break;
                case 8: DecodeAvgEq<8, 16>(curr, prev, width, dst); break;
                default:
                    for (int i = 0; i < srcN; ++i)
                        dst[i] = curr[i] + (prev[i] >> 1);
                    for (int i = srcN, n = srcN * width; i < n; ++i)
                        dst[i] = curr[i] + ((prev[i] + dst[i - dstN]) >> 1);
                    break;
                }
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i] + (prev[i] >> 1);
                FillExtra(dst, srcN, dstN);
                curr += srcN;
                prev += dstN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + ((prev[i] + dst[i - dstN]) >> 1);
                    FillExtra(dst, srcN, dstN);
                    curr += srcN;
                    prev += dstN;
                    dst += dstN;
                }
            }
        }

        void DecodeLine4(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                switch (srcN)
                {
                case 1: DecodePaethEq<1, 16>(curr, prev, width, dst); break;
                case 2: DecodePaethEq<2, 16>(curr, prev, width, dst); break;
                case 3: DecodePaethEq<3, 15>(curr, prev, width, dst); break;
                case 4: DecodePaethEq<4, 16>(curr, prev, width, dst); break;
                case 6: DecodePaethEq<6, 12>(curr, prev, width, dst); break;
                case 8: DecodePaethEq<8, 16>(curr, prev, width, dst); break;
                default:
                    for (int i = 0; i < srcN; ++i)
                        dst[i] = curr[i] + Base::Paeth(0, prev[i], 0);
                    for (int i = srcN, n = srcN * width; i < n; ++i)
                        dst[i] = curr[i] + Base::Paeth(dst[i - dstN], prev[i], prev[i - dstN]);
                    break;
                }
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i] + Base::Paeth(0, prev[i], 0);
                FillExtra(dst, srcN, dstN);
                curr += srcN;
                prev += dstN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + Base::Paeth(dst[i - dstN], prev[i], prev[i - dstN]);
                    FillExtra(dst, srcN, dstN);
                    curr += srcN;
                    prev += dstN;
                    dst += dstN;
                }
            }
        }

        void DecodeLine5(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            if (srcN == dstN)
            {
                switch (srcN)
                {
                case 1: DecodeAvgFirstEq<1, 16>(curr, width, dst); break;
                case 2: DecodeAvgFirstEq<2, 16>(curr, width, dst); break;
                case 3: DecodeAvgFirstEq<3, 15>(curr, width, dst); break;
                case 4: DecodeAvgFirstEq<4, 16>(curr, width, dst); break;
                case 6: DecodeAvgFirstEq<6, 12>(curr, width, dst); break;
                case 8: DecodeAvgFirstEq<8, 16>(curr, width, dst); break;
                default:
                    for (int i = 0; i < srcN; ++i)
                        dst[i] = curr[i];
                    for (int i = srcN, n = srcN * width; i < n; ++i)
                        dst[i] = curr[i] + (dst[i - dstN] >> 1);
                    break;
                }
            }
            else
            {
                int i = 0;
                for (; i < srcN; ++i)
                    dst[i] = curr[i];
                FillExtra(dst, srcN, dstN);
                curr += srcN;
                dst += dstN;
                for (int x = 1; x < width; ++x)
                {
                    i = 0;
                    for (; i < srcN; ++i)
                        dst[i] = curr[i] + (dst[i - dstN] >> 1);
                    FillExtra(dst, srcN, dstN);
                    curr += srcN;
                    dst += dstN;
                }
            }
        }

        void DecodeLine6(const uint8_t* curr, const uint8_t* prev, int width, int srcN, int dstN, uint8_t* dst)
        {
            DecodeLine1(curr, prev, width, srcN, dstN, dst);
        }

        //-------------------------------------------------------------------------------------------------

        ImagePngLoader::ImagePngLoader(const ImageLoaderParam& param)
            : Base::ImagePngLoader(param)
        {
            if (_param.format == SimdPixelFormatNone)
                _param.format = SimdPixelFormatRgb24;
        }

        void ImagePngLoader::SetHandlers()
        {
            Base::ImagePngLoader::SetHandlers();
            if (_width >= A)
            {
                _decodeLine[0] = DecodeLine0;
                _decodeLine[1] = DecodeLine1;
                _decodeLine[2] = DecodeLine2;
                _decodeLine[3] = DecodeLine3;
                _decodeLine[4] = DecodeLine4;
                _decodeLine[5] = DecodeLine5;
                _decodeLine[6] = DecodeLine6;
                if (_depth <= 8)
                {
                    size_t channels = _paletteChannels ? _paletteChannels : _outN;
                    if (channels == 1)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = Copy<1>; break;
                        case SimdPixelFormatBgr24: _converter = GrayToBgr; break;
                        case SimdPixelFormatRgb24: _converter = GrayToBgr; break;
                        case SimdPixelFormatBgra32: _converter = GrayToBgra; break;
                        case SimdPixelFormatRgba32: _converter = GrayToBgra; break;
                        }
                    }
                    else if (channels == 3)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = RgbToGray; break;
                        case SimdPixelFormatBgr24: _converter = BgrToRgb; break;
                        case SimdPixelFormatRgb24: _converter = Copy<3>; break;
                        case SimdPixelFormatBgra32: _converter = RgbToBgra; break;
                        case SimdPixelFormatRgba32: _converter = BgrToBgra; break;
                        }
                    }
                    else if (channels == 4)
                    {
                        switch (_param.format)
                        {
                        case SimdPixelFormatGray8: _converter = RgbaToGray; break;
                        case SimdPixelFormatBgr24: _converter = BgraToRgb; break;
                        case SimdPixelFormatRgb24: _converter = BgraToBgr; break;
                        case SimdPixelFormatBgra32: _converter = BgraToRgba; break;
                        case SimdPixelFormatRgba32: _converter = Copy<4>; break;
                        }
                    }
                }
            }
        }
    }
#endif
}

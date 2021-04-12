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
#include "Simd/SimdImageSaveJpeg.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdAvx2.h"

namespace Simd
{
#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        const uint32_t JpegZigZagTi32[64] = {
            0, 8, 1, 2, 9, 16, 24, 17,
            10, 3, 4, 11, 18, 25, 32, 40,
            33, 26, 19, 12, 5, 6, 13, 20,
            27, 34, 41, 48, 56, 49, 42, 35,
            28, 21, 14, 7, 15, 22, 29, 36,
            43, 50, 57, 58, 51, 44, 37, 30,
            23, 31, 38, 45, 52, 59, 60, 53,
            46, 39, 47, 54, 61, 62, 55, 63 };

        //---------------------------------------------------------------------

        SIMD_INLINE void JpegDctV(const float* src, size_t srcStride, float *dst, size_t dstStride)
        {
            __m256 d0 = _mm256_loadu_ps(src + 0 * srcStride);
            __m256 d1 = _mm256_loadu_ps(src + 1 * srcStride);
            __m256 d2 = _mm256_loadu_ps(src + 2 * srcStride);
            __m256 d3 = _mm256_loadu_ps(src + 3 * srcStride);
            __m256 d4 = _mm256_loadu_ps(src + 4 * srcStride);
            __m256 d5 = _mm256_loadu_ps(src + 5 * srcStride);
            __m256 d6 = _mm256_loadu_ps(src + 6 * srcStride);
            __m256 d7 = _mm256_loadu_ps(src + 7 * srcStride);

            __m256 tmp0 = _mm256_add_ps(d0, d7);
            __m256 tmp7 = _mm256_sub_ps(d0, d7);
            __m256 tmp1 = _mm256_add_ps(d1, d6);
            __m256 tmp6 = _mm256_sub_ps(d1, d6);
            __m256 tmp2 = _mm256_add_ps(d2, d5);
            __m256 tmp5 = _mm256_sub_ps(d2, d5);
            __m256 tmp3 = _mm256_add_ps(d3, d4);
            __m256 tmp4 = _mm256_sub_ps(d3, d4);

            __m256 tmp10 = _mm256_add_ps(tmp0, tmp3);
            __m256 tmp13 = _mm256_sub_ps(tmp0, tmp3);
            __m256 tmp11 = _mm256_add_ps(tmp1, tmp2);
            __m256 tmp12 = _mm256_sub_ps(tmp1, tmp2);

            d0 = _mm256_add_ps(tmp10, tmp11);
            d4 = _mm256_sub_ps(tmp10, tmp11);

            __m256 z1 = _mm256_mul_ps(_mm256_add_ps(tmp12, tmp13), _mm256_set1_ps(0.707106781f));
            d2 = _mm256_add_ps(tmp13, z1);
            d6 = _mm256_sub_ps(tmp13, z1);

            tmp10 = _mm256_add_ps(tmp4, tmp5);
            tmp11 = _mm256_add_ps(tmp5, tmp6);
            tmp12 = _mm256_add_ps(tmp6, tmp7);

            __m256 z5 = _mm256_mul_ps(_mm256_sub_ps(tmp10, tmp12),  _mm256_set1_ps(0.382683433f));
            __m256 z2 = _mm256_add_ps(_mm256_mul_ps(tmp10, _mm256_set1_ps(0.541196100f)), z5);
            __m256 z4 = _mm256_add_ps(_mm256_mul_ps(tmp12, _mm256_set1_ps(1.306562965f)), z5);
            __m256 z3 = _mm256_mul_ps(tmp11, _mm256_set1_ps(0.707106781f));

            __m256 z11 = _mm256_add_ps(tmp7, z3);
            __m256 z13 = _mm256_sub_ps(tmp7, z3);

            _mm256_storeu_ps(dst + 0 * dstStride, d0);
            _mm256_storeu_ps(dst + 1 * dstStride, _mm256_add_ps(z11, z4));
            _mm256_storeu_ps(dst + 2 * dstStride, d2);
            _mm256_storeu_ps(dst + 3 * dstStride, _mm256_sub_ps(z13, z2));
            _mm256_storeu_ps(dst + 4 * dstStride, d4);
            _mm256_storeu_ps(dst + 5 * dstStride, _mm256_add_ps(z13, z2));
            _mm256_storeu_ps(dst + 6 * dstStride, d6);
            _mm256_storeu_ps(dst + 7 * dstStride, _mm256_sub_ps(z11, z4));
        }

        SIMD_INLINE void JpegDctH(const float* src, size_t srcStride, const float * fdt, int* dst)
        {
            __m256 tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
            __m256 d0 = Avx::Load<false>(src + 0 * srcStride, src + 4 * srcStride);
            __m256 d1 = Avx::Load<false>(src + 1 * srcStride, src + 5 * srcStride);
            __m256 d2 = Avx::Load<false>(src + 2 * srcStride, src + 6 * srcStride);
            __m256 d3 = Avx::Load<false>(src + 3 * srcStride, src + 7 * srcStride);
            tmp0 = _mm256_unpacklo_ps(d0, d2);
            tmp1 = _mm256_unpackhi_ps(d0, d2);
            tmp2 = _mm256_unpacklo_ps(d1, d3);
            tmp3 = _mm256_unpackhi_ps(d1, d3);
            d0 = _mm256_unpacklo_ps(tmp0, tmp2);
            d1 = _mm256_unpackhi_ps(tmp0, tmp2);
            d2 = _mm256_unpacklo_ps(tmp1, tmp3);
            d3 = _mm256_unpackhi_ps(tmp1, tmp3);

            src += 4;
            __m256 d4 = Avx::Load<false>(src + 0 * srcStride, src + 4 * srcStride);
            __m256 d5 = Avx::Load<false>(src + 1 * srcStride, src + 5 * srcStride);
            __m256 d6 = Avx::Load<false>(src + 2 * srcStride, src + 6 * srcStride);
            __m256 d7 = Avx::Load<false>(src + 3 * srcStride, src + 7 * srcStride);
            tmp0 = _mm256_unpacklo_ps(d4, d6);
            tmp1 = _mm256_unpackhi_ps(d4, d6);
            tmp2 = _mm256_unpacklo_ps(d5, d7);
            tmp3 = _mm256_unpackhi_ps(d5, d7);
            d4 = _mm256_unpacklo_ps(tmp0, tmp2);
            d5 = _mm256_unpackhi_ps(tmp0, tmp2);
            d6 = _mm256_unpacklo_ps(tmp1, tmp3);
            d7 = _mm256_unpackhi_ps(tmp1, tmp3);

            tmp0 = _mm256_add_ps(d0, d7);
            tmp1 = _mm256_add_ps(d1, d6);
            tmp2 = _mm256_add_ps(d2, d5);
            tmp3 = _mm256_add_ps(d3, d4);
            tmp7 = _mm256_sub_ps(d0, d7);
            tmp6 = _mm256_sub_ps(d1, d6);
            tmp5 = _mm256_sub_ps(d2, d5);
            tmp4 = _mm256_sub_ps(d3, d4);

            __m256 tmp10 = _mm256_add_ps(tmp0, tmp3);
            __m256 tmp13 = _mm256_sub_ps(tmp0, tmp3);
            __m256 tmp11 = _mm256_add_ps(tmp1, tmp2);
            __m256 tmp12 = _mm256_sub_ps(tmp1, tmp2);

            d0 = _mm256_add_ps(tmp10, tmp11);
            d4 = _mm256_sub_ps(tmp10, tmp11);

            __m256 z1 = _mm256_mul_ps(_mm256_add_ps(tmp12, tmp13), _mm256_set1_ps(0.707106781f));
            d2 = _mm256_add_ps(tmp13, z1);
            d6 = _mm256_sub_ps(tmp13, z1);

            tmp10 = _mm256_add_ps(tmp4, tmp5);
            tmp11 = _mm256_add_ps(tmp5, tmp6);
            tmp12 = _mm256_add_ps(tmp6, tmp7);

            __m256 z5 = _mm256_mul_ps(_mm256_sub_ps(tmp10, tmp12), _mm256_set1_ps(0.382683433f));
            __m256 z2 = _mm256_add_ps(_mm256_mul_ps(tmp10, _mm256_set1_ps(0.541196100f)), z5);
            __m256 z4 = _mm256_add_ps(_mm256_mul_ps(tmp12, _mm256_set1_ps(1.306562965f)), z5);
            __m256 z3 = _mm256_mul_ps(tmp11, _mm256_set1_ps(0.707106781f));

            __m256 z11 = _mm256_add_ps(tmp7, z3);
            __m256 z13 = _mm256_sub_ps(tmp7, z3);

            d1 = _mm256_add_ps(z11, z4);
            d3 = _mm256_sub_ps(z13, z2);
            d5 = _mm256_add_ps(z13, z2);
            d7 = _mm256_sub_ps(z11, z4);

            _mm256_storeu_si256((__m256i*)dst + 0, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 0), d0)));
            _mm256_storeu_si256((__m256i*)dst + 1, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 1), d1)));
            _mm256_storeu_si256((__m256i*)dst + 2, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 2), d2)));
            _mm256_storeu_si256((__m256i*)dst + 3, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 3), d3)));
            _mm256_storeu_si256((__m256i*)dst + 4, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 4), d4)));
            _mm256_storeu_si256((__m256i*)dst + 5, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 5), d5)));
            _mm256_storeu_si256((__m256i*)dst + 6, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 6), d6)));
            _mm256_storeu_si256((__m256i*)dst + 7, _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(fdt + F * 7), d7)));
        }

        static int JpegProcessDu(Base::BitBuf& bitBuf, float* CDU, int stride, const float* fdtbl, int DC, const uint16_t HTDC[256][2], const uint16_t HTAC[256][2])
        {
            JpegDctV(CDU, stride, CDU, stride);
            SIMD_ALIGNED(32) int DUO[64], DU[64];
            JpegDctH(CDU, stride, fdtbl, DUO);
            union
            {
                uint64_t u64[1];
                uint32_t u32[2];
                uint8_t u8[8];
            } dum;
            for (int i = 0, j = 0; i < 64; i += 8, j++)
            {
                __m256i du = _mm256_i32gather_epi32(DUO, _mm256_loadu_si256((__m256i*)(JpegZigZagTi32 + i)), 4);
                dum.u8[j] = ~_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(du, Avx2::K_ZERO)));
                _mm256_storeu_si256((__m256i*)(DU + i), du);
            }
            int diff = DU[0] - DC;
            if (diff == 0)
                bitBuf.Push(HTDC[0]);
            else
            {
                uint16_t bits[2];
                Base::JpegCalcBits(diff, bits);
                bitBuf.Push(HTDC[bits[1]]);
                bitBuf.Push(bits);
            }
#if defined(SIMD_X64_ENABLE)
            if (dum.u64[0] == 0)
            {
                bitBuf.Push(HTAC[0x00]);
                return DU[0];
            }
            dum.u64[0] >>= 1;
            int i = 1;
            for (; dum.u64[0]; ++i, dum.u64[0] >>= 1)
            {
                int nrzeroes = _tzcnt_u64(dum.u64[0]);
                i += nrzeroes;
                dum.u64[0] >>= nrzeroes;
                if (nrzeroes >= 16)
                {
                    for (int nrmarker = 16; nrmarker <= nrzeroes; nrmarker += 16)
                        bitBuf.Push(HTAC[0xF0]);
                    nrzeroes &= 15;
                }
                uint16_t bits[2];
                Base::JpegCalcBits(DU[i], bits);
                bitBuf.Push(HTAC[(nrzeroes << 4) + bits[1]]);
                bitBuf.Push(bits);
            }
            if (i < 64)
                bitBuf.Push(HTAC[0x00]);
#else
            int end0pos = 64;
            do
            {
                end0pos -= 8;
                int mask = ~_mm256_movemask_epi8(_mm256_cmpeq_epi32(_mm256_loadu_si256((__m256i*)(DU + end0pos)), Avx2::K_ZERO));
                if (mask)
                {
                    end0pos += 7 - _lzcnt_u32(mask) / 4;
                    break;
                }
            } 
            while (end0pos > 0);
            if (end0pos == 0)
            {
                bitBuf.Push(HTAC[0x00]);
                return DU[0];
            }
            for (int i = 1; i <= end0pos; ++i)
            {
                int startpos = i;
                for (; DU[i] == 0 && i <= end0pos; ++i);
                int nrzeroes = i - startpos;
                if (nrzeroes >= 16)
                {
                    int lng = nrzeroes >> 4;
                    int nrmarker;
                    for (nrmarker = 1; nrmarker <= lng; ++nrmarker)
                        bitBuf.Push(HTAC[0xF0]);
                    nrzeroes &= 15;
                }
                uint16_t bits[2];
                Base::JpegCalcBits(DU[i], bits);
                bitBuf.Push(HTAC[(nrzeroes << 4) + bits[1]]);
                bitBuf.Push(bits);
            }
            if (end0pos != 63)
                bitBuf.Push(HTAC[0x00]);
#endif
            return DU[0];
        }

        SIMD_INLINE void RgbToYuvInit(__m256 k[10])
        {
            k[0] = _mm256_set1_ps(+0.29900f);
            k[1] = _mm256_set1_ps(+0.58700f);
            k[2] = _mm256_set1_ps(+0.11400f);
            k[3] = _mm256_set1_ps(-128.000f);
            k[4] = _mm256_set1_ps(-0.16874f);
            k[5] = _mm256_set1_ps(-0.33126f);
            k[6] = _mm256_set1_ps(+0.50000f);
            k[7] = _mm256_set1_ps(+0.50000f);
            k[8] = _mm256_set1_ps(-0.41869f);
            k[9] = _mm256_set1_ps(-0.08131f);
        }

        SIMD_INLINE void RgbToYuv(const uint8_t* r, const uint8_t* g, const uint8_t* b, int stride, int height, 
            const __m256 k[10], float* y, float* u, float* v, int size)
        {
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += 8)
                {
                    __m256 _r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(r + col))));
                    __m256 _g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(g + col))));
                    __m256 _b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(b + col))));
                    _mm256_storeu_ps(y + col, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_r, k[0]), _mm256_mul_ps(_g, k[1])), _mm256_mul_ps(_b, k[2])), k[3]));
                    //_mm256_storeu_ps(y + col, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_r, _yr), _mm256_mul_ps(_g, _yg)), _mm256_add_ps(_mm256_mul_ps(_b, _yb), _yt)));
                    _mm256_storeu_ps(u + col, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_r, k[4]), _mm256_mul_ps(_g, k[5])), _mm256_mul_ps(_b, k[6])));
                    _mm256_storeu_ps(v + col, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_r, k[7]), _mm256_mul_ps(_g, k[8])), _mm256_mul_ps(_b, k[9])));
                }
                if(++row < height)
                    r += stride, g += stride, b += stride;
                y += size, u += size, v += size;
            }
        }

        SIMD_INLINE void GrayToY(const uint8_t* g, int stride, int height, const __m256 k[10], float* y, int size)
        {
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += 8)
                {
                    __m256 _g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(g + col))));
                    _mm256_storeu_ps(y + col, _mm256_add_ps(_g, k[3]));
                }
                if (++row < height)
                    g += stride;
                y += size;
            }
        }

        SIMD_INLINE void SubUv(const float * src, float * dst)
        {
            __m256 _0_25 = _mm256_set1_ps(0.25f), s0, s1;
            for (int yy = 0; yy < 8; yy += 1)
            {
                s0 = _mm256_add_ps(_mm256_loadu_ps(src + 0), _mm256_loadu_ps(src + 16));
                s1 = _mm256_add_ps(_mm256_loadu_ps(src + 8), _mm256_loadu_ps(src + 24));
                _mm256_storeu_ps(dst + 0, _mm256_mul_ps(PermutedHorizontalAdd(s0, s1), _0_25));
                src += 32;
                dst += 8;
            }
        }

        void JpegWriteBlockSubs(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            __m256 k[10];
            RgbToYuvInit(k);
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width16 = width & (~15);
            bool gray = red == green && red == blue;
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 16)
            {
                int x = 0;
                SIMD_ALIGNED(16) float Y[256], U[256], V[256];
                SIMD_ALIGNED(16) float subU[64], subV[64];
                for (; x < width16; x += 16)
                {
                    if (gray)
                        GrayToY(red + x, stride, height - y, k, Y, 16);
                    else
                        RgbToYuv(red + x, green + x, blue + x, stride, height - y, k, Y, U, V, 16);
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        SubUv(U, subU);
                        SubUv(V, subV);
                        DCU = JpegProcessDu(bitBuf, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        stream.WriteJpegBits(bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
                for (; x < width; x += 16)
                {
                    if (gray)
                        Base::GrayToY(red + x, stride, height - y, width - x, Y, 16);
                    else
                        Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 16);
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        SubUv(U, subU);
                        SubUv(V, subV);
                        DCU = JpegProcessDu(bitBuf, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
            }
            stream.WriteJpegBits(bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        void JpegWriteBlockFull(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            __m256 k[10];
            RgbToYuvInit(k);
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width8 = width & (~7);
            bool gray = red == green && red == blue;
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 8)
            {
                int x = 0;
                SIMD_ALIGNED(16) float Y[64], U[64], V[64];
                for (; x < width8; x += 8)
                {
                    if (gray)
                        GrayToY(red + x, stride, height - y, k, Y, 8);
                    else
                        RgbToYuv(red + x, green + x, blue + x, stride, height - y, k, Y, U, V, 8);
                    DCY = JpegProcessDu(bitBuf, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        stream.WriteJpegBits(bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
                for (; x < width; x += 8)
                {
                    if (gray)
                        Base::GrayToY(red + x, stride, height - y, width - x, Y, 8);
                    else
                        Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 8);
                    DCY = JpegProcessDu(bitBuf, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
                stream.WriteJpegBits(bitBuf.data, bitBuf.size);
                bitBuf.Clear();
            }
        }

        //---------------------------------------------------------------------

        ImageJpegSaver::ImageJpegSaver(const ImageSaverParam& param)
            : Sse41::ImageJpegSaver(param)
        {
        }

        void ImageJpegSaver::Init()
        {
            Sse41::ImageJpegSaver::Init();
            if (_param.width >= 32)
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24:
                case SimdPixelFormatRgb24:
                    _deintBgr = Avx2::DeinterleaveBgr;
                    break;
                case SimdPixelFormatBgra32:
                case SimdPixelFormatRgba32:
                    _deintBgra = Avx2::DeinterleaveBgra;
                    break;
                }
            }
            _writeBlock = _subSample ? JpegWriteBlockSubs : JpegWriteBlockFull;
        }
    }
#endif// SIMD_AVX2_ENABLE
}

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
#include "Simd/SimdImageSaveJpeg.h"
#include "Simd/SimdLoad.h"
#include "Simd/SimdAvx512bw.h"
#include "Simd/SimdSet.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        SIMD_INLINE void JpegDctVx2(const float* src, size_t srcStride, float *dst, size_t dstStride)
        {
            __m512 d0 = _mm512_loadu_ps(src + 0 * srcStride);
            __m512 d1 = _mm512_loadu_ps(src + 1 * srcStride);
            __m512 d2 = _mm512_loadu_ps(src + 2 * srcStride);
            __m512 d3 = _mm512_loadu_ps(src + 3 * srcStride);
            __m512 d4 = _mm512_loadu_ps(src + 4 * srcStride);
            __m512 d5 = _mm512_loadu_ps(src + 5 * srcStride);
            __m512 d6 = _mm512_loadu_ps(src + 6 * srcStride);
            __m512 d7 = _mm512_loadu_ps(src + 7 * srcStride);

            __m512 tmp0 = _mm512_add_ps(d0, d7);
            __m512 tmp7 = _mm512_sub_ps(d0, d7);
            __m512 tmp1 = _mm512_add_ps(d1, d6);
            __m512 tmp6 = _mm512_sub_ps(d1, d6);
            __m512 tmp2 = _mm512_add_ps(d2, d5);
            __m512 tmp5 = _mm512_sub_ps(d2, d5);
            __m512 tmp3 = _mm512_add_ps(d3, d4);
            __m512 tmp4 = _mm512_sub_ps(d3, d4);

            __m512 tmp10 = _mm512_add_ps(tmp0, tmp3);
            __m512 tmp13 = _mm512_sub_ps(tmp0, tmp3);
            __m512 tmp11 = _mm512_add_ps(tmp1, tmp2);
            __m512 tmp12 = _mm512_sub_ps(tmp1, tmp2);

            d0 = _mm512_add_ps(tmp10, tmp11);
            d4 = _mm512_sub_ps(tmp10, tmp11);

            __m512 z1 = _mm512_mul_ps(_mm512_add_ps(tmp12, tmp13), _mm512_set1_ps(0.707106781f));
            d2 = _mm512_add_ps(tmp13, z1);
            d6 = _mm512_sub_ps(tmp13, z1);

            tmp10 = _mm512_add_ps(tmp4, tmp5);
            tmp11 = _mm512_add_ps(tmp5, tmp6);
            tmp12 = _mm512_add_ps(tmp6, tmp7);

            __m512 z5 = _mm512_mul_ps(_mm512_sub_ps(tmp10, tmp12),  _mm512_set1_ps(0.382683433f));
            __m512 z2 = _mm512_add_ps(_mm512_mul_ps(tmp10, _mm512_set1_ps(0.541196100f)), z5);
            __m512 z4 = _mm512_add_ps(_mm512_mul_ps(tmp12, _mm512_set1_ps(1.306562965f)), z5);
            __m512 z3 = _mm512_mul_ps(tmp11, _mm512_set1_ps(0.707106781f));

            __m512 z11 = _mm512_add_ps(tmp7, z3);
            __m512 z13 = _mm512_sub_ps(tmp7, z3);

            _mm512_storeu_ps(dst + 0 * dstStride, d0);
            _mm512_storeu_ps(dst + 1 * dstStride, _mm512_add_ps(z11, z4));
            _mm512_storeu_ps(dst + 2 * dstStride, d2);
            _mm512_storeu_ps(dst + 3 * dstStride, _mm512_sub_ps(z13, z2));
            _mm512_storeu_ps(dst + 4 * dstStride, d4);
            _mm512_storeu_ps(dst + 5 * dstStride, _mm512_add_ps(z13, z2));
            _mm512_storeu_ps(dst + 6 * dstStride, d6);
            _mm512_storeu_ps(dst + 7 * dstStride, _mm512_sub_ps(z11, z4));
        }

        SIMD_INLINE void JpegDctH(const float* src, size_t srcStride, const float* fdt, int* dst)
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

            _mm512_storeu_si512(dst + F * 0, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(fdt + F * 0), Set(d0, d1))));
            _mm512_storeu_si512(dst + F * 1, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(fdt + F * 1), Set(d2, d3))));
            _mm512_storeu_si512(dst + F * 2, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(fdt + F * 2), Set(d4, d5))));
            _mm512_storeu_si512(dst + F * 3, _mm512_cvtps_epi32(_mm512_mul_ps(_mm512_loadu_ps(fdt + F * 3), Set(d6, d7))));
        }

        template<bool vert> SIMD_INLINE int JpegProcessDu(Base::BitBuf& bitBuf, float* CDU, int stride, const float* fdtbl, int DC, const uint16_t HTDC[256][2], const uint16_t HTAC[256][2])
        {
            SIMD_ALIGNED(64) int DUO[64], DU[64];
            if(vert)
                Avx2::JpegDct(CDU, stride, fdtbl, DUO);
            else
                JpegDctH(CDU, stride, fdtbl, DUO);
            union
            {
                uint64_t u64[1];
                uint32_t u32[2];
                uint16_t u16[4];
            } dum;
            for (int i = 0, j = 0; i < 64; i += 16, j++)
            {
                __m512i du = _mm512_i32gather_epi32(_mm512_loadu_si512(Avx2::JpegZigZagTi32 + i), DUO, 4);
                dum.u16[j] = _mm512_cmp_epi32_mask(du, Avx512bw::K_ZERO, _MM_CMPINT_NE);
                _mm512_storeu_si512(DU + i, du);
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
                int nrzeroes = (int)_tzcnt_u64(dum.u64[0]);
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
                end0pos -= 16;
                __mmask16 mask = ~_mm512_cmp_epi32_mask(_mm512_loadu_epi32(DU + end0pos), K_ZERO, _MM_CMPINT_EQ);
                if (mask)
                {
                    end0pos += 31 - _lzcnt_u32(mask);
                    break;
                }
            } while (end0pos > 0);
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

        SIMD_INLINE void RgbToYuvInit(__m512 k[10])
        {
            k[0] = _mm512_set1_ps(+0.29900f);
            k[1] = _mm512_set1_ps(+0.58700f);
            k[2] = _mm512_set1_ps(+0.11400f);
            k[3] = _mm512_set1_ps(-128.000f);
            k[4] = _mm512_set1_ps(-0.16874f);
            k[5] = _mm512_set1_ps(-0.33126f);
            k[6] = _mm512_set1_ps(+0.50000f);
            k[7] = _mm512_set1_ps(+0.50000f);
            k[8] = _mm512_set1_ps(-0.41869f);
            k[9] = _mm512_set1_ps(-0.08131f);
        }

        SIMD_INLINE void RgbToYuv(const uint8_t* r, const uint8_t* g, const uint8_t* b, int stride, int height, const __m256 k[10], float* y, float* u, float* v)
        {
            for (int row = 0; row < 8;)
            {
                __m256 _r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(r))));
                __m256 _g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(g))));
                __m256 _b = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(b))));
                _mm256_storeu_ps(y, _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_r, k[0]), _mm256_mul_ps(_g, k[1])), _mm256_mul_ps(_b, k[2])), k[3]));
                _mm256_storeu_ps(u, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_r, k[4]), _mm256_mul_ps(_g, k[5])), _mm256_mul_ps(_b, k[6])));
                _mm256_storeu_ps(v, _mm256_add_ps(_mm256_add_ps(_mm256_mul_ps(_r, k[7]), _mm256_mul_ps(_g, k[8])), _mm256_mul_ps(_b, k[9])));
                if(++row < height)
                    r += stride, g += stride, b += stride;
                y += 8, u += 8, v += 8;
            }
        }

        SIMD_INLINE void RgbToYuv(const uint8_t* r, const uint8_t* g, const uint8_t* b, int stride, int height, const __m512 k[10], float* y, float* u, float* v)
        {
            for (int row = 0; row < 16;)
            {
                __m512 _r = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(r))));
                __m512 _g = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(g))));
                __m512 _b = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(b))));
                _mm512_storeu_ps(y, _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(_r, k[0]), _mm512_mul_ps(_g, k[1])), _mm512_mul_ps(_b, k[2])), k[3]));
                _mm512_storeu_ps(u, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(_r, k[4]), _mm512_mul_ps(_g, k[5])), _mm512_mul_ps(_b, k[6])));
                _mm512_storeu_ps(v, _mm512_add_ps(_mm512_add_ps(_mm512_mul_ps(_r, k[7]), _mm512_mul_ps(_g, k[8])), _mm512_mul_ps(_b, k[9])));
                if (++row < height)
                    r += stride, g += stride, b += stride;
                y += 16, u += 16, v += 16;
            }
        }

        template<int size> void GrayToY(const uint8_t* g, int stride, int height, float* y);

        template<> SIMD_INLINE void GrayToY<8>(const uint8_t* g, int stride, int height, float* y)
        {
            __m256 k = _mm256_set1_ps(-128.000f);
            for (int row = 0; row < 8;)
            {
                __m256 _g = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((__m128i*)(g))));
                _mm256_storeu_ps(y, _mm256_add_ps(_g, k));
                if (++row < height)
                    g += stride;
                y += 8;
            }
        }

        template<> SIMD_INLINE void GrayToY<16>(const uint8_t* g, int stride, int height, float* y)
        {
            __m512 k = _mm512_set1_ps(-128.000f);
            for (int row = 0; row < 16;)
            {
                __m512 _g = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((__m128i*)(g))));
                _mm512_storeu_ps(y, _mm512_add_ps(_g, k));
                if (++row < height)
                    g += stride;
                y += 16;
            }
        }

        SIMD_INLINE void SubUv(const float * src, float * dst)
        {
            __m256 _0_25 = _mm256_set1_ps(0.25f), s0, s1;
            for (int yy = 0; yy < 8; yy += 1)
            {
                s0 = _mm256_add_ps(_mm256_loadu_ps(src + 0), _mm256_loadu_ps(src + 16));
                s1 = _mm256_add_ps(_mm256_loadu_ps(src + 8), _mm256_loadu_ps(src + 24));
                _mm256_storeu_ps(dst + 0, _mm256_mul_ps(Avx2::PermutedHorizontalAdd(s0, s1), _0_25));
                src += 32;
                dst += 8;
            }
        }

        void JpegWriteBlockSubs(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            bool gray = red == green && red == blue;
            __m512 k[10];
            if(!gray)
                RgbToYuvInit(k);
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width16 = width & (~15);
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 16)
            {
                int x = 0;
                SIMD_ALIGNED(64) float Y[256], U[256], V[256];
                SIMD_ALIGNED(64) float subU[64], subV[64];
                for (; x < width16; x += 16)
                {
                    if (gray)
                        GrayToY<16>(red + x, stride, height - y, Y);
                    else
                        RgbToYuv(red + x, green + x, blue + x, stride, height - y, k, Y, U, V);
                    JpegDctVx2(Y + 0, 16, Y + 0, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    JpegDctVx2(Y + 128, 16, Y + 128, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        SubUv(U, subU);
                        SubUv(V, subV);
                        DCU = JpegProcessDu<true>(bitBuf, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu<true>(bitBuf, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
                for (; x < width; x += 16)
                {
                    if (gray)
                        Base::GrayToY(red + x, stride, height - y, width - x, Y, 16);
                    else
                        Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 16);
                    JpegDctVx2(Y + 0, 16, Y + 0, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    JpegDctVx2(Y + 128, 16, Y + 128, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        SubUv(U, subU);
                        SubUv(V, subV);
                        DCU = JpegProcessDu<true>(bitBuf, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu<true>(bitBuf, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        void JpegWriteBlockFull(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            bool gray = red == green && red == blue;
            __m256 k[10];
            if(!gray)
                RgbToYuvInit(k);
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width8 = width & (~7);
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 8)
            {
                int x = 0;
                SIMD_ALIGNED(16) float Y[64], U[64], V[64];
                for (; x < width8; x += 8)
                {
                    if (gray)
                        GrayToY<8>(red + x, stride, height - y, Y);
                    else
                        RgbToYuv(red + x, green + x, blue + x, stride, height - y, k, Y, U, V);
                    DCY = JpegProcessDu<true>(bitBuf, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        DCU = JpegProcessDu<true>(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu<true>(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
                for (; x < width; x += 8)
                {
                    if (gray)
                        Base::GrayToY(red + x, stride, height - y, width - x, Y, 8);
                    else
                        Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 8);
                    DCY = JpegProcessDu<true>(bitBuf, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        DCU = JpegProcessDu<true>(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu<true>(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
                Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                bitBuf.Clear();
            }
        }

        void JpegWriteBlockNv12(OutputMemoryStream& stream, int width, int height, const uint8_t* ySrc, int yStride,
            const uint8_t* uvSrc, int uvStride, const float* fY, const float* fUv, int dc[3])
        {
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width16 = width & (~15);
            SIMD_ALIGNED(64) float Y[256], U[64], V[64];
            bool gray = (uvSrc == NULL);
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 16)
            {
                int x = 0;
                for (; x < width16; x += 16)
                {
                    GrayToY<16>(ySrc + x, yStride, height - y, Y);
                    JpegDctVx2(Y + 0, 16, Y + 0, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    JpegDctVx2(Y + 128, 16, Y + 128, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        Avx2::Nv12ToUv(uvSrc + x, uvStride, Base::UvSize(height - y), U, V);
                        DCU = JpegProcessDu<true>(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu<true>(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
                for (; x < width; x += 16)
                {
                    Base::GrayToY(ySrc + x, yStride, height - y, width - x, Y, 16);
                    JpegDctVx2(Y + 0, 16, Y + 0, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    JpegDctVx2(Y + 128, 16, Y + 128, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        Base::Nv12ToUv(uvSrc + x, uvStride, Base::UvSize(height - y), Base::UvSize(width - x), U, V);
                        DCU = JpegProcessDu<true>(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu<true>(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        void JpegWriteBlockYuv420p(OutputMemoryStream& stream, int width, int height, const uint8_t* ySrc, int yStride,
            const uint8_t* uSrc, int uStride, const uint8_t* vSrc, int vStride, const float* fY, const float* fUv, int dc[3])
        {
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width16 = width & (~15);
            SIMD_ALIGNED(64) float Y[256], U[64], V[64];
            bool gray = (uSrc == NULL || vSrc == NULL);
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 16)
            {
                int x = 0;
                for (; x < width16; x += 16)
                {
                    GrayToY<16>(ySrc + x, yStride, height - y, Y);
                    JpegDctVx2(Y + 0, 16, Y + 0, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    JpegDctVx2(Y + 128, 16, Y + 128, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        GrayToY<8>(uSrc + Base::UvSize(x), uStride, Base::UvSize(height - y), U);
                        GrayToY<8>(vSrc + Base::UvSize(x), vStride, Base::UvSize(height - y), V);
                        DCU = JpegProcessDu<true>(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu<true>(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                    if (bitBuf.Full())
                    {
                        Base::WriteBits(stream, bitBuf.data, bitBuf.size);
                        bitBuf.Clear();
                    }
                }
                for (; x < width; x += 16)
                {
                    Base::GrayToY(ySrc + x, yStride, height - y, width - x, Y, 16);
                    JpegDctVx2(Y + 0, 16, Y + 0, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    JpegDctVx2(Y + 128, 16, Y + 128, 16);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu<false>(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        Base::GrayToY(uSrc + Base::UvSize(x), uStride, Base::UvSize(height - y), Base::UvSize(width - x), U, 8);
                        Base::GrayToY(vSrc + Base::UvSize(x), vStride, Base::UvSize(height - y), Base::UvSize(width - x), V, 8);
                        DCU = JpegProcessDu<true>(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu<true>(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        //---------------------------------------------------------------------

        ImageJpegSaver::ImageJpegSaver(const ImageSaverParam& param)
            : Avx2::ImageJpegSaver(param)
        {
        }

        void ImageJpegSaver::Init()
        {
            Avx2::ImageJpegSaver::Init();
            if (_param.yuvType == SimdYuvUnknown)
            {
                if (_param.width > 32)
                {
                    switch (_param.format)
                    {
                    case SimdPixelFormatBgr24:
                    case SimdPixelFormatRgb24:
                        _deintBgr = Avx512bw::DeinterleaveBgr;
                        break;
                    case SimdPixelFormatBgra32:
                    case SimdPixelFormatRgba32:
                        _deintBgra = Avx512bw::DeinterleaveBgra;
                        break;
                    default:
                        break;
                    }
                }
                _writeBlock = _subSample ? JpegWriteBlockSubs : JpegWriteBlockFull;
            }
            else
            {
                _writeNv12Block = JpegWriteBlockNv12;
                _writeYuv420pBlock = JpegWriteBlockYuv420p;
            }
        }

        //---------------------------------------------------------------------

        uint8_t* Nv12SaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* uv, size_t uvStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size)
        {
            ImageSaverParam param(width, height, quality, yuvType);
            if (param.Validate())
            {
                Holder<ImageJpegSaver> saver(new ImageJpegSaver(param));
                if (saver)
                {
                    if (saver->ToStream(y, yStride, uv, uvStride))
                        return saver->Release(size);
                }
            }
            return NULL;
        }

        uint8_t* Yuv420pSaveAsJpegToMemory(const uint8_t* y, size_t yStride, const uint8_t* u, size_t uStride, const uint8_t* v, size_t vStride, size_t width, size_t height, SimdYuvType yuvType, int quality, size_t* size)
        {
            ImageSaverParam param(width, height, quality, yuvType);
            if (param.Validate())
            {
                Holder<ImageJpegSaver> saver(new ImageJpegSaver(param));
                if (saver)
                {
                    if (saver->ToStream(y, yStride, u, uStride, v, vStride))
                        return saver->Release(size);
                }
            }
            return NULL;
        }
    }
#endif
}

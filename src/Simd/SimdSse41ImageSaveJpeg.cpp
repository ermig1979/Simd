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
#include "Simd/SimdBase.h"
#include "Simd/SimdSsse3.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE void JpegDct(float* d0p, float* d1p, float* d2p, float* d3p, float* d4p, float* d5p, float* d6p, float* d7p)
        {
            float d0 = *d0p, d1 = *d1p, d2 = *d2p, d3 = *d3p, d4 = *d4p, d5 = *d5p, d6 = *d6p, d7 = *d7p;
            float z1, z2, z3, z4, z5, z11, z13;
            float tmp0 = d0 + d7;
            float tmp7 = d0 - d7;
            float tmp1 = d1 + d6;
            float tmp6 = d1 - d6;
            float tmp2 = d2 + d5;
            float tmp5 = d2 - d5;
            float tmp3 = d3 + d4;
            float tmp4 = d3 - d4;

            float tmp10 = tmp0 + tmp3;
            float tmp13 = tmp0 - tmp3;
            float tmp11 = tmp1 + tmp2;
            float tmp12 = tmp1 - tmp2;

            d0 = tmp10 + tmp11;
            d4 = tmp10 - tmp11;

            z1 = (tmp12 + tmp13) * 0.707106781f;
            d2 = tmp13 + z1;
            d6 = tmp13 - z1;

            tmp10 = tmp4 + tmp5;
            tmp11 = tmp5 + tmp6;
            tmp12 = tmp6 + tmp7;

            z5 = (tmp10 - tmp12) * 0.382683433f;
            z2 = tmp10 * 0.541196100f + z5;
            z4 = tmp12 * 1.306562965f + z5;
            z3 = tmp11 * 0.707106781f;

            z11 = tmp7 + z3;
            z13 = tmp7 - z3;

            *d5p = z13 + z2;
            *d3p = z13 - z2;
            *d1p = z11 + z4;
            *d7p = z11 - z4;

            *d0p = d0;  *d2p = d2;  *d4p = d4;  *d6p = d6;
        }

        SIMD_INLINE void JpegDctV(float* data, size_t stride)
        {
            for (float * end = data + 8; data < end; data += 4)
            {
                __m128 d0 = _mm_loadu_ps(data + 0 * stride);
                __m128 d1 = _mm_loadu_ps(data + 1 * stride);
                __m128 d2 = _mm_loadu_ps(data + 2 * stride);
                __m128 d3 = _mm_loadu_ps(data + 3 * stride);
                __m128 d4 = _mm_loadu_ps(data + 4 * stride);
                __m128 d5 = _mm_loadu_ps(data + 5 * stride);
                __m128 d6 = _mm_loadu_ps(data + 6 * stride);
                __m128 d7 = _mm_loadu_ps(data + 7 * stride);

                __m128 tmp0 = _mm_add_ps(d0, d7);
                __m128 tmp7 = _mm_sub_ps(d0, d7);
                __m128 tmp1 = _mm_add_ps(d1, d6);
                __m128 tmp6 = _mm_sub_ps(d1, d6);
                __m128 tmp2 = _mm_add_ps(d2, d5);
                __m128 tmp5 = _mm_sub_ps(d2, d5);
                __m128 tmp3 = _mm_add_ps(d3, d4);
                __m128 tmp4 = _mm_sub_ps(d3, d4);

                __m128 tmp10 = _mm_add_ps(tmp0, tmp3);
                __m128 tmp13 = _mm_sub_ps(tmp0, tmp3);
                __m128 tmp11 = _mm_add_ps(tmp1, tmp2);
                __m128 tmp12 = _mm_sub_ps(tmp1, tmp2);

                d0 = _mm_add_ps(tmp10, tmp11);
                d4 = _mm_sub_ps(tmp10, tmp11);

                __m128 z1 = _mm_mul_ps(_mm_add_ps(tmp12, tmp13), _mm_set1_ps(0.707106781f));
                d2 = _mm_add_ps(tmp13, z1);
                d6 = _mm_sub_ps(tmp13, z1);

                tmp10 = _mm_add_ps(tmp4, tmp5);
                tmp11 = _mm_add_ps(tmp5, tmp6);
                tmp12 = _mm_add_ps(tmp6, tmp7);

                __m128 z5 = _mm_mul_ps(_mm_sub_ps(tmp10, tmp12),  _mm_set1_ps(0.382683433f));
                __m128 z2 = _mm_add_ps(_mm_mul_ps(tmp10, _mm_set1_ps(0.541196100f)), z5);
                __m128 z4 = _mm_add_ps(_mm_mul_ps(tmp12, _mm_set1_ps(1.306562965f)), z5);
                __m128 z3 = _mm_mul_ps(tmp11, _mm_set1_ps(0.707106781f));

                __m128 z11 = _mm_add_ps(tmp7, z3);
                __m128 z13 = _mm_sub_ps(tmp7, z3);

                _mm_storeu_ps(data + 0 * stride, d0);
                _mm_storeu_ps(data + 1 * stride, _mm_add_ps(z11, z4));
                _mm_storeu_ps(data + 2 * stride, d2);
                _mm_storeu_ps(data + 3 * stride, _mm_sub_ps(z13, z2));
                _mm_storeu_ps(data + 4 * stride, d4);
                _mm_storeu_ps(data + 5 * stride, _mm_add_ps(z13, z2));
                _mm_storeu_ps(data + 6 * stride, d6);
                _mm_storeu_ps(data + 7 * stride, _mm_sub_ps(z11, z4));
            }
        }

        static int JpegProcessDu(OutputMemoryStream& stream, float* CDU, int stride, const float* fdtbl, int DC, const uint16_t HTDC[256][2], const uint16_t HTAC[256][2])
        {
            int offs, i, j, n, diff, end0pos, x, y;
            for (offs = 0, n = stride * 8; offs < n; offs += stride)
                JpegDct(&CDU[offs], &CDU[offs + 1], &CDU[offs + 2], &CDU[offs + 3], &CDU[offs + 4], &CDU[offs + 5], &CDU[offs + 6], &CDU[offs + 7]);
            JpegDctV(CDU, stride);
            //for (offs = 0; offs < 8; ++offs)
            //    JpegDct(&CDU[offs], &CDU[offs + stride], &CDU[offs + stride * 2], &CDU[offs + stride * 3], &CDU[offs + stride * 4],
            //        &CDU[offs + stride * 5], &CDU[offs + stride * 6], &CDU[offs + stride * 7]);
            int DU[64];
            for (y = 0, j = 0; y < 8; ++y)
            {
                for (x = 0; x < 8; ++x, ++j)
                {
                    i = y * stride + x;
                    float v = CDU[i] * fdtbl[j];
                    DU[Base::JpegZigZag[j]] = Round(v);
                }
            }
            diff = DU[0] - DC;
            if (diff == 0)
                stream.WriteJpegBits(HTDC[0]);
            else
            {
                uint16_t bits[2];
                Base::JpegCalcBits(diff, bits);
                stream.WriteJpegBits(HTDC[bits[1]]);
                stream.WriteJpegBits(bits);
            }
            end0pos = 63;
            for (; (end0pos > 0) && (DU[end0pos] == 0); --end0pos);
            if (end0pos == 0)
            {
                stream.WriteJpegBits(HTAC[0x00]);
                return DU[0];
            }
            for (i = 1; i <= end0pos; ++i)
            {
                int startpos = i;
                int nrzeroes;
                uint16_t bits[2];
                for (; DU[i] == 0 && i <= end0pos; ++i);
                nrzeroes = i - startpos;
                if (nrzeroes >= 16)
                {
                    int lng = nrzeroes >> 4;
                    int nrmarker;
                    for (nrmarker = 1; nrmarker <= lng; ++nrmarker)
                        stream.WriteJpegBits(HTAC[0xF0]);
                    nrzeroes &= 15;
                }
                Base::JpegCalcBits(DU[i], bits);
                stream.WriteJpegBits(HTAC[(nrzeroes << 4) + bits[1]]);
                stream.WriteJpegBits(bits);
            }
            if (end0pos != 63)
                stream.WriteJpegBits(HTAC[0x00]);
            return DU[0];
        }

        SIMD_INLINE void RgbToYuvInit(__m128 k[10])
        {
            k[0] = _mm_set1_ps(+0.29900f);
            k[1] = _mm_set1_ps(+0.58700f);
            k[2] = _mm_set1_ps(+0.11400f);
            k[3] = _mm_set1_ps(-128.000f);
            k[4] = _mm_set1_ps(-0.16874f);
            k[5] = _mm_set1_ps(-0.33126f);
            k[6] = _mm_set1_ps(+0.50000f);
            k[7] = _mm_set1_ps(+0.50000f);
            k[8] = _mm_set1_ps(-0.41869f);
            k[9] = _mm_set1_ps(-0.08131f);
        }

        SIMD_INLINE void RgbToYuv(const uint8_t* r, const uint8_t* g, const uint8_t* b, int stride, int height, 
            const __m128 k[10], float* y, float* u, float* v, int size)
        {
            for (int row = 0; row < size; ++row)
            {
                for (int col = 0; col < size; col += 4)
                {
                    __m128 _r = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(r + col))));
                    __m128 _g = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(g + col))));
                    __m128 _b = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(b + col))));
                    _mm_storeu_ps(y + col, _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_mul_ps(_r, k[0]), _mm_mul_ps(_g, k[1])), _mm_mul_ps(_b, k[2])), k[3]));
                    //_mm_storeu_ps(y + col, _mm_add_ps(_mm_add_ps(_mm_mul_ps(_r, _yr), _mm_mul_ps(_g, _yg)), _mm_add_ps(_mm_mul_ps(_b, _yb), _yt)));
                    _mm_storeu_ps(u + col, _mm_add_ps(_mm_add_ps(_mm_mul_ps(_r, k[4]), _mm_mul_ps(_g, k[5])), _mm_mul_ps(_b, k[6])));
                    _mm_storeu_ps(v + col, _mm_add_ps(_mm_add_ps(_mm_mul_ps(_r, k[7]), _mm_mul_ps(_g, k[8])), _mm_mul_ps(_b, k[9])));
                }
                if(row < height)
                    r += stride, g += stride, b += stride;
                y += size, u += size, v += size;
            }
        }

        SIMD_INLINE void SubUv(const float * src, float * dst)
        {
            __m128 _0_25 = _mm_set1_ps(0.25f), s0, s1;
            for (int yy = 0; yy < 8; yy += 1)
            {
                s0 = _mm_add_ps(_mm_loadu_ps(src + 0), _mm_loadu_ps(src + 16));
                s1 = _mm_add_ps(_mm_loadu_ps(src + 4), _mm_loadu_ps(src + 20));
                _mm_storeu_ps(dst + 0, _mm_mul_ps(_mm_hadd_ps(s0, s1), _0_25));
                s0 = _mm_add_ps(_mm_loadu_ps(src + 8), _mm_loadu_ps(src + 24));
                s1 = _mm_add_ps(_mm_loadu_ps(src + 12), _mm_loadu_ps(src + 28));
                _mm_storeu_ps(dst + 4, _mm_mul_ps(_mm_hadd_ps(s0, s1), _0_25));
                src += 32;
                dst += 8;
            }
        }

        void JpegWriteBlockSubs(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            __m128 k[10];
            RgbToYuvInit(k);
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width16 = width& (~15);
            for (int y = 0; y < height; y += 16)
            {
                int x = 0;
                SIMD_ALIGNED(16) float Y[256], U[256], V[256];
                SIMD_ALIGNED(16) float subU[64], subV[64];
                for (; x < width16; x += 16)
                {
                    RgbToYuv(red + x, green + x, blue + x, stride, height - y, k, Y, U, V, 16);
                    DCY = JpegProcessDu(stream, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    SubUv(U, subU);
                    SubUv(V, subV);
                    DCU = JpegProcessDu(stream, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                    DCV = JpegProcessDu(stream, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                }
                for (; x < width; x += 16)
                {
                    Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 16);
                    DCY = JpegProcessDu(stream, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    SubUv(U, subU);
                    SubUv(V, subV);
                    DCU = JpegProcessDu(stream, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                    DCV = JpegProcessDu(stream, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                }
            }
        }

        void JpegWriteBlockFull(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            __m128 k[10];
            RgbToYuvInit(k);
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width8 = width & (~7);
            for (int y = 0; y < height; y += 8)
            {
                int x = 0;
                SIMD_ALIGNED(16) float Y[64], U[64], V[64];
                for (; x < width8; x += 8)
                {
                    RgbToYuv(red + x, green + x, blue + x, stride, height - y, k, Y, U, V, 8);
                    DCY = JpegProcessDu(stream, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCU = JpegProcessDu(stream, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                    DCV = JpegProcessDu(stream, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                }
                for (; x < width; x += 8)
                {
                    Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 8);
                    DCY = JpegProcessDu(stream, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCU = JpegProcessDu(stream, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                    DCV = JpegProcessDu(stream, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                }
            }
        }

        //---------------------------------------------------------------------

        ImageJpegSaver::ImageJpegSaver(const ImageSaverParam& param)
            : Base::ImageJpegSaver(param)
        {
            switch (_param.format)
            {
            case SimdPixelFormatBgr24:
            case SimdPixelFormatRgb24:
                _deintBgr = Ssse3::DeinterleaveBgr;
                break;
            case SimdPixelFormatBgra32:
            case SimdPixelFormatRgba32:
                _deintBgra = Ssse3::DeinterleaveBgra;
                break;
            }
            _writeBlock = _subSample ? JpegWriteBlockSubs : JpegWriteBlockFull;
        }
    }
#endif// SIMD_SSE41_ENABLE
}

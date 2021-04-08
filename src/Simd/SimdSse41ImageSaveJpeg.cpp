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
#include "Simd/SimdSsse3.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
    {
        SIMD_INLINE void JpegDctV(const float* src, size_t srcStride, float *dst, size_t dstStride)
        {
            for (int i = 0; i < 2; i++, src += 4, dst += 4)
            {
                __m128 d0 = _mm_loadu_ps(src + 0 * srcStride);
                __m128 d1 = _mm_loadu_ps(src + 1 * srcStride);
                __m128 d2 = _mm_loadu_ps(src + 2 * srcStride);
                __m128 d3 = _mm_loadu_ps(src + 3 * srcStride);
                __m128 d4 = _mm_loadu_ps(src + 4 * srcStride);
                __m128 d5 = _mm_loadu_ps(src + 5 * srcStride);
                __m128 d6 = _mm_loadu_ps(src + 6 * srcStride);
                __m128 d7 = _mm_loadu_ps(src + 7 * srcStride);

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

                _mm_storeu_ps(dst + 0 * dstStride, d0);
                _mm_storeu_ps(dst + 1 * dstStride, _mm_add_ps(z11, z4));
                _mm_storeu_ps(dst + 2 * dstStride, d2);
                _mm_storeu_ps(dst + 3 * dstStride, _mm_sub_ps(z13, z2));
                _mm_storeu_ps(dst + 4 * dstStride, d4);
                _mm_storeu_ps(dst + 5 * dstStride, _mm_add_ps(z13, z2));
                _mm_storeu_ps(dst + 6 * dstStride, d6);
                _mm_storeu_ps(dst + 7 * dstStride, _mm_sub_ps(z11, z4));
            }
        }

        SIMD_INLINE void JpegDctH(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            for (int i = 0; i < 2; i++, src += 4 * srcStride, dst += 4)
            {
                __m128 tmp0, tmp1, tmp2, tmp3;
                __m128 d0 = _mm_loadu_ps(src + 0 * srcStride);
                __m128 d1 = _mm_loadu_ps(src + 1 * srcStride);
                __m128 d2 = _mm_loadu_ps(src + 2 * srcStride);
                __m128 d3 = _mm_loadu_ps(src + 3 * srcStride);
                tmp0 = _mm_unpacklo_ps(d0, d2);
                tmp1 = _mm_unpackhi_ps(d0, d2);
                tmp2 = _mm_unpacklo_ps(d1, d3);
                tmp3 = _mm_unpackhi_ps(d1, d3);
                d0 = _mm_unpacklo_ps(tmp0, tmp2);
                d1 = _mm_unpackhi_ps(tmp0, tmp2);
                d2 = _mm_unpacklo_ps(tmp1, tmp3);
                d3 = _mm_unpackhi_ps(tmp1, tmp3);

                __m128 d4 = _mm_loadu_ps(src + 0 * srcStride + 4);
                __m128 d5 = _mm_loadu_ps(src + 1 * srcStride + 4);
                __m128 d6 = _mm_loadu_ps(src + 2 * srcStride + 4);
                __m128 d7 = _mm_loadu_ps(src + 3 * srcStride + 4);
                tmp0 = _mm_unpacklo_ps(d4, d6);
                tmp1 = _mm_unpackhi_ps(d4, d6);
                tmp2 = _mm_unpacklo_ps(d5, d7);
                tmp3 = _mm_unpackhi_ps(d5, d7);
                d4 = _mm_unpacklo_ps(tmp0, tmp2);
                d5 = _mm_unpackhi_ps(tmp0, tmp2);
                d6 = _mm_unpacklo_ps(tmp1, tmp3);
                d7 = _mm_unpackhi_ps(tmp1, tmp3);

                tmp0 = _mm_add_ps(d0, d7);
                tmp1 = _mm_add_ps(d1, d6);
                tmp2 = _mm_add_ps(d2, d5);
                tmp3 = _mm_add_ps(d3, d4);
                __m128 tmp7 = _mm_sub_ps(d0, d7);
                __m128 tmp6 = _mm_sub_ps(d1, d6);
                __m128 tmp5 = _mm_sub_ps(d2, d5);
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

                __m128 z5 = _mm_mul_ps(_mm_sub_ps(tmp10, tmp12), _mm_set1_ps(0.382683433f));
                __m128 z2 = _mm_add_ps(_mm_mul_ps(tmp10, _mm_set1_ps(0.541196100f)), z5);
                __m128 z4 = _mm_add_ps(_mm_mul_ps(tmp12, _mm_set1_ps(1.306562965f)), z5);
                __m128 z3 = _mm_mul_ps(tmp11, _mm_set1_ps(0.707106781f));

                __m128 z11 = _mm_add_ps(tmp7, z3);
                __m128 z13 = _mm_sub_ps(tmp7, z3);

                d1 = _mm_add_ps(z11, z4);
                d3 = _mm_sub_ps(z13, z2);
                d5 = _mm_add_ps(z13, z2);
                d7 = _mm_sub_ps(z11, z4);

                _mm_storeu_ps(dst + 0 * dstStride, d0);
                _mm_storeu_ps(dst + 1 * dstStride, d1);
                _mm_storeu_ps(dst + 2 * dstStride, d2);
                _mm_storeu_ps(dst + 3 * dstStride, d3);
                _mm_storeu_ps(dst + 4 * dstStride, d4);
                _mm_storeu_ps(dst + 5 * dstStride, d5);
                _mm_storeu_ps(dst + 6 * dstStride, d6);
                _mm_storeu_ps(dst + 7 * dstStride, d7);
            }
        }

        static int JpegProcessDu(OutputMemoryStream& stream, float* CDU, int stride, const float* fdtbl, int DC, const uint16_t HTDC[256][2], const uint16_t HTAC[256][2])
        {
            JpegDctV(CDU, stride, CDU, stride);
            SIMD_ALIGNED(16) float CDUT[64];
            JpegDctH(CDU, stride, CDUT, 8);
            SIMD_ALIGNED(16) int DU[64];
            for (int i = 0; i < 64; ++i)
            {
                float v = CDUT[i] * fdtbl[i];
                DU[Base::JpegZigZagT[i]] = Round(v);
            }
            uint16_t buf[128][2];
            int len = 0;
            int diff = DU[0] - DC;
            if (diff == 0)
                Base::PushBits(buf[len++], HTDC[0]);
            else
            {
                uint16_t bits[2];
                Base::JpegCalcBits(diff, bits);
                Base::PushBits(buf[len++], HTDC[bits[1]]);
                Base::PushBits(buf[len++], bits);
            }
            int end0pos4 = 60;
            for (; end0pos4 > 0 && _mm_testz_si128(_mm_loadu_si128((__m128i*)(DU + end0pos4)), Sse2::K_INV_ZERO); end0pos4 -= 4);
            int end0pos = end0pos4 + 3;
            for (; (end0pos > 0) && (DU[end0pos] == 0); --end0pos);
            if (end0pos == 0)
            {
                Base::PushBits(buf[len++], HTAC[0x00]);
                stream.WriteJpegBits(buf, len);
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
                        Base::PushBits(buf[len++], HTAC[0xF0]);
                    nrzeroes &= 15;
                }
                uint16_t bits[2];
                Base::JpegCalcBits(DU[i], bits);
                Base::PushBits(buf[len++], HTAC[(nrzeroes << 4) + bits[1]]);
                Base::PushBits(buf[len++], bits);
            }
            if (end0pos != 63)
                Base::PushBits(buf[len++], HTAC[0x00]);
            stream.WriteJpegBits(buf, len);
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
            for (int row = 0; row < size;)
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
                if(++row < height)
                    r += stride, g += stride, b += stride;
                y += size, u += size, v += size;
            }
        }

        SIMD_INLINE void GrayToY(const uint8_t* g, int stride, int height, const __m128 k[10], float* y, int size)
        {
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += 4)
                {
                    __m128 _g = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(int32_t*)(g + col))));
                    _mm_storeu_ps(y + col, _mm_add_ps(_g, k[3]));
                }
                if (++row < height)
                    g += stride;
                y += size;
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
            bool gray = red == green && red == blue;
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
                    DCY = JpegProcessDu(stream, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(stream);
                    else
                    {
                        SubUv(U, subU);
                        SubUv(V, subV);
                        DCU = JpegProcessDu(stream, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(stream, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
                for (; x < width; x += 16)
                {
                    if (gray)
                        Base::GrayToY(red + x, stride, height - y, width - x, Y, 16);
                    else
                        Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 16);
                    DCY = JpegProcessDu(stream, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(stream, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(stream);
                    else
                    {
                        SubUv(U, subU);
                        SubUv(V, subV);
                        DCU = JpegProcessDu(stream, subU, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(stream, subV, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
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
            bool gray = red == green && red == blue;
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
                    DCY = JpegProcessDu(stream, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(stream);
                    else
                    {
                        DCU = JpegProcessDu(stream, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(stream, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
                for (; x < width; x += 8)
                {
                    if (gray)
                        Base::GrayToY(red + x, stride, height - y, width - x, Y, 8);
                    else
                        Base::RgbToYuv(red + x, green + x, blue + x, stride, height - y, width - x, Y, U, V, 8);
                    DCY = JpegProcessDu(stream, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(stream);
                    else
                    {
                        DCU = JpegProcessDu(stream, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(stream, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
            }
        }

        //---------------------------------------------------------------------

        ImageJpegSaver::ImageJpegSaver(const ImageSaverParam& param)
            : Base::ImageJpegSaver(param)
        {
        }

        void ImageJpegSaver::Init()
        {
            InitParams(true);
            switch (_param.format)
            {
            case SimdPixelFormatBgr24:
            case SimdPixelFormatRgb24:
                _deintBgr = _param.width < 16 ? Base::DeinterleaveBgr : Ssse3::DeinterleaveBgr;
                break;
            case SimdPixelFormatBgra32:
            case SimdPixelFormatRgba32:
                _deintBgra = _param.width < 16 ? Base::DeinterleaveBgra : Ssse3::DeinterleaveBgra;
                break;
            }
            _writeBlock = _subSample ? JpegWriteBlockSubs : JpegWriteBlockFull;
        }
    }
#endif// SIMD_SSE41_ENABLE
}
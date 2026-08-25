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
#include "Simd/SimdImageSaveJpeg.h"
#include "Simd/SimdSve2.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svint32_t Round(const svbool_t& mask, const svfloat32_t& value)
        {
            svbool_t pos = svcmpge_n_f32(mask, value, 0.0f);
            svfloat32_t round = svsel_f32(pos, svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svcvt_s32_f32_x(mask, svadd_f32_x(mask, value, round));
        }

        SIMD_INLINE svfloat32_t LoadU8AsF32(const svbool_t& mask, const uint8_t* src)
        {
            return svcvt_f32_u32_x(mask, svld1ub_u32(mask, src));
        }

        SIMD_INLINE void JpegDct1D(const svbool_t& mask,
            svfloat32_t& d0, svfloat32_t& d1, svfloat32_t& d2, svfloat32_t& d3,
            svfloat32_t& d4, svfloat32_t& d5, svfloat32_t& d6, svfloat32_t& d7)
        {
            const svfloat32_t c0707 = svdup_n_f32(0.707106781f);
            const svfloat32_t c0383 = svdup_n_f32(0.382683433f);
            const svfloat32_t c0541 = svdup_n_f32(0.541196100f);
            const svfloat32_t c1307 = svdup_n_f32(1.306562965f);

            svfloat32_t tmp0 = svadd_f32_x(mask, d0, d7);
            svfloat32_t tmp7 = svsub_f32_x(mask, d0, d7);
            svfloat32_t tmp1 = svadd_f32_x(mask, d1, d6);
            svfloat32_t tmp6 = svsub_f32_x(mask, d1, d6);
            svfloat32_t tmp2 = svadd_f32_x(mask, d2, d5);
            svfloat32_t tmp5 = svsub_f32_x(mask, d2, d5);
            svfloat32_t tmp3 = svadd_f32_x(mask, d3, d4);
            svfloat32_t tmp4 = svsub_f32_x(mask, d3, d4);

            svfloat32_t tmp10 = svadd_f32_x(mask, tmp0, tmp3);
            svfloat32_t tmp13 = svsub_f32_x(mask, tmp0, tmp3);
            svfloat32_t tmp11 = svadd_f32_x(mask, tmp1, tmp2);
            svfloat32_t tmp12 = svsub_f32_x(mask, tmp1, tmp2);

            d0 = svadd_f32_x(mask, tmp10, tmp11);
            d4 = svsub_f32_x(mask, tmp10, tmp11);

            svfloat32_t z1 = svmul_f32_x(mask, svadd_f32_x(mask, tmp12, tmp13), c0707);
            d2 = svadd_f32_x(mask, tmp13, z1);
            d6 = svsub_f32_x(mask, tmp13, z1);

            tmp10 = svadd_f32_x(mask, tmp4, tmp5);
            tmp11 = svadd_f32_x(mask, tmp5, tmp6);
            tmp12 = svadd_f32_x(mask, tmp6, tmp7);

            svfloat32_t z5 = svmul_f32_x(mask, svsub_f32_x(mask, tmp10, tmp12), c0383);
            svfloat32_t z2 = svmla_f32_x(mask, z5, tmp10, c0541);
            svfloat32_t z4 = svmla_f32_x(mask, z5, tmp12, c1307);
            svfloat32_t z3 = svmul_f32_x(mask, tmp11, c0707);

            svfloat32_t z11 = svadd_f32_x(mask, tmp7, z3);
            svfloat32_t z13 = svsub_f32_x(mask, tmp7, z3);

            d1 = svadd_f32_x(mask, z11, z4);
            d3 = svsub_f32_x(mask, z13, z2);
            d5 = svadd_f32_x(mask, z13, z2);
            d7 = svsub_f32_x(mask, z11, z4);
        }

        SIMD_INLINE void JpegDctV(const float* src, size_t srcStride, float* dst, size_t dstStride)
        {
            const svbool_t m4 = svwhilelt_b32((uint64_t)0, (uint64_t)4);
            for (int i = 0; i < 2; ++i, src += 4, dst += 4)
            {
                svfloat32_t d0 = svld1_f32(m4, src + 0 * srcStride);
                svfloat32_t d1 = svld1_f32(m4, src + 1 * srcStride);
                svfloat32_t d2 = svld1_f32(m4, src + 2 * srcStride);
                svfloat32_t d3 = svld1_f32(m4, src + 3 * srcStride);
                svfloat32_t d4 = svld1_f32(m4, src + 4 * srcStride);
                svfloat32_t d5 = svld1_f32(m4, src + 5 * srcStride);
                svfloat32_t d6 = svld1_f32(m4, src + 6 * srcStride);
                svfloat32_t d7 = svld1_f32(m4, src + 7 * srcStride);
                JpegDct1D(m4, d0, d1, d2, d3, d4, d5, d6, d7);
                svst1_f32(m4, dst + 0 * dstStride, d0);
                svst1_f32(m4, dst + 1 * dstStride, d1);
                svst1_f32(m4, dst + 2 * dstStride, d2);
                svst1_f32(m4, dst + 3 * dstStride, d3);
                svst1_f32(m4, dst + 4 * dstStride, d4);
                svst1_f32(m4, dst + 5 * dstStride, d5);
                svst1_f32(m4, dst + 6 * dstStride, d6);
                svst1_f32(m4, dst + 7 * dstStride, d7);
            }
        }

        SIMD_INLINE void JpegDctH(const float* src, size_t srcStride, const float* fdt, int* dst)
        {
            const svbool_t m4 = svwhilelt_b32((uint64_t)0, (uint64_t)4);
            for (int i = 0; i < 2; ++i, fdt += 4, dst += 4)
            {
                svfloat32_t r0 = svld1_f32(m4, src + 0 * srcStride);
                svfloat32_t r1 = svld1_f32(m4, src + 1 * srcStride);
                svfloat32_t r2 = svld1_f32(m4, src + 2 * srcStride);
                svfloat32_t r3 = svld1_f32(m4, src + 3 * srcStride);
                svfloat32_t t0l = svzip1_f32(r0, r2);
                svfloat32_t t0h = svzip2_f32(r0, r2);
                svfloat32_t t1l = svzip1_f32(r1, r3);
                svfloat32_t t1h = svzip2_f32(r1, r3);
                svfloat32_t d00 = svzip1_f32(t0l, t1l);
                svfloat32_t d01 = svzip2_f32(t0l, t1l);
                svfloat32_t d10 = svzip1_f32(t0h, t1h);
                svfloat32_t d11 = svzip2_f32(t0h, t1h);

                r0 = svld1_f32(m4, src + 0 * srcStride + 4);
                r1 = svld1_f32(m4, src + 1 * srcStride + 4);
                r2 = svld1_f32(m4, src + 2 * srcStride + 4);
                r3 = svld1_f32(m4, src + 3 * srcStride + 4);
                t0l = svzip1_f32(r0, r2);
                t0h = svzip2_f32(r0, r2);
                t1l = svzip1_f32(r1, r3);
                t1h = svzip2_f32(r1, r3);
                svfloat32_t d20 = svzip1_f32(t0l, t1l);
                svfloat32_t d21 = svzip2_f32(t0l, t1l);
                svfloat32_t d30 = svzip1_f32(t0h, t1h);
                svfloat32_t d31 = svzip2_f32(t0h, t1h);
                src += 4 * srcStride;

                svfloat32_t t00 = svadd_f32_x(m4, d00, d31);
                svfloat32_t t01 = svadd_f32_x(m4, d01, d30);
                svfloat32_t t10 = svadd_f32_x(m4, d10, d21);
                svfloat32_t t11 = svadd_f32_x(m4, d11, d20);
                svfloat32_t tmp7 = svsub_f32_x(m4, d00, d31);
                svfloat32_t tmp6 = svsub_f32_x(m4, d01, d30);
                svfloat32_t tmp5 = svsub_f32_x(m4, d10, d21);
                svfloat32_t tmp4 = svsub_f32_x(m4, d11, d20);

                svfloat32_t tmp10 = svadd_f32_x(m4, t00, t11);
                svfloat32_t tmp13 = svsub_f32_x(m4, t00, t11);
                svfloat32_t tmp11 = svadd_f32_x(m4, t01, t10);
                svfloat32_t tmp12 = svsub_f32_x(m4, t01, t10);

                d00 = svadd_f32_x(m4, tmp10, tmp11);
                d20 = svsub_f32_x(m4, tmp10, tmp11);

                const svfloat32_t c0707 = svdup_n_f32(0.707106781f);
                const svfloat32_t c0383 = svdup_n_f32(0.382683433f);
                const svfloat32_t c0541 = svdup_n_f32(0.541196100f);
                const svfloat32_t c1307 = svdup_n_f32(1.306562965f);

                svfloat32_t z1 = svmul_f32_x(m4, svadd_f32_x(m4, tmp12, tmp13), c0707);
                d10 = svadd_f32_x(m4, tmp13, z1);
                d30 = svsub_f32_x(m4, tmp13, z1);

                tmp10 = svadd_f32_x(m4, tmp4, tmp5);
                tmp11 = svadd_f32_x(m4, tmp5, tmp6);
                tmp12 = svadd_f32_x(m4, tmp6, tmp7);

                svfloat32_t z5 = svmul_f32_x(m4, svsub_f32_x(m4, tmp10, tmp12), c0383);
                svfloat32_t z2 = svmla_f32_x(m4, z5, tmp10, c0541);
                svfloat32_t z4 = svmla_f32_x(m4, z5, tmp12, c1307);
                svfloat32_t z3 = svmul_f32_x(m4, tmp11, c0707);

                svfloat32_t z11 = svadd_f32_x(m4, tmp7, z3);
                svfloat32_t z13 = svsub_f32_x(m4, tmp7, z3);

                d01 = svadd_f32_x(m4, z11, z4);
                d11 = svsub_f32_x(m4, z13, z2);
                d21 = svadd_f32_x(m4, z13, z2);
                d31 = svsub_f32_x(m4, z11, z4);

                svst1_s32(m4, dst + 0x00, Round(m4, svmul_f32_x(m4, svld1_f32(m4, fdt + 8 * 0), d00)));
                svst1_s32(m4, dst + 0x08, Round(m4, svmul_f32_x(m4, svld1_f32(m4, fdt + 8 * 1), d01)));
                svst1_s32(m4, dst + 0x10, Round(m4, svmul_f32_x(m4, svld1_f32(m4, fdt + 8 * 2), d10)));
                svst1_s32(m4, dst + 0x18, Round(m4, svmul_f32_x(m4, svld1_f32(m4, fdt + 8 * 3), d11)));
                svst1_s32(m4, dst + 0x20, Round(m4, svmul_f32_x(m4, svld1_f32(m4, fdt + 8 * 4), d20)));
                svst1_s32(m4, dst + 0x28, Round(m4, svmul_f32_x(m4, svld1_f32(m4, fdt + 8 * 5), d21)));
                svst1_s32(m4, dst + 0x30, Round(m4, svmul_f32_x(m4, svld1_f32(m4, fdt + 8 * 6), d30)));
                svst1_s32(m4, dst + 0x38, Round(m4, svmul_f32_x(m4, svld1_f32(m4, fdt + 8 * 7), d31)));
            }
        }

        SIMD_INLINE void JpegDct8(const float* src, size_t stride, const float* fdt, int* dst)
        {
            const svbool_t m8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
            svfloat32_t d0 = svld1_f32(m8, src + 0 * stride);
            svfloat32_t d1 = svld1_f32(m8, src + 1 * stride);
            svfloat32_t d2 = svld1_f32(m8, src + 2 * stride);
            svfloat32_t d3 = svld1_f32(m8, src + 3 * stride);
            svfloat32_t d4 = svld1_f32(m8, src + 4 * stride);
            svfloat32_t d5 = svld1_f32(m8, src + 5 * stride);
            svfloat32_t d6 = svld1_f32(m8, src + 6 * stride);
            svfloat32_t d7 = svld1_f32(m8, src + 7 * stride);
            JpegDct1D(m8, d0, d1, d2, d3, d4, d5, d6, d7);

            SIMD_ALIGNED(64) float buf[64];
            svst1_f32(m8, buf + 0 * 8, d0);
            svst1_f32(m8, buf + 1 * 8, d1);
            svst1_f32(m8, buf + 2 * 8, d2);
            svst1_f32(m8, buf + 3 * 8, d3);
            svst1_f32(m8, buf + 4 * 8, d4);
            svst1_f32(m8, buf + 5 * 8, d5);
            svst1_f32(m8, buf + 6 * 8, d6);
            svst1_f32(m8, buf + 7 * 8, d7);

            svuint32_t idx = svlsl_n_u32_x(m8, svindex_u32(0, 1), 3);
            d0 = svld1_gather_u32index_f32(m8, buf + 0, idx);
            d1 = svld1_gather_u32index_f32(m8, buf + 1, idx);
            d2 = svld1_gather_u32index_f32(m8, buf + 2, idx);
            d3 = svld1_gather_u32index_f32(m8, buf + 3, idx);
            d4 = svld1_gather_u32index_f32(m8, buf + 4, idx);
            d5 = svld1_gather_u32index_f32(m8, buf + 5, idx);
            d6 = svld1_gather_u32index_f32(m8, buf + 6, idx);
            d7 = svld1_gather_u32index_f32(m8, buf + 7, idx);
            JpegDct1D(m8, d0, d1, d2, d3, d4, d5, d6, d7);

            svst1_s32(m8, dst + 8 * 0, Round(m8, svmul_f32_x(m8, svld1_f32(m8, fdt + 8 * 0), d0)));
            svst1_s32(m8, dst + 8 * 1, Round(m8, svmul_f32_x(m8, svld1_f32(m8, fdt + 8 * 1), d1)));
            svst1_s32(m8, dst + 8 * 2, Round(m8, svmul_f32_x(m8, svld1_f32(m8, fdt + 8 * 2), d2)));
            svst1_s32(m8, dst + 8 * 3, Round(m8, svmul_f32_x(m8, svld1_f32(m8, fdt + 8 * 3), d3)));
            svst1_s32(m8, dst + 8 * 4, Round(m8, svmul_f32_x(m8, svld1_f32(m8, fdt + 8 * 4), d4)));
            svst1_s32(m8, dst + 8 * 5, Round(m8, svmul_f32_x(m8, svld1_f32(m8, fdt + 8 * 5), d5)));
            svst1_s32(m8, dst + 8 * 6, Round(m8, svmul_f32_x(m8, svld1_f32(m8, fdt + 8 * 6), d6)));
            svst1_s32(m8, dst + 8 * 7, Round(m8, svmul_f32_x(m8, svld1_f32(m8, fdt + 8 * 7), d7)));
        }

        static int JpegProcessDu(Base::BitBuf& bitBuf, float* CDU, int stride, const float* fdtbl, int DC, const uint16_t HTDC[256][2], const uint16_t HTAC[256][2])
        {
            SIMD_ALIGNED(64) int DUO[64], DU[64];
            if (svcntw() >= 8)
                JpegDct8(CDU, stride, fdtbl, DUO);
            else
            {
                SIMD_ALIGNED(64) float BUF[64];
                JpegDctV(CDU, stride, BUF, 8);
                JpegDctH(BUF, 8, fdtbl, DUO);
            }
            const size_t F = svcntw();
            for (size_t i = 0; i < 64; i += F)
            {
                svbool_t mask = svwhilelt_b32(i, (size_t)64);
                svuint32_t idx = svld1ub_u32(mask, Base::JpegZigZagT + i);
                svst1_scatter_u32index_s32(mask, DU, idx, svld1_s32(mask, DUO + i));
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
            int end0pos = 63;
            for (; (end0pos > 0) && (DU[end0pos] == 0); --end0pos);
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
                    for (int nrmarker = 1; nrmarker <= lng; ++nrmarker)
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
            return DU[0];
        }

        SIMD_INLINE void RgbToYuv(const uint8_t* r, const uint8_t* g, const uint8_t* b, int stride, int height,
            float* y, float* u, float* v, int size)
        {
            const svfloat32_t k0 = svdup_n_f32(+0.29900f);
            const svfloat32_t k1 = svdup_n_f32(+0.58700f);
            const svfloat32_t k2 = svdup_n_f32(+0.11400f);
            const svfloat32_t k3 = svdup_n_f32(-128.000f);
            const svfloat32_t k4 = svdup_n_f32(-0.16874f);
            const svfloat32_t k5 = svdup_n_f32(-0.33126f);
            const svfloat32_t k6 = svdup_n_f32(+0.50000f);
            const svfloat32_t k7 = svdup_n_f32(+0.50000f);
            const svfloat32_t k8 = svdup_n_f32(-0.41869f);
            const svfloat32_t k9 = svdup_n_f32(-0.08131f);
            const size_t F = svcntw();
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += (int)F)
                {
                    svbool_t mask = svwhilelt_b32((uint64_t)col, (uint64_t)size);
                    svfloat32_t _r = LoadU8AsF32(mask, r + col);
                    svfloat32_t _g = LoadU8AsF32(mask, g + col);
                    svfloat32_t _b = LoadU8AsF32(mask, b + col);
                    svfloat32_t _y = svmla_f32_x(mask, k3, _r, k0);
                    _y = svmla_f32_x(mask, _y, _g, k1);
                    _y = svmla_f32_x(mask, _y, _b, k2);
                    svst1_f32(mask, y + col, _y);
                    svfloat32_t _u = svmul_f32_x(mask, _r, k4);
                    _u = svmla_f32_x(mask, _u, _g, k5);
                    _u = svmla_f32_x(mask, _u, _b, k6);
                    svst1_f32(mask, u + col, _u);
                    svfloat32_t _v = svmul_f32_x(mask, _r, k7);
                    _v = svmla_f32_x(mask, _v, _g, k8);
                    _v = svmla_f32_x(mask, _v, _b, k9);
                    svst1_f32(mask, v + col, _v);
                }
                if (++row < height)
                    r += stride, g += stride, b += stride;
                y += size, u += size, v += size;
            }
        }

        SIMD_INLINE void GrayToY(const uint8_t* g, int stride, int height, float* y, int size)
        {
            const svfloat32_t k = svdup_n_f32(-128.000f);
            const size_t F = svcntw();
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += (int)F)
                {
                    svbool_t mask = svwhilelt_b32((uint64_t)col, (uint64_t)size);
                    svst1_f32(mask, y + col, svadd_f32_x(mask, LoadU8AsF32(mask, g + col), k));
                }
                if (++row < height)
                    g += stride;
                y += size;
            }
        }

        SIMD_INLINE void SubUv(const float* src, float* dst)
        {
            const size_t F = svcntw();
            const svfloat32_t q = svdup_n_f32(0.25f);
            const svbool_t m4 = svwhilelt_b32((uint64_t)0, (uint64_t)4);
            const svbool_t m8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
            const svuint32_t even = svlsl_n_u32_x(m4, svindex_u32(0, 1), 1);
            const svuint32_t odd = svorr_n_u32_x(m4, even, 1);
            for (int yy = 0; yy < 8; ++yy)
            {
                if (F >= 8)
                {
                    svfloat32_t s0 = svadd_f32_x(m8, svld1_f32(m8, src + 0), svld1_f32(m8, src + 16));
                    svfloat32_t s1 = svadd_f32_x(m8, svld1_f32(m8, src + 8), svld1_f32(m8, src + 24));
                    svst1_f32(m4, dst + 0, svmul_f32_x(m4, svadd_f32_x(m4, svtbl_f32(s0, even), svtbl_f32(s0, odd)), q));
                    svst1_f32(m4, dst + 4, svmul_f32_x(m4, svadd_f32_x(m4, svtbl_f32(s1, even), svtbl_f32(s1, odd)), q));
                }
                else
                {
                    svfloat32_t s0 = svadd_f32_x(m4, svld1_f32(m4, src + 0), svld1_f32(m4, src + 16));
                    svfloat32_t s1 = svadd_f32_x(m4, svld1_f32(m4, src + 4), svld1_f32(m4, src + 20));
                    svst1_f32(m4, dst + 0, svmul_f32_x(m4, svaddp_f32_x(m4, s0, s1), q));
                    s0 = svadd_f32_x(m4, svld1_f32(m4, src + 8), svld1_f32(m4, src + 24));
                    s1 = svadd_f32_x(m4, svld1_f32(m4, src + 12), svld1_f32(m4, src + 28));
                    svst1_f32(m4, dst + 4, svmul_f32_x(m4, svaddp_f32_x(m4, s0, s1), q));
                }
                src += 32;
                dst += 8;
            }
        }

        SIMD_INLINE void Nv12ToUv(const uint8_t* uvSrc, int uvStride, int height, float* u, float* v)
        {
            const svbool_t m8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
            const svuint32_t off = svindex_u32(0, 2);
            const svfloat32_t k = svdup_n_f32(-128.000f);
            for (int row = 0; row < 8;)
            {
                svfloat32_t _u = svadd_f32_x(m8, svcvt_f32_u32_x(m8, svld1ub_gather_u32offset_u32(m8, uvSrc, off)), k);
                svfloat32_t _v = svadd_f32_x(m8, svcvt_f32_u32_x(m8, svld1ub_gather_u32offset_u32(m8, uvSrc + 1, off)), k);
                svst1_f32(m8, u, _u);
                svst1_f32(m8, v, _v);
                if (++row < height)
                    uvSrc += uvStride;
                u += 8, v += 8;
            }
        }

        void JpegWriteBlockSubs(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            bool gray = red == green && red == blue;
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
                        GrayToY(red + x, stride, height - y, Y, 16);
                    else
                        RgbToYuv(red + x, green + x, blue + x, stride, height - y, Y, U, V, 16);
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
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        void JpegWriteBlockFull(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            bool gray = red == green && red == blue;
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width8 = width & (~7);
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 8)
            {
                int x = 0;
                SIMD_ALIGNED(64) float Y[64], U[64], V[64];
                for (; x < width8; x += 8)
                {
                    if (gray)
                        GrayToY(red + x, stride, height - y, Y, 8);
                    else
                        RgbToYuv(red + x, green + x, blue + x, stride, height - y, Y, U, V, 8);
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
                    DCY = JpegProcessDu(bitBuf, Y, 8, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
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
                    GrayToY(ySrc + x, yStride, height - y, Y, 16);
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        Nv12ToUv(uvSrc + x, uvStride, Base::UvSize(height - y), U, V);
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
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
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        Base::Nv12ToUv(uvSrc + x, uvStride, Base::UvSize(height - y), Base::UvSize(width - x), U, V);
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
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
                    GrayToY(ySrc + x, yStride, height - y, Y, 16);
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        GrayToY(uSrc + Base::UvSize(x), uStride, Base::UvSize(height - y), U, 8);
                        GrayToY(vSrc + Base::UvSize(x), vStride, Base::UvSize(height - y), V, 8);
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
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
                    DCY = JpegProcessDu(bitBuf, Y + 0, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 8, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 128, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    DCY = JpegProcessDu(bitBuf, Y + 136, 16, fY, DCY, Base::HuffmanYdc, Base::HuffmanYac);
                    if (gray)
                        Base::JpegProcessDuGrayUv(bitBuf);
                    else
                    {
                        Base::GrayToY(uSrc + Base::UvSize(x), uStride, Base::UvSize(height - y), Base::UvSize(width - x), U, 8);
                        Base::GrayToY(vSrc + Base::UvSize(x), vStride, Base::UvSize(height - y), Base::UvSize(width - x), V, 8);
                        DCU = JpegProcessDu(bitBuf, U, 8, fUv, DCU, Base::HuffmanUVdc, Base::HuffmanUVac);
                        DCV = JpegProcessDu(bitBuf, V, 8, fUv, DCV, Base::HuffmanUVdc, Base::HuffmanUVac);
                    }
                }
            }
            Base::WriteBits(stream, bitBuf.data, bitBuf.size);
            bitBuf.Clear();
        }

        //---------------------------------------------------------------------

        ImageJpegSaver::ImageJpegSaver(const ImageSaverParam& param)
            : Neon::ImageJpegSaver(param)
        {
        }

        void ImageJpegSaver::Init()
        {
            InitParams(true);
            if (_param.yuvType == SimdYuvUnknown)
            {
                switch (_param.format)
                {
                case SimdPixelFormatBgr24:
                case SimdPixelFormatRgb24:
                    _deintBgr = _param.width < svcntb() ? Base::DeinterleaveBgr : Sve2::DeinterleaveBgr;
                    break;
                case SimdPixelFormatBgra32:
                case SimdPixelFormatRgba32:
                    _deintBgra = _param.width < svcntb() ? Base::DeinterleaveBgra : Sve2::DeinterleaveBgra;
                    break;
                default:
                    break;
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

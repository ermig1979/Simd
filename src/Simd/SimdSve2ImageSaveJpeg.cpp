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
            svfloat32_t offset = svsel_f32(pos, svdup_n_f32(0.5f), svdup_n_f32(-0.5f));
            return svcvt_s32_f32_x(mask, svadd_f32_x(mask, value, offset));
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

        SIMD_INLINE void Transpose4x4(
            svfloat32_t r0, svfloat32_t r1, svfloat32_t r2, svfloat32_t r3,
            svfloat32_t& c0, svfloat32_t& c1, svfloat32_t& c2, svfloat32_t& c3)
        {
            svfloat32_t t0 = svzip1_f32(r0, r2);
            svfloat32_t t1 = svzip2_f32(r0, r2);
            svfloat32_t t2 = svzip1_f32(r1, r3);
            svfloat32_t t3 = svzip2_f32(r1, r3);
            c0 = svzip1_f32(t0, t2);
            c1 = svzip2_f32(t0, t2);
            c2 = svzip1_f32(t1, t3);
            c3 = svzip2_f32(t1, t3);
        }

        SIMD_INLINE svfloat32_t Zip128Lo(const svfloat32_t& a, const svfloat32_t& b)
        {
            const svbool_t lo = svwhilelt_b32((uint64_t)0, (uint64_t)4);
            return svsel_f32(lo, a, svext_f32(b, b, 4));
        }

        SIMD_INLINE svfloat32_t Zip128Hi(const svfloat32_t& a, const svfloat32_t& b)
        {
            const svbool_t lo = svwhilelt_b32((uint64_t)0, (uint64_t)4);
            return svsel_f32(lo, svext_f32(a, a, 4), b);
        }

        SIMD_INLINE void Transpose8x8(
            svfloat32_t& r0, svfloat32_t& r1, svfloat32_t& r2, svfloat32_t& r3,
            svfloat32_t& r4, svfloat32_t& r5, svfloat32_t& r6, svfloat32_t& r7)
        {
            svfloat32_t a0 = svzip1_f32(r0, r1);
            svfloat32_t a1 = svzip2_f32(r0, r1);
            svfloat32_t a2 = svzip1_f32(r2, r3);
            svfloat32_t a3 = svzip2_f32(r2, r3);
            svfloat32_t a4 = svzip1_f32(r4, r5);
            svfloat32_t a5 = svzip2_f32(r4, r5);
            svfloat32_t a6 = svzip1_f32(r6, r7);
            svfloat32_t a7 = svzip2_f32(r6, r7);

            svfloat64_t b0 = svzip1_f64(svreinterpret_f64_f32(a0), svreinterpret_f64_f32(a2));
            svfloat64_t b1 = svzip2_f64(svreinterpret_f64_f32(a0), svreinterpret_f64_f32(a2));
            svfloat64_t b2 = svzip1_f64(svreinterpret_f64_f32(a1), svreinterpret_f64_f32(a3));
            svfloat64_t b3 = svzip2_f64(svreinterpret_f64_f32(a1), svreinterpret_f64_f32(a3));
            svfloat64_t b4 = svzip1_f64(svreinterpret_f64_f32(a4), svreinterpret_f64_f32(a6));
            svfloat64_t b5 = svzip2_f64(svreinterpret_f64_f32(a4), svreinterpret_f64_f32(a6));
            svfloat64_t b6 = svzip1_f64(svreinterpret_f64_f32(a5), svreinterpret_f64_f32(a7));
            svfloat64_t b7 = svzip2_f64(svreinterpret_f64_f32(a5), svreinterpret_f64_f32(a7));

            svfloat32_t c0 = svreinterpret_f32_f64(b0);
            svfloat32_t c1 = svreinterpret_f32_f64(b1);
            svfloat32_t c2 = svreinterpret_f32_f64(b2);
            svfloat32_t c3 = svreinterpret_f32_f64(b3);
            svfloat32_t c4 = svreinterpret_f32_f64(b4);
            svfloat32_t c5 = svreinterpret_f32_f64(b5);
            svfloat32_t c6 = svreinterpret_f32_f64(b6);
            svfloat32_t c7 = svreinterpret_f32_f64(b7);

            r0 = Zip128Lo(c0, c4);
            r1 = Zip128Hi(c0, c4);
            r2 = Zip128Lo(c1, c5);
            r3 = Zip128Hi(c1, c5);
            r4 = Zip128Lo(c2, c6);
            r5 = Zip128Hi(c2, c6);
            r6 = Zip128Lo(c3, c7);
            r7 = Zip128Hi(c3, c7);
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
                svfloat32_t d00, d01, d10, d11;
                Transpose4x4(r0, r1, r2, r3, d00, d01, d10, d11);

                r0 = svld1_f32(m4, src + 0 * srcStride + 4);
                r1 = svld1_f32(m4, src + 1 * srcStride + 4);
                r2 = svld1_f32(m4, src + 2 * srcStride + 4);
                r3 = svld1_f32(m4, src + 3 * srcStride + 4);
                svfloat32_t d20, d21, d30, d31;
                Transpose4x4(r0, r1, r2, r3, d20, d21, d30, d31);
                src += 4 * srcStride;

                JpegDct1D(m4, d00, d01, d10, d11, d20, d21, d30, d31);

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
            Transpose8x8(d0, d1, d2, d3, d4, d5, d6, d7);
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
            if (svcntw() == 8)
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
            const svfloat32_t kY0 = svdup_n_f32(+0.29900f);
            const svfloat32_t kY1 = svdup_n_f32(+0.58700f);
            const svfloat32_t kY2 = svdup_n_f32(+0.11400f);
            const svfloat32_t kY3 = svdup_n_f32(-128.000f);
            const svfloat32_t kU0 = svdup_n_f32(+0.16874f);
            const svfloat32_t kU1 = svdup_n_f32(+0.33126f);
            const svfloat32_t kU2 = svdup_n_f32(+0.50000f);
            const svfloat32_t kV0 = svdup_n_f32(+0.50000f);
            const svfloat32_t kV1 = svdup_n_f32(+0.41869f);
            const svfloat32_t kV2 = svdup_n_f32(+0.08131f);
            const size_t F = svcntw();
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += (int)F)
                {
                    svbool_t mask = svwhilelt_b32((uint64_t)col, (uint64_t)size);
                    svfloat32_t _r = LoadU8AsF32(mask, r + col);
                    svfloat32_t _g = LoadU8AsF32(mask, g + col);
                    svfloat32_t _b = LoadU8AsF32(mask, b + col);
                    svfloat32_t _y = svmla_f32_x(mask, kY3, _r, kY0);
                    _y = svmla_f32_x(mask, _y, _g, kY1);
                    _y = svmla_f32_x(mask, _y, _b, kY2);
                    svfloat32_t _u = svmul_f32_x(mask, _b, kU2);
                    _u = svmls_f32_x(mask, _u, _r, kU0);
                    _u = svmls_f32_x(mask, _u, _g, kU1);
                    svfloat32_t _v = svmul_f32_x(mask, _r, kV0);
                    _v = svmls_f32_x(mask, _v, _g, kV1);
                    _v = svmls_f32_x(mask, _v, _b, kV2);
                    svst1_f32(mask, y + col, _y);
                    svst1_f32(mask, u + col, _u);
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

        SIMD_INLINE svfloat32_t Hadd32f(const svfloat32_t& a, const svfloat32_t& b)
        {
            return svadd_f32_x(svptrue_b32(), svuzp1_f32(a, b), svuzp2_f32(a, b));
        }

        SIMD_INLINE void SubUv(const float* src, float* dst)
        {
            const size_t F = svcntw();
            const svfloat32_t q = svdup_n_f32(0.25f);
            const svbool_t m4 = svwhilelt_b32((uint64_t)0, (uint64_t)4);
            const svbool_t m8 = svwhilelt_b32((uint64_t)0, (uint64_t)8);
            const svuint32_t even = svindex_u32(0, 2);
            const svuint32_t odd = svindex_u32(1, 2);
            for (int yy = 0; yy < 8; ++yy)
            {
                if (F >= 8)
                {
                    svfloat32_t s0 = svadd_f32_x(m8, svld1_f32(m8, src + 0), svld1_f32(m8, src + 16));
                    svfloat32_t s1 = svadd_f32_x(m8, svld1_f32(m8, src + 8), svld1_f32(m8, src + 24));
                    if (F == 8)
                        svst1_f32(m8, dst, svmul_f32_x(m8, Hadd32f(s0, s1), q));
                    else
                    {
                        svst1_f32(m4, dst + 0, svmul_f32_x(m4, svadd_f32_x(m4, svtbl_f32(s0, even), svtbl_f32(s0, odd)), q));
                        svst1_f32(m4, dst + 4, svmul_f32_x(m4, svadd_f32_x(m4, svtbl_f32(s1, even), svtbl_f32(s1, odd)), q));
                    }
                }
                else
                {
                    svfloat32_t s0 = svadd_f32_x(m4, svld1_f32(m4, src + 0), svld1_f32(m4, src + 16));
                    svfloat32_t s1 = svadd_f32_x(m4, svld1_f32(m4, src + 4), svld1_f32(m4, src + 20));
                    svst1_f32(m4, dst + 0, svmul_f32_x(m4, Hadd32f(s0, s1), q));
                    s0 = svadd_f32_x(m4, svld1_f32(m4, src + 8), svld1_f32(m4, src + 24));
                    s1 = svadd_f32_x(m4, svld1_f32(m4, src + 12), svld1_f32(m4, src + 28));
                    svst1_f32(m4, dst + 4, svmul_f32_x(m4, Hadd32f(s0, s1), q));
                }
                src += 32;
                dst += 8;
            }
        }

        SIMD_INLINE void Nv12ToUv(const uint8_t* uvSrc, int uvStride, int height, float* u, float* v)
        {
            const svfloat32_t k = svdup_n_f32(-128.000f);
            const svbool_t m4 = svwhilelt_b32((uint64_t)0, (uint64_t)4);
            const svbool_t m8b = svwhilelt_b8((uint64_t)0, (uint64_t)8);
            for (int row = 0; row < 8;)
            {
                for (int col = 0; col < 8; col += 4)
                {
                    svuint8_t uv = svld1_u8(m8b, uvSrc + col * 2);
                    svuint8_t u8 = svuzp1_u8(uv, uv);
                    svuint8_t v8 = svuzp2_u8(uv, uv);
                    svuint32_t u32 = svunpklo_u32(svunpklo_u16(u8));
                    svuint32_t v32 = svunpklo_u32(svunpklo_u16(v8));
                    svst1_f32(m4, u + col, svadd_f32_x(m4, svcvt_f32_u32_x(m4, u32), k));
                    svst1_f32(m4, v + col, svadd_f32_x(m4, svcvt_f32_u32_x(m4, v32), k));
                }
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

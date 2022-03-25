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
#include "Simd/SimdStore.h"
#include "Simd/SimdNeon.h"
#include "Simd/SimdBase.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
#define SIMD_NEON_JPEG_DVT_VER 1

        SIMD_INLINE void JpegDctV(const float* src, size_t srcStride, float *dst, size_t dstStride)
        {
            for (int i = 0; i < 2; i++, src += 4)
            {
                float32x4_t d0 = Load<false>(src + 0 * srcStride);
                float32x4_t d1 = Load<false>(src + 1 * srcStride);
                float32x4_t d2 = Load<false>(src + 2 * srcStride);
                float32x4_t d3 = Load<false>(src + 3 * srcStride);
                float32x4_t d4 = Load<false>(src + 4 * srcStride);
                float32x4_t d5 = Load<false>(src + 5 * srcStride);
                float32x4_t d6 = Load<false>(src + 6 * srcStride);
                float32x4_t d7 = Load<false>(src + 7 * srcStride);

                float32x4_t tmp0 = vaddq_f32(d0, d7);
                float32x4_t tmp7 = vsubq_f32(d0, d7);
                float32x4_t tmp1 = vaddq_f32(d1, d6);
                float32x4_t tmp6 = vsubq_f32(d1, d6);
                float32x4_t tmp2 = vaddq_f32(d2, d5);
                float32x4_t tmp5 = vsubq_f32(d2, d5);
                float32x4_t tmp3 = vaddq_f32(d3, d4);
                float32x4_t tmp4 = vsubq_f32(d3, d4);

                float32x4_t tmp10 = vaddq_f32(tmp0, tmp3);
                float32x4_t tmp13 = vsubq_f32(tmp0, tmp3);
                float32x4_t tmp11 = vaddq_f32(tmp1, tmp2);
                float32x4_t tmp12 = vsubq_f32(tmp1, tmp2);
#if SIMD_NEON_JPEG_DVT_VER == 1
                d0 = vaddq_f32(tmp10, tmp11);
                d4 = vsubq_f32(tmp10, tmp11);

                float32x4_t z1 = vmulq_f32(vaddq_f32(tmp12, tmp13), vdupq_n_f32(0.707106781f));
                d2 = vaddq_f32(tmp13, z1);
                d6 = vsubq_f32(tmp13, z1);

                tmp10 = vaddq_f32(tmp4, tmp5);
                tmp11 = vaddq_f32(tmp5, tmp6);
                tmp12 = vaddq_f32(tmp6, tmp7);

                float32x4_t z5 = vmulq_f32(vsubq_f32(tmp10, tmp12),  vdupq_n_f32(0.382683433f));
                float32x4_t z2 = vaddq_f32(vmulq_f32(tmp10, vdupq_n_f32(0.541196100f)), z5);
                float32x4_t z4 = vaddq_f32(vmulq_f32(tmp12, vdupq_n_f32(1.306562965f)), z5);
                float32x4_t z3 = vmulq_f32(tmp11, vdupq_n_f32(0.707106781f));

                float32x4_t z11 = vaddq_f32(tmp7, z3);
                float32x4_t z13 = vsubq_f32(tmp7, z3);

                Store<false>(dst + 0 * dstStride, d0);
                Store<false>(dst + 1 * dstStride, vaddq_f32(z11, z4));
                Store<false>(dst + 2 * dstStride, d2);
                Store<false>(dst + 3 * dstStride, vsubq_f32(z13, z2));
                Store<false>(dst + 4 * dstStride, d4);
                Store<false>(dst + 5 * dstStride, vaddq_f32(z13, z2));
                Store<false>(dst + 6 * dstStride, d6);
                Store<false>(dst + 7 * dstStride, vsubq_f32(z11, z4));
                dst += 4;
#else
                float32x4x4_t dst0, dst1;
                dst0.val[0] = vaddq_f32(tmp10, tmp11);
                dst1.val[0] = vsubq_f32(tmp10, tmp11);

                float32x4_t z1 = vmulq_f32(vaddq_f32(tmp12, tmp13), vdupq_n_f32(0.707106781f));
                dst0.val[2] = vaddq_f32(tmp13, z1);
                dst1.val[2] = vsubq_f32(tmp13, z1);

                tmp10 = vaddq_f32(tmp4, tmp5);
                tmp11 = vaddq_f32(tmp5, tmp6);
                tmp12 = vaddq_f32(tmp6, tmp7);

                float32x4_t z5 = vmulq_f32(vsubq_f32(tmp10, tmp12), vdupq_n_f32(0.382683433f));
                float32x4_t z2 = vaddq_f32(vmulq_f32(tmp10, vdupq_n_f32(0.541196100f)), z5);
                float32x4_t z4 = vaddq_f32(vmulq_f32(tmp12, vdupq_n_f32(1.306562965f)), z5);
                float32x4_t z3 = vmulq_f32(tmp11, vdupq_n_f32(0.707106781f));

                float32x4_t z11 = vaddq_f32(tmp7, z3);
                float32x4_t z13 = vsubq_f32(tmp7, z3);

                dst0.val[1] = vaddq_f32(z11, z4);
                dst0.val[3] = vsubq_f32(z13, z2);

                dst1.val[1] = vaddq_f32(z13, z2);
                dst1.val[3] = vsubq_f32(z11, z4);

                Store4<false>(dst + 0 * F, dst0);
                Store4<false>(dst + 8 * F, dst1);
                dst += 4 * F;
#endif
            }
        }

        SIMD_INLINE void JpegDctH(const float* src, size_t srcStride, const float * fdt, int* dst)
        {
            for (int i = 0; i < 2; i++, fdt += 4, dst += 4)
            {
                float32x4x2_t d0, d1, d2, d3, t0, t1;
#if SIMD_NEON_JPEG_DVT_VER == 1
                t0 = vzipq_f32(Load<false>(src + 0 * srcStride), Load<false>(src + 2 * srcStride));
                t1 = vzipq_f32(Load<false>(src + 1 * srcStride), Load<false>(src + 3 * srcStride));
                d0 = vzipq_f32(t0.val[0], t1.val[0]);
                d1 = vzipq_f32(t0.val[1], t1.val[1]);

                t0 = vzipq_f32(Load<false>(src + 0 * srcStride + 4), Load<false>(src + 2 * srcStride + 4));
                t1 = vzipq_f32(Load<false>(src + 1 * srcStride + 4), Load<false>(src + 3 * srcStride + 4));
                d2 = vzipq_f32(t0.val[0], t1.val[0]);
                d3 = vzipq_f32(t0.val[1], t1.val[1]);
                src += 4 * srcStride;
#else
                d0.val[0] = Load<false>(src + 0 * F);
                d0.val[1] = Load<false>(src + 1 * F);
                d1.val[0] = Load<false>(src + 2 * F);
                d1.val[1] = Load<false>(src + 3 * F);

                d2.val[0] = Load<false>(src + 4 * F);
                d2.val[1] = Load<false>(src + 5 * F);
                d3.val[0] = Load<false>(src + 6 * F);
                d3.val[1] = Load<false>(src + 7 * F);
                src += 8 * F;
#endif

                t0.val[0] = vaddq_f32(d0.val[0], d3.val[1]);
                t0.val[1] = vaddq_f32(d0.val[1], d3.val[0]);
                t1.val[0] = vaddq_f32(d1.val[0], d2.val[1]);
                t1.val[1] = vaddq_f32(d1.val[1], d2.val[0]);
                float32x4_t tmp7 = vsubq_f32(d0.val[0], d3.val[1]);
                float32x4_t tmp6 = vsubq_f32(d0.val[1], d3.val[0]);
                float32x4_t tmp5 = vsubq_f32(d1.val[0], d2.val[1]);
                float32x4_t tmp4 = vsubq_f32(d1.val[1], d2.val[0]);

                float32x4_t tmp10 = vaddq_f32(t0.val[0], t1.val[1]);
                float32x4_t tmp13 = vsubq_f32(t0.val[0], t1.val[1]);
                float32x4_t tmp11 = vaddq_f32(t0.val[1], t1.val[0]);
                float32x4_t tmp12 = vsubq_f32(t0.val[1], t1.val[0]);

                d0.val[0] = vaddq_f32(tmp10, tmp11);
                d2.val[0] = vsubq_f32(tmp10, tmp11);

                float32x4_t z1 = vmulq_f32(vaddq_f32(tmp12, tmp13), vdupq_n_f32(0.707106781f));
                d1.val[0] = vaddq_f32(tmp13, z1);
                d3.val[0] = vsubq_f32(tmp13, z1);

                tmp10 = vaddq_f32(tmp4, tmp5);
                tmp11 = vaddq_f32(tmp5, tmp6);
                tmp12 = vaddq_f32(tmp6, tmp7);

                float32x4_t z5 = vmulq_f32(vsubq_f32(tmp10, tmp12), vdupq_n_f32(0.382683433f));
                float32x4_t z2 = vaddq_f32(vmulq_f32(tmp10, vdupq_n_f32(0.541196100f)), z5);
                float32x4_t z4 = vaddq_f32(vmulq_f32(tmp12, vdupq_n_f32(1.306562965f)), z5);
                float32x4_t z3 = vmulq_f32(tmp11, vdupq_n_f32(0.707106781f));

                float32x4_t z11 = vaddq_f32(tmp7, z3);
                float32x4_t z13 = vsubq_f32(tmp7, z3);

                d0.val[1] = vaddq_f32(z11, z4);
                d1.val[1] = vsubq_f32(z13, z2);
                d2.val[1] = vaddq_f32(z13, z2);
                d3.val[1] = vsubq_f32(z11, z4);

                Store<false>(dst + 0x00, Round(vmulq_f32(Load<false>(fdt + DF * 0), d0.val[0])));
                Store<false>(dst + 0x08, Round(vmulq_f32(Load<false>(fdt + DF * 1), d0.val[1])));
                Store<false>(dst + 0x10, Round(vmulq_f32(Load<false>(fdt + DF * 2), d1.val[0])));
                Store<false>(dst + 0x18, Round(vmulq_f32(Load<false>(fdt + DF * 3), d1.val[1])));
                Store<false>(dst + 0x20, Round(vmulq_f32(Load<false>(fdt + DF * 4), d2.val[0])));
                Store<false>(dst + 0x28, Round(vmulq_f32(Load<false>(fdt + DF * 5), d2.val[1])));
                Store<false>(dst + 0x30, Round(vmulq_f32(Load<false>(fdt + DF * 6), d3.val[0])));
                Store<false>(dst + 0x38, Round(vmulq_f32(Load<false>(fdt + DF * 7), d3.val[1])));
            }
        }

        static int JpegProcessDu(Base::BitBuf& bitBuf, float* CDU, int stride, const float* fdtbl, int DC, const uint16_t HTDC[256][2], const uint16_t HTAC[256][2])
        {
            SIMD_ALIGNED(16) float BUF[64];
            JpegDctV(CDU, stride, BUF, 8);
            SIMD_ALIGNED(16) int DUO[64], DU[64];
            JpegDctH(BUF, 8, fdtbl, DUO);
            for (int i = 0; i < 64; ++i)
                DU[Base::JpegZigZagT[i]] = DUO[i];
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
            return DU[0];
        }

        SIMD_INLINE void RgbToYuvInit(float32x4_t k[10])
        {
            k[0] = vdupq_n_f32(+0.29900f);
            k[1] = vdupq_n_f32(+0.58700f);
            k[2] = vdupq_n_f32(+0.11400f);
            k[3] = vdupq_n_f32(-128.000f);
            k[4] = vdupq_n_f32(-0.16874f);
            k[5] = vdupq_n_f32(-0.33126f);
            k[6] = vdupq_n_f32(+0.50000f);
            k[7] = vdupq_n_f32(+0.50000f);
            k[8] = vdupq_n_f32(-0.41869f);
            k[9] = vdupq_n_f32(-0.08131f);
        }

        SIMD_INLINE void RgbToYuv(const uint8_t* r, const uint8_t* g, const uint8_t* b, int stride, int height, 
            const float32x4_t k[10], float* y, float* u, float* v, int size)
        {
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += 4)
                {
                    float32x4_t _r = vcvtq_f32_u32(vmovl_u16(Half<0>(vmovl_u8(LoadHalf<false>(r + col)))));
                    float32x4_t _g = vcvtq_f32_u32(vmovl_u16(Half<0>(vmovl_u8(LoadHalf<false>(g + col)))));
                    float32x4_t _b = vcvtq_f32_u32(vmovl_u16(Half<0>(vmovl_u8(LoadHalf<false>(b + col)))));
                    Store<false>(y + col, vaddq_f32(vaddq_f32(vaddq_f32(vmulq_f32(_r, k[0]), vmulq_f32(_g, k[1])), vmulq_f32(_b, k[2])), k[3]));
                    Store<false>(u + col, vaddq_f32(vaddq_f32(vmulq_f32(_r, k[4]), vmulq_f32(_g, k[5])), vmulq_f32(_b, k[6])));
                    Store<false>(v + col, vaddq_f32(vaddq_f32(vmulq_f32(_r, k[7]), vmulq_f32(_g, k[8])), vmulq_f32(_b, k[9])));
                }
                if(++row < height)
                    r += stride, g += stride, b += stride;
                y += size, u += size, v += size;
            }
        }

        SIMD_INLINE void GrayToY(const uint8_t* g, int stride, int height, float* y, int size)
        {
            float32x4_t k = vdupq_n_f32(-128.000f);
            for (int row = 0; row < size;)
            {
                for (int col = 0; col < size; col += 4)
                {
                    float32x4_t _g = vcvtq_f32_u32(vmovl_u16(Half<0>(vmovl_u8(LoadHalf<false>(g + col)))));
                    Store<false>(y + col, vaddq_f32(_g, k));
                }
                if (++row < height)
                    g += stride;
                y += size;
            }
        }

        SIMD_INLINE void SubUv(const float * src, float * dst)
        {
            float32x4_t _0_25 = vdupq_n_f32(0.25f), s0, s1;
            for (int yy = 0; yy < 8; yy += 1)
            {
                s0 = vaddq_f32(Load<false>(src + 0), Load<false>(src + 16));
                s1 = vaddq_f32(Load<false>(src + 4), Load<false>(src + 20));
                Store<false>(dst + 0, vmulq_f32(Hadd(s0, s1), _0_25));
                s0 = vaddq_f32(Load<false>(src + 8), Load<false>(src + 24));
                s1 = vaddq_f32(Load<false>(src + 12), Load<false>(src + 28));
                Store<false>(dst + 4, vmulq_f32(Hadd(s0, s1), _0_25));
                src += 32;
                dst += 8;
            }
        }

        SIMD_INLINE void Nv12ToUv(const uint8_t* uvSrc, int uvStride, int height, float* u, float* v)
        {
            float32x4_t k = vdupq_n_f32(-128.000f);
            for (int row = 0; row < 8;)
            {
                uint8x8x2_t _uv = LoadHalf2<false>(uvSrc);
                Store<false>(u + 0 * F, vaddq_f32(vcvtq_f32_u32(vmovl_u16(Half<0>(vmovl_u8(_uv.val[0])))), k));
                Store<false>(u + 1 * F, vaddq_f32(vcvtq_f32_u32(vmovl_u16(Half<1>(vmovl_u8(_uv.val[0])))), k));
                Store<false>(v + 0 * F, vaddq_f32(vcvtq_f32_u32(vmovl_u16(Half<0>(vmovl_u8(_uv.val[1])))), k));
                Store<false>(v + 1 * F, vaddq_f32(vcvtq_f32_u32(vmovl_u16(Half<1>(vmovl_u8(_uv.val[1])))), k));
                if (++row < height)
                    uvSrc += uvStride;
                u += 8, v += 8;
            }
        }

        void JpegWriteBlockSubs(OutputMemoryStream& stream, int width, int height, const uint8_t* red,
            const uint8_t* green, const uint8_t* blue, int stride, const float* fY, const float* fUv, int dc[3])
        {
            bool gray = red == green && red == blue;
            float32x4_t k[10];
            if(!gray)
                RgbToYuvInit(k);
            int& DCY = dc[0], & DCU = dc[1], & DCV = dc[2];
            int width16 = width& (~15);
            Base::BitBuf bitBuf;
            for (int y = 0; y < height; y += 16)
            {
                int x = 0;
                SIMD_ALIGNED(16) float Y[256], U[256], V[256];
                SIMD_ALIGNED(16) float subU[64], subV[64];
                for (; x < width16; x += 16)
                {
                    if (gray)
                        GrayToY(red + x, stride, height - y, Y, 16);
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
            float32x4_t k[10];
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
                        GrayToY(red + x, stride, height - y, Y, 8);
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
            SIMD_ALIGNED(16) float Y[256], U[64], V[64];
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
            SIMD_ALIGNED(16) float Y[256], U[64], V[64];
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
            : Base::ImageJpegSaver(param)
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
                    _deintBgr = _param.width < 16 ? Base::DeinterleaveBgr : Neon::DeinterleaveBgr;
                    break;
                case SimdPixelFormatBgra32:
                case SimdPixelFormatRgba32:
                    _deintBgra = _param.width < 16 ? Base::DeinterleaveBgra : Neon::DeinterleaveBgra;
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
#endif// SIMD_NEON_ENABLE
}

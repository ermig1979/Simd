/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2025 Yermalayeu Ihar.
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
#include "Simd/SimdBgrToLab.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdStore.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE
    namespace Neon
    {
        SIMD_INLINE int32x4_t CbrtIndex(int32x4_t r, int32x4_t g, int32x4_t b, const int32x4_t* c, int32x4_t round)
        {
            int32x4_t i = vaddq_s32(vaddq_s32(vmulq_s32(r, c[0]), vmulq_s32(g, c[1])), vmulq_s32(b, c[2]));
            return vshrq_n_s32(vaddq_s32(i, round), Base::LAB_SHIFT);
        }

        void BgrToLab(const uint8_t* bgr, size_t bgrStride, size_t width, size_t height, uint8_t* lab, size_t labStride)
        {
            Base::LabInitTabs();

            // Process F=4 pixels per SIMD iteration.
            // We write 8 valid bytes + 4 bytes that overlap the next iteration (or scalar tail),
            // so we leave at least 2 pixels as a tail (same trick as Sse41::BgrToLab).
            size_t widthF = AlignLo(Max<size_t>(width, 2) - 2, F);

            int32x4_t coeffs[Base::LabCoeffsTabSize];
            for (size_t i = 0; i < Base::LabCoeffsTabSize; ++i)
                coeffs[i] = vdupq_n_s32(Base::LabCoeffsTab[i]);

            const int32x4_t labRound   = vdupq_n_s32(Base::LAB_ROUND);
            const int32x4_t labLScale  = vdupq_n_s32(Base::LAB_L_SCALE);
            const int32x4_t labLShift  = vdupq_n_s32(Base::LAB_L_SHIFT);
            const int32x4_t labAScale  = vdupq_n_s32(Base::LAB_A_SCALE);
            const int32x4_t labBScale  = vdupq_n_s32(Base::LAB_B_SCALE);
            const int32x4_t labAbShift = vdupq_n_s32(Base::LAB_AB_SHIFT);

            // vtbl2 index vectors for interleaving 4 pixels of LAB from the layout:
            //   La = [L0 L1 L2 L3 a0 a1 a2 a3]  (bytes 0-7)
            //   bz = [b0 b1 b2 b3  0  0  0  0]  (bytes 8-15 of the two-register table)
            // Desired output: [L0 a0 b0 L1 a1 b1 L2 a2 | b2 L3 a3 b3 ...]
            static const uint8_t idx0_data[8] = { 0, 4, 8, 1, 5, 9, 2, 6 };
            static const uint8_t idx1_data[8] = { 10, 3, 7, 11, 255, 255, 255, 255 };
            const uint8x8_t idx0 = vld1_u8(idx0_data);
            const uint8x8_t idx1 = vld1_u8(idx1_data);

            for (size_t row = 0; row < height; ++row)
            {
                const uint8_t* pBgr = bgr + row * bgrStride;
                const uint8_t* pEnd  = pBgr + width * 3;
                const uint8_t* pEndF = pBgr + widthF * 3;
                uint8_t* pLab = lab + row * labStride;

                for (; pBgr < pEndF; pBgr += 12, pLab += 12)
                {
                    // Scalar gamma table lookups for 4 pixels.
                    const uint32_t* gamma = Base::LabGammaTab;
                    SIMD_ALIGNED(16) int32_t R[4], G[4], B[4];
                    B[0] = gamma[pBgr[0]];  G[0] = gamma[pBgr[1]];  R[0] = gamma[pBgr[2]];
                    B[1] = gamma[pBgr[3]];  G[1] = gamma[pBgr[4]];  R[1] = gamma[pBgr[5]];
                    B[2] = gamma[pBgr[6]];  G[2] = gamma[pBgr[7]];  R[2] = gamma[pBgr[8]];
                    B[3] = gamma[pBgr[9]];  G[3] = gamma[pBgr[10]]; R[3] = gamma[pBgr[11]];

                    int32x4_t _R = vld1q_s32(R);
                    int32x4_t _G = vld1q_s32(G);
                    int32x4_t _B = vld1q_s32(B);

                    // NEON: compute XYZ indices.
                    SIMD_ALIGNED(16) int32_t iX[4], iY[4], iZ[4];
                    vst1q_s32(iX, CbrtIndex(_R, _G, _B, coeffs + 0, labRound));
                    vst1q_s32(iY, CbrtIndex(_R, _G, _B, coeffs + 3, labRound));
                    vst1q_s32(iZ, CbrtIndex(_R, _G, _B, coeffs + 6, labRound));

                    // Scalar cbrt table lookups for 4 pixels.
                    const uint32_t* cbrt = Base::LabCbrtTab;
                    SIMD_ALIGNED(16) int32_t fX[4], fY[4], fZ[4];
                    fX[0] = cbrt[iX[0]]; fX[1] = cbrt[iX[1]]; fX[2] = cbrt[iX[2]]; fX[3] = cbrt[iX[3]];
                    fY[0] = cbrt[iY[0]]; fY[1] = cbrt[iY[1]]; fY[2] = cbrt[iY[2]]; fY[3] = cbrt[iY[3]];
                    fZ[0] = cbrt[iZ[0]]; fZ[1] = cbrt[iZ[1]]; fZ[2] = cbrt[iZ[2]]; fZ[3] = cbrt[iZ[3]];

                    int32x4_t _fX = vld1q_s32(fX);
                    int32x4_t _fY = vld1q_s32(fY);
                    int32x4_t _fZ = vld1q_s32(fZ);

                    // NEON: compute L, a, b.
                    int32x4_t _L = vshrq_n_s32(vaddq_s32(vmulq_s32(_fY, labLScale), labLShift), Base::LAB_SHIFT2);
                    int32x4_t _a = vshrq_n_s32(vaddq_s32(vmulq_s32(vsubq_s32(_fX, _fY), labAScale), labAbShift), Base::LAB_SHIFT2);
                    int32x4_t _b = vshrq_n_s32(vaddq_s32(vmulq_s32(vsubq_s32(_fY, _fZ), labBScale), labAbShift), Base::LAB_SHIFT2);

                    // Pack int32x4 -> int16x4 -> uint8x8.
                    // La = [L0 L1 L2 L3 a0 a1 a2 a3], bz = [b0 b1 b2 b3 0 0 0 0]
                    uint8x8_t La = vqmovun_s16(vcombine_s16(vqmovn_s32(_L), vqmovn_s32(_a)));
                    uint8x8_t bz = vqmovun_s16(vcombine_s16(vqmovn_s32(_b), vdup_n_s16(0)));

                    // Interleave to get LAB LAB LAB LAB layout using vtbl2.
                    uint8x8x2_t tbl = {{ La, bz }};
                    uint8x8_t out0 = vtbl2_u8(tbl, idx0); // L0 a0 b0 L1 a1 b1 L2 a2
                    uint8x8_t out1 = vtbl2_u8(tbl, idx1); // b2 L3 a3 b3 (+ 4 padding zeros)

                    // Store 8 valid bytes then 4 valid bytes (4 zeros overlap next iter / scalar tail).
                    vst1_u8(pLab, out0);
                    vst1_u8(pLab + 8, out1);
                }

                for (; pBgr < pEnd; pBgr += 3, pLab += 3)
                    Base::RgbToLab(pBgr[2], pBgr[1], pBgr[0], pLab);
            }
        }
    }
#endif // SIMD_NEON_ENABLE
}

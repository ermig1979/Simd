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
#include "Simd/SimdStore.h"
#include "Simd/SimdBFloat16.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        TileConf g_tileConf;
        TileReg g_tileRegs[TileRegCount];

        //-----------------------------------------------------------------------------------------

        void TileLoadConfig(const TileConf* tileConf)
        {
            memcpy(&g_tileConf, tileConf, sizeof(TileConf));
        }

        //-----------------------------------------------------------------------------------------

        void TileStoreConfig(TileConf* tileConf)
        {
            memcpy(tileConf, &g_tileConf, sizeof(TileConf));
        }

        //-----------------------------------------------------------------------------------------

        void TileRelease()
        {
            g_tileConf = TileConf();
        }

        //-----------------------------------------------------------------------------------------

        void TileZero(int tile)
        {
            assert(tile < TileRegCount);

            memset(g_tileRegs + tile, 0, sizeof(TileReg));
        }

        //-----------------------------------------------------------------------------------------

        void TileLoad(int dst, const void* base, int stride)
        {
            assert(dst < TileRegCount);

            TileReg& dstTile = g_tileRegs[dst];
            int colbs = g_tileConf.colsb[dst], rows = g_tileConf.rows[dst], start = g_tileConf.startRow;

            assert(colbs <= 64 && rows <= 16);

            if (start == 0)
                memset(&dstTile, 0, sizeof(TileReg));

            __mmask64 tail = TailMask64(colbs);
            for (; start < rows; start++)
            {
                __m512i val = _mm512_maskz_loadu_epi8(tail, (uint8_t*)base + start * stride);
                _mm512_mask_storeu_epi8(dstTile.u8[start], tail, val);
            }
        }

        //-----------------------------------------------------------------------------------------

        void TileStore(int src, void* base, int stride)
        {
            assert(src < TileRegCount);

            const TileReg& srcTile = g_tileRegs[src];
            int colbs = g_tileConf.colsb[src], rows = g_tileConf.rows[src], start = g_tileConf.startRow;

            assert(colbs <= 64 && rows <= 16);

            __mmask64 tail = TailMask64(colbs);
            for (; start < rows; start++)
            {
                __m512i val = _mm512_maskz_loadu_epi8(tail, srcTile.u8[start]);
                _mm512_mask_storeu_epi8((uint8_t*)base + start * stride, tail, val);
            }
        }

        //-----------------------------------------------------------------------------------------

        void TileMatMulBf16(int dst, int a, int b)
        {
            assert(dst < TileRegCount && a < TileRegCount && b < TileRegCount);

            TileReg& dstTile = g_tileRegs[dst];
            const TileReg& aTile = g_tileRegs[a];
            const TileReg& bTile = g_tileRegs[b];

            int M = g_tileConf.rows[dst], N = g_tileConf.colsb[dst] / 4, K = g_tileConf.colsb[a] / 4;

            assert(M <= 16 && N <= 16 && K <= 16);

            __mmask16 tailN = TailMask16(N);

            static const __m512 mask = _mm512_castsi512_ps(Bf16::MASK);
            __m512 a0, a1, b0, b1;
            int m = 0;
            for (int M4 = M & (~3); m < M4; m += 4)
            {
                __m512 dst0 = _mm512_maskz_loadu_ps(tailN, dstTile.f32[m + 0]);
                __m512 dst1 = _mm512_maskz_loadu_ps(tailN, dstTile.f32[m + 1]);
                __m512 dst2 = _mm512_maskz_loadu_ps(tailN, dstTile.f32[m + 2]);
                __m512 dst3 = _mm512_maskz_loadu_ps(tailN, dstTile.f32[m + 3]);
                for (int k = 0; k < K; k++)
                {
                    b1 = _mm512_maskz_loadu_ps(tailN, bTile.f32[k]);
                    b0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(b1), Base::Bf16::SHIFT));
                    b1 = _mm512_and_ps(b1, mask);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, aTile.u32[m + 0][k]));
                    dst0 = _mm512_fmadd_ps(a0, b0, dst0);
                    a1 = _mm512_and_ps(_mm512_set1_ps(aTile.f32[m + 0][k]), mask);
                    dst0 = _mm512_fmadd_ps(a1, b1, dst0);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, aTile.u32[m + 1][k]));
                    dst1 = _mm512_fmadd_ps(a0, b0, dst1);
                    a1 = _mm512_and_ps(_mm512_set1_ps(aTile.f32[m + 1][k]), mask);
                    dst1 = _mm512_fmadd_ps(a1, b1, dst1);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, aTile.u32[m + 2][k]));
                    dst2 = _mm512_fmadd_ps(a0, b0, dst2);
                    a1 = _mm512_and_ps(_mm512_set1_ps(aTile.f32[m + 2][k]), mask);
                    dst2 = _mm512_fmadd_ps(a1, b1, dst2);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, aTile.u32[m + 3][k]));
                    dst3 = _mm512_fmadd_ps(a0, b0, dst3);
                    a1 = _mm512_and_ps(_mm512_set1_ps(aTile.f32[m + 3][k]), mask);
                    dst3 = _mm512_fmadd_ps(a1, b1, dst3);
                }
                _mm512_mask_storeu_ps(dstTile.f32[m + 0], tailN, dst0);
                _mm512_mask_storeu_ps(dstTile.f32[m + 1], tailN, dst1);
                _mm512_mask_storeu_ps(dstTile.f32[m + 2], tailN, dst2);
                _mm512_mask_storeu_ps(dstTile.f32[m + 3], tailN, dst3);
            }
            for (int M2 = M & (~1); m < M2; m += 2)
            {
                __m512 dst0 = _mm512_maskz_loadu_ps(tailN, dstTile.f32[m + 0]);
                __m512 dst1 = _mm512_maskz_loadu_ps(tailN, dstTile.f32[m + 1]);
                for (int k = 0; k < K; k++)
                {
                    b1 = _mm512_maskz_loadu_ps(tailN, bTile.f32[k]);
                    b0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(b1), Base::Bf16::SHIFT));
                    b1 = _mm512_and_ps(b1, mask);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, aTile.u32[m + 0][k]));
                    dst0 = _mm512_fmadd_ps(a0, b0, dst0);
                    a1 = _mm512_and_ps(_mm512_set1_ps(aTile.f32[m + 0][k]), mask);
                    dst0 = _mm512_fmadd_ps(a1, b1, dst0);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, aTile.u32[m + 1][k]));
                    dst1 = _mm512_fmadd_ps(a0, b0, dst1);
                    a1 = _mm512_and_ps(_mm512_set1_ps(aTile.f32[m + 1][k]), mask);
                    dst1 = _mm512_fmadd_ps(a1, b1, dst1);
                }
                _mm512_mask_storeu_ps(dstTile.f32[m + 0], tailN, dst0);
                _mm512_mask_storeu_ps(dstTile.f32[m + 1], tailN, dst1);
            }
            for (; m < M; m++)
            {
                __m512 dst0 = _mm512_maskz_loadu_ps(tailN, dstTile.f32[m]);
                for (int k = 0; k < K; k++)
                {
                    __m512 _b = _mm512_maskz_loadu_ps(tailN, bTile.f32[k]);
                    __m512 b0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(_b), Base::Bf16::SHIFT));
                    __m512 b1 = _mm512_and_ps(_b, mask);

                    __m512 a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, aTile.u32[m][k]));
                    dst0 = _mm512_fmadd_ps(a0, b0, dst0);
                    __m512 a1 = _mm512_and_ps(_mm512_set1_ps(aTile.f32[m][k]), mask);
                    dst0 = _mm512_fmadd_ps(a1, b1, dst0);
                }
                _mm512_mask_storeu_ps(dstTile.f32[m], tailN, dst0);
            }
        }
    }
#endif
}

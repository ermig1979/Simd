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
            size_t colbs = g_tileConf.colsb[dst], rows = g_tileConf.rows[dst], start = g_tileConf.startRow;

            assert(colbs <= 64 && rows <= 16);

            if (start == 0)
                memset(&dstTile, 0, sizeof(TileReg));

            for (; start < rows; start++)
                memcpy(dstTile.u8[start], (uint8_t*)base + start * stride, colbs);
        }

        //-----------------------------------------------------------------------------------------

        void TileStore(int src, void* base, int stride)
        {
            assert(src < TileRegCount);

            const TileReg& srcTile = g_tileRegs[src];
            size_t colbs = g_tileConf.colsb[src], rows = g_tileConf.rows[src], start = g_tileConf.startRow;

            assert(colbs <= 64 && rows <= 16);

            for (; start < rows; start++)
                memcpy((uint8_t*)base + start * stride, srcTile.u8[start], colbs);
        }

        //-----------------------------------------------------------------------------------------

        void TileMatMulBf16(int dst, int a, int b)
        {
            assert(dst < TileRegCount && a < TileRegCount && b < TileRegCount);

            TileReg& dstTile = g_tileRegs[dst];
            const TileReg& aTile = g_tileRegs[a];
            const TileReg& bTile = g_tileRegs[b];

            size_t M = g_tileConf.rows[dst], N = g_tileConf.colsb[dst] / 4, K = g_tileConf.colsb[a] / 4;

            assert(M <= 16 && N <= 16 && K <= 16);

            __mmask16 tileN = TailMask16(N);

            static const __m512 mask = _mm512_castsi512_ps(Bf16::MASK);
            for (size_t m = 0; m < M; m++)
            {
                __m512 _dst = _mm512_loadu_ps(dstTile.f32[m]);
                for (size_t k = 0; k < K; k++)
                {
                    __m512 _b = _mm512_maskz_loadu_ps(tileN, bTile.f32[k]);
                    __m512 b0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(_b), Base::Bf16::SHIFT));
                    __m512 b1 = _mm512_and_ps(_b, mask);

                    __m512 a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, aTile.u32[m][k]));
                    _dst = _mm512_fmadd_ps(a0, b0, _dst);
                    __m512 a1 = _mm512_and_ps(_mm512_set1_ps(aTile.f32[m][k]), mask);
                    _dst = _mm512_fmadd_ps(a1, b1, _dst);
                }
            }
        }
    }
#endif
}

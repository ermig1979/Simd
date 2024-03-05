/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)    
    namespace Avx512bw
    {
        TileConf g_tileConf;
        TileReg g_tileRegs[TileRegCount];

        //-----------------------------------------------------------------------------------------

        void TileLoadConfig(const TileConf* tileConf)
        {
            memcpy(&g_tileConf, tileConf, sizeof(TileConf));
            memset(g_tileRegs, 0, sizeof(TileReg) * TileRegCount);
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

        void TileZero(Tile1024* dst)
        {
            memset(&dst->tile, 0, sizeof(TileReg));
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void TileLoad(TileReg & dst, int rows, int colsb, const void* base, int stride)
        {
            int start = g_tileConf.startRow;

            assert(colsb <= 64 && rows <= 16);

            if (start == 0)
                memset(&dst, 0, sizeof(TileReg));

            __mmask64 tail = TailMask64(colsb);
            for (; start < rows; start++)
            {
                __m512i val = _mm512_maskz_loadu_epi8(tail, (uint8_t*)base + start * stride);
                _mm512_mask_storeu_epi8(dst.u8[start], tail, val);
            }
        }

        void TileLoad(int dst, const void* base, int stride)
        {
            assert(dst < TileRegCount);

            TileLoad(g_tileRegs[dst], g_tileConf.rows[dst], g_tileConf.colsb[dst], base, stride);
        }

        void TileLoad(Tile1024* dst, const void* base, int stride)
        {
            TileLoad(dst->tile, dst->row, dst->col, base, stride);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void TileStore(const TileReg& src, int rows, int colsb, void* base, int stride)
        {
            int start = g_tileConf.startRow;

            assert(colsb <= 64 && rows <= 16);

            __mmask64 tail = TailMask64(colsb);
            for (; start < rows; start++)
            {
                __m512i val = _mm512_maskz_loadu_epi8(tail, src.u8[start]);
                _mm512_mask_storeu_epi8((uint8_t*)base + start * stride, tail, val);
            }
        }

        void TileStore(int src, void* base, int stride)
        {
            assert(src < TileRegCount);

            TileStore(g_tileRegs[src], g_tileConf.rows[src], g_tileConf.colsb[src], base, stride);
        }

        void TileStore(void* base, int stride, const Tile1024& src)
        {
            TileStore(src.tile, src.row, src.col, base, stride);
        }

        //-----------------------------------------------------------------------------------------

        template<bool overflow> SIMD_INLINE void TileMatMul8u8i(int M, int N, int K, TileReg& dst, const TileReg& a, const TileReg& b)
        {
            assert(M <= 16 && N <= 16 && K <= 16);

            __mmask16 tailN = TailMask16(N);

            __m512i a0, b0;
            int m = 0;
            for (int M4 = M & (~3); m < M4; m += 4)
            {
                __m512i d0 = _mm512_maskz_loadu_epi32(tailN, dst.i32[m + 0]);
                __m512i d1 = _mm512_maskz_loadu_epi32(tailN, dst.i32[m + 1]);
                __m512i d2 = _mm512_maskz_loadu_epi32(tailN, dst.i32[m + 2]);
                __m512i d3 = _mm512_maskz_loadu_epi32(tailN, dst.i32[m + 3]);
                for (int k = 0; k < K; k++)
                {
                    b0 = _mm512_maskz_loadu_epi32(tailN, b.i32[k]);
                    a0 = _mm512_set1_epi32(a.i32[m + 0][k]), Madd4<overflow>(d0, a0, b0);
                    a0 = _mm512_set1_epi32(a.i32[m + 1][k]), Madd4<overflow>(d1, a0, b0);
                    a0 = _mm512_set1_epi32(a.i32[m + 2][k]), Madd4<overflow>(d2, a0, b0);
                    a0 = _mm512_set1_epi32(a.i32[m + 3][k]), Madd4<overflow>(d3, a0, b0);
                }
                _mm512_mask_storeu_epi32(dst.i32[m + 0], tailN, d0);
                _mm512_mask_storeu_epi32(dst.i32[m + 1], tailN, d1);
                _mm512_mask_storeu_epi32(dst.i32[m + 2], tailN, d2);
                _mm512_mask_storeu_epi32(dst.i32[m + 3], tailN, d3);
            }
            for (int M2 = M & (~1); m < M2; m += 2)
            {
                __m512i d0 = _mm512_maskz_loadu_epi32(tailN, dst.i32[m + 0]);
                __m512i d1 = _mm512_maskz_loadu_epi32(tailN, dst.i32[m + 1]);
                for (int k = 0; k < K; k++)
                {
                    b0 = _mm512_maskz_loadu_epi32(tailN, b.i32[k]);
                    a0 = _mm512_set1_epi32(a.i32[m + 0][k]), Madd4<overflow>(d0, a0, b0);
                    a0 = _mm512_set1_epi32(a.i32[m + 1][k]), Madd4<overflow>(d1, a0, b0);
                }
                _mm512_mask_storeu_epi32(dst.i32[m + 0], tailN, d0);
                _mm512_mask_storeu_epi32(dst.i32[m + 1], tailN, d1);
            }
            for (; m < M; m++)
            {
                __m512i d0 = _mm512_maskz_loadu_epi32(tailN, dst.i32[m]);
                for (int k = 0; k < K; k++)
                {
                    b0 = _mm512_maskz_loadu_epi32(tailN, b.i32[k]);
                    a0 = _mm512_set1_epi32(a.i32[m][k]), Madd4<overflow>(d0, a0, b0);
                }
                _mm512_mask_storeu_epi32(dst.i32[m], tailN, d0);
            }
        }

        void TileMatMul8u8i(int dst, int a, int b)
        {
            assert(dst < TileRegCount&& a < TileRegCount&& b < TileRegCount); 
            
            TileMatMul8u8i<true>(g_tileConf.rows[dst], g_tileConf.colsb[dst] / 4, g_tileConf.colsb[a] / 4, g_tileRegs[dst], g_tileRegs[a], g_tileRegs[b]);
        }

        void TileMatMul8u8i(Tile1024* dst, const Tile1024& a, const Tile1024& b)
        {
            TileMatMul8u8i<true>(dst->row, dst->col, b.row, dst->tile, a.tile, b.tile);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void TileMatMulBf16(int M, int N, int K, TileReg& dst, const TileReg& a, const TileReg& b)
        {
            assert(M <= 16 && N <= 16 && K <= 16);

            __mmask16 tailN = TailMask16(N);

            static const __m512 mask = _mm512_castsi512_ps(Bf16::MASK);
            __m512 a0, a1, b0, b1;
            int m = 0;
            for (int M4 = M & (~3); m < M4; m += 4)
            {
                __m512 dst0 = _mm512_maskz_loadu_ps(tailN, dst.f32[m + 0]);
                __m512 dst1 = _mm512_maskz_loadu_ps(tailN, dst.f32[m + 1]);
                __m512 dst2 = _mm512_maskz_loadu_ps(tailN, dst.f32[m + 2]);
                __m512 dst3 = _mm512_maskz_loadu_ps(tailN, dst.f32[m + 3]);
                for (int k = 0; k < K; k++)
                {
                    b1 = _mm512_maskz_loadu_ps(tailN, b.f32[k]);
                    b0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(b1), Base::Bf16::SHIFT));
                    b1 = _mm512_and_ps(b1, mask);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, a.u32[m + 0][k]));
                    dst0 = _mm512_fmadd_ps(a0, b0, dst0);
                    a1 = _mm512_and_ps(_mm512_set1_ps(a.f32[m + 0][k]), mask);
                    dst0 = _mm512_fmadd_ps(a1, b1, dst0);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, a.u32[m + 1][k]));
                    dst1 = _mm512_fmadd_ps(a0, b0, dst1);
                    a1 = _mm512_and_ps(_mm512_set1_ps(a.f32[m + 1][k]), mask);
                    dst1 = _mm512_fmadd_ps(a1, b1, dst1);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, a.u32[m + 2][k]));
                    dst2 = _mm512_fmadd_ps(a0, b0, dst2);
                    a1 = _mm512_and_ps(_mm512_set1_ps(a.f32[m + 2][k]), mask);
                    dst2 = _mm512_fmadd_ps(a1, b1, dst2);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, a.u32[m + 3][k]));
                    dst3 = _mm512_fmadd_ps(a0, b0, dst3);
                    a1 = _mm512_and_ps(_mm512_set1_ps(a.f32[m + 3][k]), mask);
                    dst3 = _mm512_fmadd_ps(a1, b1, dst3);
                }
                _mm512_mask_storeu_ps(dst.f32[m + 0], tailN, dst0);
                _mm512_mask_storeu_ps(dst.f32[m + 1], tailN, dst1);
                _mm512_mask_storeu_ps(dst.f32[m + 2], tailN, dst2);
                _mm512_mask_storeu_ps(dst.f32[m + 3], tailN, dst3);
            }
            for (int M2 = M & (~1); m < M2; m += 2)
            {
                __m512 dst0 = _mm512_maskz_loadu_ps(tailN, dst.f32[m + 0]);
                __m512 dst1 = _mm512_maskz_loadu_ps(tailN, dst.f32[m + 1]);
                for (int k = 0; k < K; k++)
                {
                    b1 = _mm512_maskz_loadu_ps(tailN, b.f32[k]);
                    b0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(b1), Base::Bf16::SHIFT));
                    b1 = _mm512_and_ps(b1, mask);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, a.u32[m + 0][k]));
                    dst0 = _mm512_fmadd_ps(a0, b0, dst0);
                    a1 = _mm512_and_ps(_mm512_set1_ps(a.f32[m + 0][k]), mask);
                    dst0 = _mm512_fmadd_ps(a1, b1, dst0);

                    a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, a.u32[m + 1][k]));
                    dst1 = _mm512_fmadd_ps(a0, b0, dst1);
                    a1 = _mm512_and_ps(_mm512_set1_ps(a.f32[m + 1][k]), mask);
                    dst1 = _mm512_fmadd_ps(a1, b1, dst1);
                }
                _mm512_mask_storeu_ps(dst.f32[m + 0], tailN, dst0);
                _mm512_mask_storeu_ps(dst.f32[m + 1], tailN, dst1);
            }
            for (; m < M; m++)
            {
                __m512 dst0 = _mm512_maskz_loadu_ps(tailN, dst.f32[m]);
                for (int k = 0; k < K; k++)
                {
                    __m512 _b = _mm512_maskz_loadu_ps(tailN, b.f32[k]);
                    __m512 b0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(_b), Base::Bf16::SHIFT));
                    __m512 b1 = _mm512_and_ps(_b, mask);

                    __m512 a0 = _mm512_castsi512_ps(_mm512_maskz_set1_epi16(0xAAAAAAAA, a.u32[m][k]));
                    dst0 = _mm512_fmadd_ps(a0, b0, dst0);
                    __m512 a1 = _mm512_and_ps(_mm512_set1_ps(a.f32[m][k]), mask);
                    dst0 = _mm512_fmadd_ps(a1, b1, dst0);
                }
                _mm512_mask_storeu_ps(dst.f32[m], tailN, dst0);
            }
        }

        void TileMatMulBf16(int dst, int a, int b)
        {
            assert(dst < TileRegCount&& a < TileRegCount&& b < TileRegCount);

            TileMatMulBf16(g_tileConf.rows[dst], g_tileConf.colsb[dst] / 4, g_tileConf.colsb[a] / 4, g_tileRegs[dst], g_tileRegs[a], g_tileRegs[b]);
        }

        void TileMatMulBf16(Tile1024* dst, const Tile1024& a, const Tile1024& b)
        {
            TileMatMulBf16(dst->row, dst->col, b.row, dst->tile, a.tile, b.tile);
        }

        //-----------------------------------------------------------------------------------------

        SIMD_INLINE void TileMatMulFp16(int M, int N, int K, TileReg& dst, const TileReg& a, const TileReg& b)
        {
            static const __m512i B_PERM = SIMD_MM512_SETR_EPI16(
                0x00, 0x02, 0x04, 0x06, 0x08, 0x0A, 0x0C, 0x0E, 0x10, 0x12, 0x14, 0x16, 0x18, 0x1A, 0x1C, 0x1E,
                0x01, 0x03, 0x05, 0x07, 0x09, 0x0B, 0x0D, 0x0F, 0x11, 0x13, 0x15, 0x17, 0x19, 0x1B, 0x1D, 0x1F);
            assert(M <= 16 && N <= 16 && K <= 16);

            __mmask16 tailN = TailMask16(N);

            int m = 0;
            for (; m < M; m++)
            {
                __m512 dst0 = _mm512_maskz_loadu_ps(tailN, dst.f32[m]);
                for (int k = 0; k < K; k ++)
                {
                    __m512i _b = _mm512_permutexvar_epi16(B_PERM, _mm512_maskz_loadu_epi32(tailN, b.u32[k]));
                    __m512 b0 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_b, 0));
                    __m512 b1 = _mm512_cvtph_ps(_mm512_extracti64x4_epi64(_b, 1));

                    __m512 _a = _mm512_broadcast_f32x2(_mm_cvtph_ps(_mm_cvtsi32_si128(a.u32[m][k])));
                    dst0 = _mm512_fmadd_ps(Broadcast<0>(_a), b0, dst0);
                    dst0 = _mm512_fmadd_ps(Broadcast<1>(_a), b1, dst0);
                }
                _mm512_mask_storeu_ps(dst.f32[m], tailN, dst0);
            }
        }

        void TileMatMulFp16(int dst, int a, int b)
        {
            assert(dst < TileRegCount&& a < TileRegCount&& b < TileRegCount);

            TileMatMulFp16(g_tileConf.rows[dst], g_tileConf.colsb[dst] / 4, g_tileConf.colsb[a] / 4, g_tileRegs[dst], g_tileRegs[a], g_tileRegs[b]);
        }

        void TileMatMulFp16(Tile1024* dst, const Tile1024& a, const Tile1024& b)
        {
            TileMatMulFp16(dst->row, dst->col, b.row, dst->tile, a.tile, b.tile);
        }
    }
#endif
}

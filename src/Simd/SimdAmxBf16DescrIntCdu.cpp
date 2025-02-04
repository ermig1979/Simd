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
#include "Simd/SimdMemory.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdArray.h"
#include "Simd/SimdUnpack.h"
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdTile.h"

namespace Simd
{
#ifdef SIMD_AMXBF16_ENABLE    
    namespace AmxBf16
    {
        template<int bits> void UnpackDataA(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t size64 = AlignLo(size, 64);
            __mmask64 srcBody = TailMask64(8 * bits), dstBody = __mmask64(-1), srcTail = TailMask64((size - size64) / 8 * bits), dstTail = TailMask64(size - size64);
            for (size_t i = 0; i < count; i++)
            {
                const uint8_t* ps = src[i] + 16;
                uint8_t* pd = (uint8_t*)dst + i * stride;
                size_t j = 0;
                for (; j < size64; j += 64, ps += 8 * bits, pd += 64)
                    _mm512_mask_storeu_epi8(pd, dstBody, Load8u<bits>(ps, srcBody));
                if (j < size)
                    _mm512_mask_storeu_epi8(pd, dstBody, Load8u<bits>(ps, srcTail));
            }
        }

        //-------------------------------------------------------------------------------------------------

        template<int bits, int N> SIMD_INLINE void UnpackDataBx16xN(const uint8_t* const* src, size_t offset, uint8_t* dst)
        {
            __mmask64 mask = TailMask64(bits * N);
            __m512i a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, aA, aB, aC, aD, aE, aF, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, bA, bB, bC, bD, bE, bF;

            a0 = Load8u<bits>(src[0x0] + offset, mask);
            a1 = Load8u<bits>(src[0x1] + offset, mask);
            a2 = Load8u<bits>(src[0x2] + offset, mask);
            a3 = Load8u<bits>(src[0x3] + offset, mask);
            a4 = Load8u<bits>(src[0x4] + offset, mask);
            a5 = Load8u<bits>(src[0x5] + offset, mask);
            a6 = Load8u<bits>(src[0x6] + offset, mask);
            a7 = Load8u<bits>(src[0x7] + offset, mask);
            a8 = Load8u<bits>(src[0x8] + offset, mask);
            a9 = Load8u<bits>(src[0x9] + offset, mask);
            aA = Load8u<bits>(src[0xA] + offset, mask);
            aB = Load8u<bits>(src[0xB] + offset, mask);
            aC = Load8u<bits>(src[0xC] + offset, mask);
            aD = Load8u<bits>(src[0xD] + offset, mask);
            aE = Load8u<bits>(src[0xE] + offset, mask);
            aF = Load8u<bits>(src[0xF] + offset, mask);

            b0 = _mm512_unpacklo_epi32(a0, a2);
            b1 = _mm512_unpacklo_epi32(a1, a3);
            b2 = _mm512_unpackhi_epi32(a0, a2);
            b3 = _mm512_unpackhi_epi32(a1, a3);
            b4 = _mm512_unpacklo_epi32(a4, a6);
            b5 = _mm512_unpacklo_epi32(a5, a7);
            b6 = _mm512_unpackhi_epi32(a4, a6);
            b7 = _mm512_unpackhi_epi32(a5, a7);
            b8 = _mm512_unpacklo_epi32(a8, aA);
            b9 = _mm512_unpacklo_epi32(a9, aB);
            bA = _mm512_unpackhi_epi32(a8, aA);
            bB = _mm512_unpackhi_epi32(a9, aB);
            bC = _mm512_unpacklo_epi32(aC, aE);
            bD = _mm512_unpacklo_epi32(aD, aF);
            bE = _mm512_unpackhi_epi32(aC, aE);
            bF = _mm512_unpackhi_epi32(aD, aF);

            a0 = _mm512_unpacklo_epi32(b0, b1);
            a1 = _mm512_unpackhi_epi32(b0, b1);
            a2 = _mm512_unpacklo_epi32(b2, b3);
            a3 = _mm512_unpackhi_epi32(b2, b3);
            a4 = _mm512_unpacklo_epi32(b4, b5);
            a5 = _mm512_unpackhi_epi32(b4, b5);
            a6 = _mm512_unpacklo_epi32(b6, b7);
            a7 = _mm512_unpackhi_epi32(b6, b7);
            a8 = _mm512_unpacklo_epi32(b8, b9);
            a9 = _mm512_unpackhi_epi32(b8, b9);
            aA = _mm512_unpacklo_epi32(bA, bB);
            aB = _mm512_unpackhi_epi32(bA, bB);
            aC = _mm512_unpacklo_epi32(bC, bD);
            aD = _mm512_unpackhi_epi32(bC, bD);
            aE = _mm512_unpacklo_epi32(bE, bF);
            aF = _mm512_unpackhi_epi32(bE, bF);

            b0 = _mm512_shuffle_i32x4(a0, a4, 0x44);
            b1 = _mm512_shuffle_i32x4(a1, a5, 0x44);
            b2 = _mm512_shuffle_i32x4(a2, a6, 0x44);
            b3 = _mm512_shuffle_i32x4(a3, a7, 0x44);
            b4 = _mm512_shuffle_i32x4(a0, a4, 0xEE);
            b5 = _mm512_shuffle_i32x4(a1, a5, 0xEE);
            b6 = _mm512_shuffle_i32x4(a2, a6, 0xEE);
            b7 = _mm512_shuffle_i32x4(a3, a7, 0xEE);
            b8 = _mm512_shuffle_i32x4(a8, aC, 0x44);
            b9 = _mm512_shuffle_i32x4(a9, aD, 0x44);
            bA = _mm512_shuffle_i32x4(aA, aE, 0x44);
            bB = _mm512_shuffle_i32x4(aB, aF, 0x44);
            bC = _mm512_shuffle_i32x4(a8, aC, 0xEE);
            bD = _mm512_shuffle_i32x4(a9, aD, 0xEE);
            bE = _mm512_shuffle_i32x4(aA, aE, 0xEE);
            bF = _mm512_shuffle_i32x4(aB, aF, 0xEE);

            a0 = _mm512_shuffle_i32x4(b0, b8, 0x88);
            a1 = _mm512_shuffle_i32x4(b1, b9, 0x88);
            a2 = _mm512_shuffle_i32x4(b2, bA, 0x88);
            a3 = _mm512_shuffle_i32x4(b3, bB, 0x88);
            a4 = _mm512_shuffle_i32x4(b0, b8, 0xDD);
            a5 = _mm512_shuffle_i32x4(b1, b9, 0xDD);
            a6 = _mm512_shuffle_i32x4(b2, bA, 0xDD);
            a7 = _mm512_shuffle_i32x4(b3, bB, 0xDD);
            a8 = _mm512_shuffle_i32x4(b4, bC, 0x88);
            a9 = _mm512_shuffle_i32x4(b5, bD, 0x88);
            aA = _mm512_shuffle_i32x4(b6, bE, 0x88);
            aB = _mm512_shuffle_i32x4(b7, bF, 0x88);
            aC = _mm512_shuffle_i32x4(b4, bC, 0xDD);
            aD = _mm512_shuffle_i32x4(b5, bD, 0xDD);
            aE = _mm512_shuffle_i32x4(b6, bE, 0xDD);
            aF = _mm512_shuffle_i32x4(b7, bF, 0xDD);

            if (N > 0) _mm512_storeu_si512(dst + 0x0 * DA, a0); else _mm512_storeu_si512(dst + 0x0 * DA, _mm512_setzero_si512());
            if (N > 0) _mm512_storeu_si512(dst + 0x1 * DA, a1); else _mm512_storeu_si512(dst + 0x1 * DA, _mm512_setzero_si512());
            if (N > 1) _mm512_storeu_si512(dst + 0x2 * DA, a2); else _mm512_storeu_si512(dst + 0x2 * DA, _mm512_setzero_si512());
            if (N > 1) _mm512_storeu_si512(dst + 0x3 * DA, a3); else _mm512_storeu_si512(dst + 0x3 * DA, _mm512_setzero_si512());
            if (N > 2) _mm512_storeu_si512(dst + 0x4 * DA, a4); else _mm512_storeu_si512(dst + 0x4 * DA, _mm512_setzero_si512());
            if (N > 2) _mm512_storeu_si512(dst + 0x5 * DA, a5); else _mm512_storeu_si512(dst + 0x5 * DA, _mm512_setzero_si512());
            if (N > 3) _mm512_storeu_si512(dst + 0x6 * DA, a6); else _mm512_storeu_si512(dst + 0x6 * DA, _mm512_setzero_si512());
            if (N > 3) _mm512_storeu_si512(dst + 0x7 * DA, a7); else _mm512_storeu_si512(dst + 0x7 * DA, _mm512_setzero_si512());
            if (N > 4) _mm512_storeu_si512(dst + 0x8 * DA, a8); else _mm512_storeu_si512(dst + 0x8 * DA, _mm512_setzero_si512());
            if (N > 4) _mm512_storeu_si512(dst + 0x9 * DA, a9); else _mm512_storeu_si512(dst + 0x9 * DA, _mm512_setzero_si512());
            if (N > 5) _mm512_storeu_si512(dst + 0xA * DA, aA); else _mm512_storeu_si512(dst + 0xA * DA, _mm512_setzero_si512());
            if (N > 5) _mm512_storeu_si512(dst + 0xB * DA, aB); else _mm512_storeu_si512(dst + 0xB * DA, _mm512_setzero_si512());
            if (N > 6) _mm512_storeu_si512(dst + 0xC * DA, aC); else _mm512_storeu_si512(dst + 0xC * DA, _mm512_setzero_si512());
            if (N > 6) _mm512_storeu_si512(dst + 0xD * DA, aD); else _mm512_storeu_si512(dst + 0xD * DA, _mm512_setzero_si512());
            if (N > 7) _mm512_storeu_si512(dst + 0xE * DA, aE); else _mm512_storeu_si512(dst + 0xE * DA, _mm512_setzero_si512());
            if (N > 7) _mm512_storeu_si512(dst + 0xF * DA, aF); else _mm512_storeu_si512(dst + 0xF * DA, _mm512_setzero_si512());
}

        typedef void (*UnpackDataBx16xN_Ptr)(const uint8_t* const* src, size_t offset, uint8_t* dst);

        template<int bits> UnpackDataBx16xN_Ptr GetUnpackDataBx16xN(size_t tail)
        {
            switch (tail / 8)
            {
            case 0: return NULL;
            case 1: return UnpackDataBx16xN<bits, 1>;
            case 2: return UnpackDataBx16xN<bits, 2>;
            case 3: return UnpackDataBx16xN<bits, 3>;
            case 4: return UnpackDataBx16xN<bits, 4>;
            case 5: return UnpackDataBx16xN<bits, 5>;
            case 6: return UnpackDataBx16xN<bits, 6>;
            case 7: return UnpackDataBx16xN<bits, 7>;
            case 8: return UnpackDataBx16xN<bits, 8>;
            default:
                assert(0);  return NULL;
            }
        }

        template<int bits> void UnpackDataB(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride)
        {
            size_t countDF = AlignLo(count, DF), size64 = AlignLo(size, 64), tail = size - size64, i, j, o;
            UnpackDataBx16xN_Ptr unpackDataMain = GetUnpackDataBx16xN<bits>(64);
            UnpackDataBx16xN_Ptr unpackDataTail = GetUnpackDataBx16xN<bits>(tail);
            for (i = 0; i < countDF; i += DF, src += DF)
            {
                for (j = 0, o = 16; j < size64; j += 64, o += 8 * bits, dst += 32 * A)
                {
                    unpackDataMain(src + 0, o, dst + 0);
                    unpackDataMain(src + F, o, dst + A);
                }
                if (j < size)
                {
                    unpackDataTail(src + 0, o, dst + 0);
                    unpackDataTail(src + F, o, dst + A);
                    dst += 64 * DF;
                }
            }
            if (i < count)
            {
                size_t tail = count - countDF;
                const uint8_t* _src[DF];
                for (size_t j = 0; j < DF; i++, j++)
                    _src[j] = i < count ? *src++ : src[-1];
                for (j = 0, o = 16; j < size64; j += 64, o += 8 * bits, dst += 32 * A)
                {
                    unpackDataMain(_src + 0, o, dst + 0);
                    if (tail > F)
                        unpackDataMain(_src + F, o, dst + A);
                }
                if (j < size)
                {
                    unpackDataTail(_src + 0, o, dst + 0);
                    if (tail > F)
                        unpackDataTail(_src + F, o, dst + A);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        inline void Correlation8_32x32(size_t M, size_t N, size_t K, const uint8_t* ad00, size_t adStride, const uint8_t* bd00, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            const uint8_t* ad16 = ad00 + 16 * adStride, *bd64 = bd00 + 64;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[1] = 16;
            conf.rows[2] = uint8_t(M - 16);
            conf.rows[3] = uint8_t(M - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(M - 16);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t((N - 16) * 4);
            conf.colsb[2] = 64;
            conf.colsb[3] = uint16_t((N - 16) * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t((N - 16) * 4);
            _tile_loadconfig(&conf);

            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            for (size_t k = 0; k < K; k += 64)
            {
                _tile_stream_loadd(4, ad00 + k, (int)adStride);
                _tile_loadd(6, bd00 + k * 32, 128);
                _tile_dpbuud(0, 4, 6);
                _tile_loadd(7, bd64 + k * 32, 128);
                _tile_dpbuud(1, 4, 7);
                _tile_stream_loadd(5, ad16 + k, (int)adStride);
                _tile_dpbuud(2, 5, 6);
                _tile_dpbuud(3, 5, 7);
            }
            SIMD_ALIGNED(64) int32_t buf[32][32];
            _tile_stored(0, buf[0] + 0, 128);
            _tile_stored(1, buf[0] + F, 128);
            _tile_stored(2, buf[16] + 0, 128);
            _tile_stored(3, buf[16] + F, 128);

            __mmask16 tail = TailMask16(N - F);
            for (size_t i = 0; i < M; ++i)
            {
                DecodeCosineDistances1xF(an, bn + 0, bnStride, _mm512_loadu_si512(buf[i] + 0), distances + 0);
                DecodeCosineDistances1xF(an, bn + F, bnStride, _mm512_loadu_si512(buf[i] + F), distances + F, tail);
                an += 4, distances += stride;
            }
        }

        inline void Correlation8_32x16(size_t M, size_t N, size_t K, const uint8_t* ad00, size_t adStride, const uint8_t* bd00, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            const uint8_t* ad16 = ad00 + 16 * adStride;

            TileConf conf;
            conf.rows[0] = 16;
            conf.rows[2] = uint8_t(M - 16);
            conf.rows[4] = 16;
            conf.rows[5] = uint8_t(M - 16);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(N * 4);
            conf.colsb[2] = uint16_t(N * 4);
            conf.colsb[4] = 64;
            conf.colsb[5] = 64;
            conf.colsb[6] = uint16_t(N * 4);
            _tile_loadconfig(&conf);

            _tile_zero(0);
            _tile_zero(2);
            for (size_t k = 0; k < K; k += 64)
            {
                _tile_stream_loadd(4, ad00 + k, (int)adStride);
                _tile_loadd(6, bd00 + k * 32, 128);
                _tile_dpbuud(0, 4, 6);
                _tile_stream_loadd(5, ad16 + k, (int)adStride);
                _tile_dpbuud(2, 5, 6);
            }
            SIMD_ALIGNED(64) int32_t buf[32][16];
            _tile_stored(0, buf[0] + 0, 64);
            _tile_stored(2, buf[16] + 0, 64);

            __mmask16 tail = TailMask16(N);
            for (size_t i = 0; i < M; ++i)
            {
                DecodeCosineDistances1xF(an, bn + 0, bnStride, _mm512_loadu_si512(buf[i] + 0), distances + 0, tail);
                an += 4, distances += stride;
            }
        }

        inline void Correlation8_16x32(size_t M, size_t N, size_t K, const uint8_t* ad00, size_t adStride, const uint8_t* bd00, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            const uint8_t* bd64 = bd00 + 64;

            TileConf conf;
            conf.rows[0] = uint8_t(M);
            conf.rows[1] = uint8_t(M);
            conf.rows[4] = uint8_t(M);
            conf.rows[6] = 16;
            conf.rows[7] = 16;
            conf.colsb[0] = 64;
            conf.colsb[1] = uint16_t((N - 16) * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = 64;
            conf.colsb[7] = uint16_t((N - 16) * 4);
            _tile_loadconfig(&conf);

            _tile_zero(0);
            _tile_zero(1);
            for (size_t k = 0; k < K; k += 64)
            {
                _tile_stream_loadd(4, ad00 + k, (int)adStride);
                _tile_loadd(6, bd00 + k * 32, 128);
                _tile_dpbuud(0, 4, 6);
                _tile_loadd(7, bd64 + k * 32, 128);
                _tile_dpbuud(1, 4, 7);
            }
            SIMD_ALIGNED(64) int32_t buf[16][32];
            _tile_stored(0, buf[0] + 0, 128);
            _tile_stored(1, buf[0] + F, 128);

            __mmask16 tail = TailMask16(N - F);
            for (size_t i = 0; i < M; ++i)
            {
                DecodeCosineDistances1xF(an, bn + 0, bnStride, _mm512_loadu_si512(buf[i] + 0), distances + 0);
                DecodeCosineDistances1xF(an, bn + F, bnStride, _mm512_loadu_si512(buf[i] + F), distances + F, tail);
                an += 4, distances += stride;
            }
        }

        inline void Correlation8_16x16(size_t M, size_t N, size_t K, const uint8_t* ad00, size_t adStride, const uint8_t* bd00, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride)
        {
            TileConf conf;
            conf.rows[0] = uint8_t(M);
            conf.rows[4] = uint8_t(M);
            conf.rows[6] = 16;
            conf.colsb[0] = uint16_t(N * 4);
            conf.colsb[4] = 64;
            conf.colsb[6] = uint16_t(N * 4);
            _tile_loadconfig(&conf);

            _tile_zero(0);
            for (size_t k = 0; k < K; k += 64)
            {
                _tile_stream_loadd(4, ad00 + k, (int)adStride);
                _tile_loadd(6, bd00 + k * 32, 128);
                _tile_dpbuud(0, 4, 6);
            }
            SIMD_ALIGNED(64) int32_t buf[16][16];
            _tile_stored(0, buf[0] + 0, 64);

            __mmask16 tail = TailMask16(N);
            for (size_t i = 0; i < M; ++i)
            {
                DecodeCosineDistances1xF(an, bn + 0, bnStride, _mm512_loadu_si512(buf[i] + 0), distances + 0, tail);
                an += 4, distances += stride;
            }
        }

        typedef void(*Correlation8_Ptr)(size_t M, size_t N, size_t K, const uint8_t* ad00, size_t adStride, const uint8_t* bd00, const float* an, const float* bn, size_t bnStride, float* distances, size_t stride);

        void MacroCorrelation8(size_t M, size_t N, size_t K, const uint8_t* ad, const float* an, const uint8_t* bd, const float* bn, float* distances, size_t stride)
        {
            size_t M32 = AlignLo(M, 32), N32 = AlignLo(N, 32), K64 = AlignHi(K, 64), MT = M - M32, NT = N - N32;
            Correlation8_Ptr correlationBody = Correlation8_32x32;
            Correlation8_Ptr correlationTail = MT > 16 ? Correlation8_32x32 : Correlation8_16x32;
            size_t j = 0;
            for (; j < N32; j += 32)
            {
                size_t i = 0;
                for (; i < M32; i += 32)
                    correlationBody(32, 32, K, ad + i * K64, K64, bd, an + i * 4, bn, N, distances + i * stride, stride);
                if (i < M)
                    correlationTail(MT, 32, K, ad + i * K64, K64, bd, an + i * 4, bn, N, distances + i * stride, stride);
                bd += K64 * 32;
                bn += 32;
                distances += 32;
            }
            if (j < N)
            {
                if (NT <= 16)
                {
                    correlationBody = Correlation8_32x16;
                    correlationTail = MT > 16 ? Correlation8_32x16 : Correlation8_16x16;
                }
                size_t i = 0;
                for (; i < M32; i += 32)
                    correlationBody(32, NT, K, ad + i * K64, K64, bd, an + i * 4, bn, N, distances + i * stride, stride);
                if (i < M)
                    correlationTail(MT, NT, K, ad + i * K64, K64, bd, an + i * 4, bn, N, distances + i * stride, stride);
            }
        }

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::UnpackDataPtr GetUnpackData(size_t depth, bool transpose)
        {
            switch (depth)
            {
            case 4: return transpose ? UnpackDataB<4> : UnpackDataA<4>;
            case 5: return transpose ? UnpackDataB<5> : UnpackDataA<5>;
            case 6: return transpose ? UnpackDataB<6> : UnpackDataA<6>;
            case 7: return transpose ? UnpackDataB<7> : UnpackDataA<7>;
            case 8: return transpose ? UnpackDataB<8> : UnpackDataA<8>;
            default: return NULL;
            }
        }

        Base::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth)
        {
            return MacroCorrelation8;
        }
    }
#endif
}

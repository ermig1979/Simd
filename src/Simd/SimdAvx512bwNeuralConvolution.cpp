/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#include "Simd/SimdStream.h"
#include "Simd/SimdAvx2.h"
#include "Simd/SimdExtract.h"
#include "Simd/SimdNeural.h"

namespace Simd
{
#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
    {
        namespace Ncf
        {
            namespace Ver0
            {
                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY,
                    size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, float* dst)
                {
                    const size_t K = kernelX * kernelY * srcDepth, N = dstHeight * dstWidth;
                    if (dilationX * dilationY * strideX * strideY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow * strideY - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol * strideX - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow * dilationY;
                                        if (srcRow < srcHeight)
                                        {
                                            const float* psrc = src + (channel * srcHeight + srcRow) * srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol * dilationX;
                                                if (srcCol < srcWidth)
                                                    *(dst++) = psrc[srcCol];
                                                else
                                                    *(dst++) = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                                *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else if (kernelX * kernelY != 1)
                    {
                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                        {
                            size_t srcRow0 = dstRow - padY;
                            for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                            {
                                size_t srcCol0 = dstCol - padX;
                                for (size_t channel = 0; channel < srcDepth; ++channel)
                                {
                                    for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                    {
                                        size_t srcRow = srcRow0 + kernelRow;
                                        if (srcRow < srcHeight)
                                        {
                                            const float* psrc = src + (channel * srcHeight + srcRow) * srcWidth;
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                            {
                                                size_t srcCol = srcCol0 + kernelCol;
                                                if (srcCol < srcWidth)
                                                    *(dst++) = psrc[srcCol];
                                                else
                                                    *(dst++) = 0;
                                            }
                                        }
                                        else
                                        {
                                            for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol)
                                                *(dst++) = 0;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    else
                    {
                        for (size_t i = 0; i < N; ++i)
                        {
                            for (size_t k = 0; k < K; ++k)
                                *(dst++) = src[k * N + i];
                        }
                    }
                }

                template <bool align> static SIMD_INLINE void Kernel1x4x16(const __m512& a, size_t K, const float* b, __m512* sums)
                {
                    sums[0] = _mm512_fmadd_ps(a, Avx512f::Load<align>(b + 0 * K), sums[0]);
                    sums[1] = _mm512_fmadd_ps(a, Avx512f::Load<align>(b + 1 * K), sums[1]);
                    sums[2] = _mm512_fmadd_ps(a, Avx512f::Load<align>(b + 2 * K), sums[2]);
                    sums[3] = _mm512_fmadd_ps(a, Avx512f::Load<align>(b + 3 * K), sums[3]);
                }

                template <bool align> static SIMD_INLINE void Kernel1x1x16(const __m512& a, const float* b, __m512& sum)
                {
                    sum = _mm512_fmadd_ps(a, Avx512f::Load<align>(b), sum);
                }

                SIMD_INLINE void Add4ExtractedSums(const __m512* src, float* dst)
                {
                    __m512 sum02 = _mm512_add_ps(_mm512_unpacklo_ps(src[0], src[2]), _mm512_unpackhi_ps(src[0], src[2]));
                    __m512 sum13 = _mm512_add_ps(_mm512_unpacklo_ps(src[1], src[3]), _mm512_unpackhi_ps(src[1], src[3]));
                    __m512 sum512 = _mm512_add_ps(_mm512_unpacklo_ps(sum02, sum13), _mm512_unpackhi_ps(sum02, sum13));
                    __m128 sum128 = _mm_add_ps(_mm_add_ps(_mm512_extractf32x4_ps(sum512, 0), _mm512_extractf32x4_ps(sum512, 1)),
                        _mm_add_ps(_mm512_extractf32x4_ps(sum512, 2), _mm512_extractf32x4_ps(sum512, 3)));
                    _mm_storeu_ps(dst, _mm_add_ps(_mm_loadu_ps(dst), sum128));
                }

                template <bool align> static SIMD_INLINE void Kernel6x4x16(const __m512* a, size_t K, const float* b, __m512* sums)
                {
                    __m512 _b;
                    _b = Avx512f::Load<align>(b + 0 * K);
                    sums[0x00] = _mm512_fmadd_ps(a[0], _b, sums[0x00]);
                    sums[0x04] = _mm512_fmadd_ps(a[1], _b, sums[0x04]);
                    sums[0x08] = _mm512_fmadd_ps(a[2], _b, sums[0x08]);
                    sums[0x0C] = _mm512_fmadd_ps(a[3], _b, sums[0x0C]);
                    sums[0x10] = _mm512_fmadd_ps(a[4], _b, sums[0x10]);
                    sums[0x14] = _mm512_fmadd_ps(a[5], _b, sums[0x14]);
                    _b = Avx512f::Load<align>(b + 1 * K);
                    sums[0x01] = _mm512_fmadd_ps(a[0], _b, sums[0x01]);
                    sums[0x05] = _mm512_fmadd_ps(a[1], _b, sums[0x05]);
                    sums[0x09] = _mm512_fmadd_ps(a[2], _b, sums[0x09]);
                    sums[0x0D] = _mm512_fmadd_ps(a[3], _b, sums[0x0D]);
                    sums[0x11] = _mm512_fmadd_ps(a[4], _b, sums[0x11]);
                    sums[0x15] = _mm512_fmadd_ps(a[5], _b, sums[0x15]);
                    _b = Avx512f::Load<align>(b + 2 * K);
                    sums[0x02] = _mm512_fmadd_ps(a[0], _b, sums[0x02]);
                    sums[0x06] = _mm512_fmadd_ps(a[1], _b, sums[0x06]);
                    sums[0x0A] = _mm512_fmadd_ps(a[2], _b, sums[0x0A]);
                    sums[0x0E] = _mm512_fmadd_ps(a[3], _b, sums[0x0E]);
                    sums[0x12] = _mm512_fmadd_ps(a[4], _b, sums[0x12]);
                    sums[0x16] = _mm512_fmadd_ps(a[5], _b, sums[0x16]);
                    _b = Avx512f::Load<align>(b + 3 * K);
                    sums[0x03] = _mm512_fmadd_ps(a[0], _b, sums[0x03]);
                    sums[0x07] = _mm512_fmadd_ps(a[1], _b, sums[0x07]);
                    sums[0x0B] = _mm512_fmadd_ps(a[2], _b, sums[0x0B]);
                    sums[0x0F] = _mm512_fmadd_ps(a[3], _b, sums[0x0F]);
                    sums[0x13] = _mm512_fmadd_ps(a[4], _b, sums[0x13]);
                    sums[0x17] = _mm512_fmadd_ps(a[5], _b, sums[0x17]);
                }

                template <bool align> static SIMD_INLINE void Kernel6x1x16(const __m512* a, const float* b, __m512* sums)
                {
                    __m512 b0 = Avx512f::Load<align>(b);
                    sums[0] = _mm512_fmadd_ps(a[0], b0, sums[0]);
                    sums[1] = _mm512_fmadd_ps(a[1], b0, sums[1]);
                    sums[2] = _mm512_fmadd_ps(a[2], b0, sums[2]);
                    sums[3] = _mm512_fmadd_ps(a[3], b0, sums[3]);
                    sums[4] = _mm512_fmadd_ps(a[4], b0, sums[4]);
                    sums[5] = _mm512_fmadd_ps(a[5], b0, sums[5]);
                }

                template <bool align> static SIMD_INLINE void Kernel3x4x16(const __m512* a, size_t K, const float* b, __m512* sums)
                {
                    __m512 _b;
                    _b = Avx512f::Load<align>(b + 0 * K);
                    sums[0x0] = _mm512_fmadd_ps(a[0], _b, sums[0x0]);
                    sums[0x4] = _mm512_fmadd_ps(a[1], _b, sums[0x4]);
                    sums[0x8] = _mm512_fmadd_ps(a[2], _b, sums[0x8]);
                    _b = Avx512f::Load<align>(b + 1 * K);
                    sums[0x1] = _mm512_fmadd_ps(a[0], _b, sums[0x1]);
                    sums[0x5] = _mm512_fmadd_ps(a[1], _b, sums[0x5]);
                    sums[0x9] = _mm512_fmadd_ps(a[2], _b, sums[0x9]);
                    _b = Avx512f::Load<align>(b + 2 * K);
                    sums[0x2] = _mm512_fmadd_ps(a[0], _b, sums[0x2]);
                    sums[0x6] = _mm512_fmadd_ps(a[1], _b, sums[0x6]);
                    sums[0xA] = _mm512_fmadd_ps(a[2], _b, sums[0xA]);
                    _b = Avx512f::Load<align>(b + 3 * K);
                    sums[0x3] = _mm512_fmadd_ps(a[0], _b, sums[0x3]);
                    sums[0x7] = _mm512_fmadd_ps(a[1], _b, sums[0x7]);
                    sums[0xB] = _mm512_fmadd_ps(a[2], _b, sums[0xB]);
                }

                template <bool align> static SIMD_INLINE void Kernel3x1x16(const __m512* a, const float* b, __m512* sums)
                {
                    __m512 _b = Avx512f::Load<align>(b);
                    sums[0x0] = _mm512_fmadd_ps(a[0], _b, sums[0x0]);
                    sums[0x1] = _mm512_fmadd_ps(a[1], _b, sums[0x1]);
                    sums[0x2] = _mm512_fmadd_ps(a[2], _b, sums[0x2]);
                }

                template <bool align, bool mask> static SIMD_INLINE void Load6(const float* p, __m512* a, size_t step, __mmask16 tail = -1)
                {
                    a[0] = Avx512f::Load<align, mask>(p + 0 * step, tail);
                    a[1] = Avx512f::Load<align, mask>(p + 1 * step, tail);
                    a[2] = Avx512f::Load<align, mask>(p + 2 * step, tail);
                    a[3] = Avx512f::Load<align, mask>(p + 3 * step, tail);
                    a[4] = Avx512f::Load<align, mask>(p + 4 * step, tail);
                    a[5] = Avx512f::Load<align, mask>(p + 5 * step, tail);
                }

                template <bool align, bool mask> static SIMD_INLINE void Load3(const float* p, __m512* a, size_t step, __mmask16 tail = -1)
                {
                    a[0] = Avx512f::Load<align, mask>(p + 0 * step, tail);
                    a[1] = Avx512f::Load<align, mask>(p + 1 * step, tail);
                    a[2] = Avx512f::Load<align, mask>(p + 2 * step, tail);
                }

                template <bool align> void Execute(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    size_t M3 = M / 3 * 3;
                    size_t M6 = M / 6 * 6;
                    size_t N4 = Simd::AlignLo(N, 4);
                    size_t K16 = Simd::AlignLo(K, 16);
                    __mmask16 tailMask = TailMask16(K - K16);
                    size_t i = 0;
#if SIMD_ZMM_COUNT == 32
                    for (; i < M6; i += 6)
                    {
                        const float* pa = a + i * K;
                        float* pc = c + i * N;
                        size_t j = 0;
                        __m512 _a[6];
                        for (; j < N4; j += 4)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[24] = {
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                Load6<false, false>(pa + k, _a, K);
                                Kernel6x4x16<align>(_a, K, pb + k, sums);
                            }
                            if (k < K)
                            {
                                Load6<false, true>(pa + k, _a, K, tailMask);
                                Kernel6x4x16<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0x00, pc + 0 * N + j);
                            Add4ExtractedSums(sums + 0x04, pc + 1 * N + j);
                            Add4ExtractedSums(sums + 0x08, pc + 2 * N + j);
                            Add4ExtractedSums(sums + 0x0C, pc + 3 * N + j);
                            Add4ExtractedSums(sums + 0x10, pc + 4 * N + j);
                            Add4ExtractedSums(sums + 0x14, pc + 5 * N + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[6] = {
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                Load6<false, false>(pa + k, _a, K);
                                Kernel6x1x16<align>(_a, pb + k, sums);
                            }
                            if (k < K)
                            {
                                Load6<false, true>(pa + k, _a, K, tailMask);
                                Kernel6x1x16<false>(_a, pb + k, sums);
                            }
                            pc[0 * N + j] += Avx512f::ExtractSum(sums[0]);
                            pc[1 * N + j] += Avx512f::ExtractSum(sums[1]);
                            pc[2 * N + j] += Avx512f::ExtractSum(sums[2]);
                            pc[3 * N + j] += Avx512f::ExtractSum(sums[3]);
                            pc[4 * N + j] += Avx512f::ExtractSum(sums[4]);
                            pc[5 * N + j] += Avx512f::ExtractSum(sums[5]);
                        }
                    }
#endif
                    for (; i < M3; i += 3)
                    {
                        const float* pa = a + i * K;
                        float* pc = c + i * N;
                        size_t j = 0;
                        __m512 _a[3];
                        for (; j < N4; j += 4)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[12] = {
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                                _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                Load3<false, false>(pa + k, _a, K);
                                Kernel3x4x16<align>(_a, K, pb + k, sums);
                            }
                            if (k < K)
                            {
                                Load3<false, true>(pa + k, _a, K, tailMask);
                                Kernel3x4x16<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0x0, pc + 0 * N + j);
                            Add4ExtractedSums(sums + 0x4, pc + 1 * N + j);
                            Add4ExtractedSums(sums + 0x8, pc + 2 * N + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[3] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                Load3<false, false>(pa + k, _a, K);
                                Kernel3x1x16<align>(_a, pb + k, sums);
                            }
                            if (k < K)
                            {
                                Load3<false, true>(pa + k, _a, K, tailMask);
                                Kernel3x1x16<false>(_a, pb + k, sums);
                            }
                            pc[0 * N + j] += Avx512f::ExtractSum(sums[0]);
                            pc[1 * N + j] += Avx512f::ExtractSum(sums[1]);
                            pc[2 * N + j] += Avx512f::ExtractSum(sums[2]);
                        }
                    }
                    for (; i < M; ++i)
                    {
                        const float* pa = a + i * K;
                        float* pc = c + i * N;
                        size_t j = 0;
                        for (; j < N4; j += 4)
                        {
                            const float* pb = b + j * K;
                            __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                __m512 _a = Avx512f::Load<false>(pa + k);
                                Kernel1x4x16<align>(_a, K, pb + k, sums);
                            }
                            if (k < K)
                            {
                                __m512 _a = Avx512f::Load<false, true>(pa + k, tailMask);
                                Kernel1x4x16<false>(_a, K, pb + k, sums);
                            }
                            Add4ExtractedSums(sums + 0, pc + j);
                        }
                        for (; j < N; ++j)
                        {
                            const float* pb = b + j * K;
                            __m512 sum = _mm512_setzero_ps();
                            size_t k = 0;
                            for (; k < K16; k += 16)
                            {
                                __m512 _a = Avx512f::Load<false>(pa + k);
                                Kernel1x1x16<align>(_a, pb + k, sum);
                            }
                            if (k < K)
                            {
                                __m512 _a = Avx512f::Load<false, true>(pa + k, tailMask);
                                Kernel1x1x16<false>(_a, pb + k, sum);
                            }
                            pc[j] += Avx512f::ExtractSum(sum);
                        }
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    if (Aligned(K, F))
                        Execute<true>(M, N, K, a, b, c);
                    else
                        Execute<false>(M, N, K, a, b, c);
                }
            }

            namespace Ver1
            {
                void PrepareA(const float* src, size_t M, size_t K, size_t cell, float* dst)
                {
                    size_t K4 = AlignLo(K, 4), K8 = AlignLo(K, 8);
                    for (size_t i = 0; i < M; i += cell)
                    {
                        size_t n = Simd::Min(cell, M - i), k = 0;
                        if (cell == 4 && n == 4)
                        {
                            for (; k < K8; k += 8)
                            {
                                const float* ps = src + k;
                                __m256 s0 = Avx::Load<false>(ps + 0 * K);
                                __m256 s1 = Avx::Load<false>(ps + 1 * K);
                                __m256 s2 = Avx::Load<false>(ps + 2 * K);
                                __m256 s3 = Avx::Load<false>(ps + 3 * K);
                                __m256 s00 = _mm256_unpacklo_ps(s0, s2);
                                __m256 s01 = _mm256_unpacklo_ps(s1, s3);
                                __m256 s10 = _mm256_unpackhi_ps(s0, s2);
                                __m256 s11 = _mm256_unpackhi_ps(s1, s3);
                                __m256 d0 = _mm256_unpacklo_ps(s00, s01);
                                __m256 d1 = _mm256_unpackhi_ps(s00, s01);
                                __m256 d2 = _mm256_unpacklo_ps(s10, s11);
                                __m256 d3 = _mm256_unpackhi_ps(s10, s11);
                                Avx::Store<false>(dst + 0, _mm256_permute2f128_ps(d0, d1, 0x20));
                                Avx::Store<false>(dst + 8, _mm256_permute2f128_ps(d2, d3, 0x20));
                                Avx::Store<false>(dst + 16, _mm256_permute2f128_ps(d0, d1, 0x31));
                                Avx::Store<false>(dst + 24, _mm256_permute2f128_ps(d2, d3, 0x31));
                                dst += 32;
                            }
                            for (; k < K4; k += 4)
                            {
                                const float* ps = src + k;
                                __m128 s0 = Sse2::Load<false>(ps + 0 * K);
                                __m128 s1 = Sse2::Load<false>(ps + 1 * K);
                                __m128 s2 = Sse2::Load<false>(ps + 2 * K);
                                __m128 s3 = Sse2::Load<false>(ps + 3 * K);
                                __m128 s00 = _mm_unpacklo_ps(s0, s2);
                                __m128 s01 = _mm_unpacklo_ps(s1, s3);
                                __m128 s10 = _mm_unpackhi_ps(s0, s2);
                                __m128 s11 = _mm_unpackhi_ps(s1, s3);
                                Sse2::Store<false>(dst + 0, _mm_unpacklo_ps(s00, s01));
                                Sse2::Store<false>(dst + 4, _mm_unpackhi_ps(s00, s01));
                                Sse2::Store<false>(dst + 8, _mm_unpacklo_ps(s10, s11));
                                Sse2::Store<false>(dst + 12, _mm_unpackhi_ps(s10, s11));
                                dst += 16;
                            }
                        }
                        for (; k < K; ++k)
                        {
                            for (size_t c = 0; c < n; ++c)
                                *(dst++) = src[c * K + k];
                        }
                        src += cell * K;
                    }
                }

                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY,
                    size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t cell, float* tmp, float* dst)
                {
                    const size_t K = kernelX * kernelY * srcDepth, N = dstHeight * dstWidth;
                    if (kernelX * kernelY != 1)
                    {
                        float* dst = tmp;
                        size_t channelSize = srcHeight * srcWidth;
                        if (dilationX * dilationY * strideX * strideY != 1)
                        {
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow * dilationY - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol * dilationX - padX;
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = src[srcRow * srcWidth + srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                    srcCol += strideX;
                                                }
                                            }
                                            else
                                            {
                                                for (size_t dstCol = 0; dstCol < dstWidth; ++dstCol)
                                                    *(dst++) = 0;
                                            }
                                            srcRow += strideY;
                                        }
                                    }
                                }
                            }
                        }
                        else
                        {
                            const size_t bodySize = dstWidth - padX * 2;
                            for (size_t channel = 0, k = 0; channel < srcDepth; ++channel, src += channelSize)
                            {
                                for (size_t kernelRow = 0; kernelRow < kernelY; ++kernelRow)
                                {
                                    for (size_t kernelCol = 0; kernelCol < kernelX; ++kernelCol, ++k)
                                    {
                                        size_t srcRow = kernelRow - padY;
                                        for (size_t dstRow = 0; dstRow < dstHeight; ++dstRow, ++srcRow)
                                        {
                                            if (srcRow < srcHeight)
                                            {
                                                size_t srcCol = kernelCol - padX, dstCol = 0;
                                                const float* psrc = src + srcRow * srcWidth;
                                                for (; dstCol < padX; ++dstCol, ++srcCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = psrc[srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                }
                                                memcpy(dst, psrc + srcCol, bodySize * 4);
                                                dst += bodySize;
                                                dstCol += bodySize;
                                                srcCol += bodySize;
                                                for (; dstCol < dstWidth; ++dstCol, ++srcCol)
                                                {
                                                    if (srcCol < srcWidth)
                                                        *(dst++) = psrc[srcCol];
                                                    else
                                                        *(dst++) = 0;
                                                }
                                            }
                                            else
                                            {
                                                memset(dst, 0, dstWidth * 4);
                                                dst += dstWidth;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        src = tmp;
                    }
                    if (cell == 48)
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            if (n == cell)
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
                                    Store<false>(dst + 0 * F, Load<false>(psrc + 0 * F));
                                    Store<false>(dst + 1 * F, Load<false>(psrc + 1 * F));
                                    Store<false>(dst + 2 * F, Load<false>(psrc + 2 * F));
                                    dst += 48;
                                }
                            }
                            else
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
                                    size_t c = 0;
                                    for (; c < n; ++c)
                                        *(dst++) = *(psrc++);
                                    for (; c < cell; ++c)
                                        *(dst++) = 0;
                                }
                            }
                            src += cell;
                        }
                    }
                    else if (cell == 16)
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            if (n == cell)
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
                                    Store<false>(dst, Load<false>(psrc));
                                    dst += 16;
                                }
                            }
                            else
                            {
                                for (size_t k = 0; k < K; ++k)
                                {
                                    const float* psrc = src + k * N;
                                    size_t c = 0;
                                    for (; c < n; ++c)
                                        *(dst++) = *(psrc++);
                                    for (; c < cell; ++c)
                                        *(dst++) = 0;
                                }
                            }
                            src += cell;
                        }
                    }
                    else
                    {
                        for (size_t j = 0; j < N; j += cell)
                        {
                            size_t n = Simd::Min(cell, N - j);
                            for (size_t k = 0; k < K; ++k)
                            {
                                const float* psrc = src + k * N;
                                size_t c = 0;
                                for (; c < n; ++c)
                                    *(dst++) = *(psrc++);
                                for (; c < cell; ++c)
                                    *(dst++) = 0;
                            }
                            src += cell;
                        }
                    }
                }

                SIMD_INLINE void AddSum(__m512 sum, float* dst)
                {
                    _mm512_storeu_ps(dst, _mm512_add_ps(_mm512_loadu_ps(dst), sum));
                }

                template<bool mask> SIMD_INLINE void AddSum(__m512 sum, float* dst, __mmask16 tail = -1)
                {
                    Avx512f::Store<false, mask>(dst, _mm512_add_ps((Avx512f::Load<false, mask>(dst, tail)), sum), tail);
                }

                template<bool mask> SIMD_INLINE void AddSums16(const __m512* sums, size_t size, float* dst, size_t stride, __mmask16 tail = -1)
                {
                    for (size_t i = 0; i < size; ++i, dst += stride)
                        AddSum<mask>(sums[i], dst, tail);
                }

                template <bool align, bool mask> SIMD_INLINE void KernelMx16(size_t N, size_t K, const float* a, const float* b, float* c, size_t m, __mmask16 tail = -1)
                {
                    __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m512 b0 = Avx512f::Load<align>(b);
                        for (size_t s = 0; s < m; ++s)
                        {
                            __m512 a0 = _mm512_set1_ps(a[s]);
                            sums[s] = _mm512_fmadd_ps(b0, a0, sums[s]);
                        }
                        b += 16;
                        a += m;
                    }
                    AddSums16<mask>(sums, m, c, N, tail);
                }

                template <bool align, bool mask> SIMD_INLINE void Kernel4x16(size_t N, size_t K, const float* a, const float* b, float* c, __mmask16 tail = -1)
                {
                    __m512 sums[4] = { _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m512 b0 = Avx512f::Load<align>(b);
                        __m512 a0 = _mm512_set1_ps(a[0]);
                        sums[0] = _mm512_fmadd_ps(b0, a0, sums[0]);
                        __m512 a1 = _mm512_set1_ps(a[1]);
                        sums[1] = _mm512_fmadd_ps(b0, a1, sums[1]);
                        __m512 a2 = _mm512_set1_ps(a[2]);
                        sums[2] = _mm512_fmadd_ps(b0, a2, sums[2]);
                        __m512 a3 = _mm512_set1_ps(a[3]);
                        sums[3] = _mm512_fmadd_ps(b0, a3, sums[3]);
                        b += 16;
                        a += 4;
                    }
                    AddSums16<mask>(sums, 4, c, N, tail);
                }

                template <bool align> void Execute4x16(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N16 = Simd::AlignLo(N, 16);
                    __mmask16 tailMask = TailMask16(N - N16);
                    size_t i = 0;
                    for (; i < M4; i += 4)
                    {
                        size_t j = 0;
                        for (; j < N16; j += 16)
                            Kernel4x16<align, false>(N, K, a + i * K, b + j * K, c + i * N + j);
                        if (j < N)
                            Kernel4x16<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, tailMask);
                    }
                    if (i < M)
                    {
                        size_t j = 0;
                        for (; j < N16; j += 16)
                            KernelMx16<align, false>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4);
                        if (j < N)
                            KernelMx16<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMask);
                    }
                }

                template <bool align, bool mask> SIMD_INLINE void KernelMx48(size_t N, size_t K, const float* a, const float* b, float* c, size_t m, const __mmask16* tails)
                {
                    __m512 sums[12] = {
                        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(),
                        _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps() };
                    for (size_t k = 0; k < K; ++k)
                    {
                        __m512 b0 = Avx512f::Load<align>(b + 00);
                        __m512 b1 = Avx512f::Load<align>(b + 16);
                        __m512 b2 = Avx512f::Load<align>(b + 32);
                        for (size_t s = 0; s < m; ++s)
                        {
                            __m512 a0 = _mm512_set1_ps(a[s]);
                            sums[s + 0] = _mm512_fmadd_ps(b0, a0, sums[s + 0]);
                            sums[s + 4] = _mm512_fmadd_ps(b1, a0, sums[s + 4]);
                            sums[s + 8] = _mm512_fmadd_ps(b2, a0, sums[s + 8]);
                        }
                        b += 48;
                        a += m;
                    }
                    for (size_t i = 0; i < m; ++i, c += N)
                    {
                        AddSum<mask>(sums[i + 0], c + 00, tails[0]);
                        AddSum<mask>(sums[i + 4], c + 16, tails[1]);
                        AddSum<mask>(sums[i + 8], c + 32, tails[2]);
                    }
                }

                void Kernel4x48(size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    __m512 _a, b0, b1, b2, c00, c01, c02, c10, c11, c12, c20, c21, c22, c30, c31, c32;

                    c00 = _mm512_setzero_ps();
                    c01 = _mm512_setzero_ps();
                    c02 = _mm512_setzero_ps();
                    c10 = _mm512_setzero_ps();
                    c11 = _mm512_setzero_ps();
                    c12 = _mm512_setzero_ps();
                    c20 = _mm512_setzero_ps();
                    c21 = _mm512_setzero_ps();
                    c22 = _mm512_setzero_ps();
                    c30 = _mm512_setzero_ps();
                    c31 = _mm512_setzero_ps();
                    c32 = _mm512_setzero_ps();

                    for (size_t k = 0; k < K; ++k)
                    {
                        b0 = _mm512_loadu_ps(b + 0 * F);
                        b1 = _mm512_loadu_ps(b + 1 * F);
                        b2 = _mm512_loadu_ps(b + 2 * F);
                        _a = _mm512_set1_ps(a[0]);
                        c00 = _mm512_fmadd_ps(b0, _a, c00);
                        c01 = _mm512_fmadd_ps(b1, _a, c01);
                        c02 = _mm512_fmadd_ps(b2, _a, c02);
                        _a = _mm512_set1_ps(a[1]);
                        c10 = _mm512_fmadd_ps(b0, _a, c10);
                        c11 = _mm512_fmadd_ps(b1, _a, c11);
                        c12 = _mm512_fmadd_ps(b2, _a, c12);
                        _a = _mm512_set1_ps(a[2]);
                        c20 = _mm512_fmadd_ps(b0, _a, c20);
                        c21 = _mm512_fmadd_ps(b1, _a, c21);
                        c22 = _mm512_fmadd_ps(b2, _a, c22);
                        _a = _mm512_set1_ps(a[3]);
                        c30 = _mm512_fmadd_ps(b0, _a, c30);
                        c31 = _mm512_fmadd_ps(b1, _a, c31);
                        c32 = _mm512_fmadd_ps(b2, _a, c32);
                        b += 48;
                        a += 4;
                    }

                    AddSum(c00, c + 0 * F);
                    AddSum(c01, c + 1 * F);
                    AddSum(c02, c + 2 * F);
                    c += N;
                    AddSum(c10, c + 0 * F);
                    AddSum(c11, c + 1 * F);
                    AddSum(c12, c + 2 * F);
                    c += N;
                    AddSum(c20, c + 0 * F);
                    AddSum(c21, c + 1 * F);
                    AddSum(c22, c + 2 * F);
                    c += N;
                    AddSum(c30, c + 0 * F);
                    AddSum(c31, c + 1 * F);
                    AddSum(c32, c + 2 * F);
                }

                template <bool align> void Execute4x48(size_t M, size_t N, size_t K, const float* a, const float* b, float* c)
                {
                    size_t M4 = Simd::AlignLo(M, 4);
                    size_t N48 = N / 48 * 48;
                    __mmask16 tailMasks[3];
                    for (size_t i = 0; i < 3; ++i)
                        tailMasks[i] = TailMask16(N - N48 - F * i);
                    if (M > N)
                    {
                        size_t i = 0;
                        for (; i < M4; i += 4)
                        {
                            size_t j = 0;
                            for (; j < N48; j += 48)
                                Kernel4x48(N, K, a + i * K, b + j * K, c + i * N + j);
                            if (j < N)
                                KernelMx48<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, 4, tailMasks);
                        }
                        if (i < M)
                        {
                            size_t j = 0;
                            for (; j < N48; j += 48)
                                KernelMx48<align, false>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMasks);
                            if (j < N)
                                KernelMx48<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMasks);
                        }
                    }
                    else
                    {
                        size_t j = 0;
                        for (; j < N48; j += 48)
                        {
                            size_t i = 0;
                            for (; i < M4; i += 4)
                                Kernel4x48(N, K, a + i * K, b + j * K, c + i * N + j);
                            if (M4 < M)
                                KernelMx48<align, false>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMasks);
                        }
                        if (N48 < N)
                        {
                            size_t i = 0;
                            for (; i < M4; i += 4)
                                KernelMx48<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, 4, tailMasks);
                            if (M4 < M)
                                KernelMx48<align, true>(N, K, a + i * K, b + j * K, c + i * N + j, M - M4, tailMasks);
                        }
                    }
                }

                void Execute(size_t M, size_t N, size_t K, const float* a, const float* b, float* c, size_t cellA, size_t cellB)
                {
                    if (cellA == 4)
                    {
                        if (cellB == 16)
                            Execute4x16<false>(M, N, K, a, b, c);
                        if (cellB == 48)
                            Execute4x48<false>(M, N, K, a, b, c);
                    }
                }
            }

            namespace Ver2
            {
                void PrepareB(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t padX, size_t padY, float* dst, size_t dstWidth, size_t dstHeight)
                {
                    for (size_t channel = 0; channel < srcDepth; ++channel)
                    {
                        const float* s = src;
                        float* d = dst;
                        memset(d, 0, padY * dstWidth * 4);
                        d += padY * dstWidth;
                        for (size_t row = padY; row < dstHeight - padY; ++row)
                        {
                            memset(d, 0, padX * 4);
                            memcpy(d + padX, s, srcWidth * 4);
                            memset(d + padX + srcWidth, 0, padX * 4);
                            d += dstWidth;
                            s += srcWidth;
                        }
                        memset(d, 0, padY * dstWidth * 4);
                        src += srcWidth * srcHeight;
                        dst += dstWidth * dstHeight;
                    }
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution8x8(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstDepth)
                {
                    __m256 _weight[kernelX * kernelY];
                    for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                    {
                        __m256 _dst[8];
                        float* pdst = dst;
                        for (size_t row = 0; row < 8; ++row, pdst += 8)
                            _dst[row] = Avx::Load<align>(pdst);
                        if (kernelY < 4)
                        {
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float* psrc = src + srcWidth * srcHeight * srcChannel;
                                Avx2::LoadWeightsForward<kernelX* kernelY>(weight, _weight);
                                for (size_t row = 0; row < 8; ++row)
                                {
                                    _dst[row] = _mm256_add_ps(_dst[row], Avx2::Convolution<kernelX, kernelY>::template Forward<align>(psrc, srcWidth, _weight));
                                    psrc += srcWidth;
                                }
                                weight += kernelX * kernelY;
                            }
                        }
                        else
                        {
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float* psrc = src + srcWidth * srcHeight * srcChannel;
                                for (size_t dy = 0; dy < kernelY; dy++)
                                {
                                    const float* ps = psrc + dy * srcWidth;
                                    Avx2::LoadWeightsForward<kernelX>(weight, _weight);
                                    for (size_t row = 0; row < 8; ++row)
                                    {
                                        _dst[row] = _mm256_add_ps(_dst[row], Avx2::Convolution<kernelX, kernelY>::template RowConvolution<align>(ps, _weight));
                                        ps += srcWidth;
                                    }
                                    weight += kernelX;
                                }
                            }
                        }
                        for (size_t row = 0; row < 8; ++row, dst += 8)
                            Avx::Store<align>(dst, _dst[row]);
                    }
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution16x16(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstDepth)
                {
                    __m512 _weight[kernelX * kernelY];
                    for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                    {
                        __m512 _dst[16];
                        float* pdst = dst;
                        for (size_t row = 0; row < 16; ++row, pdst += 16)
                            _dst[row] = Avx512f::Load<align>(pdst);
                        if (kernelY < 4)
                        {
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float* psrc = src + srcWidth * srcHeight * srcChannel;
                                LoadWeightsForward<kernelX* kernelY>(weight, _weight);
                                for (size_t row = 0; row < 16; ++row)
                                {
                                    _dst[row] = _mm512_add_ps(_dst[row], (Convolution<kernelX, kernelY>::template Forward<align, false>(psrc, srcWidth, _weight)));
                                    psrc += srcWidth;
                                }
                                weight += kernelX * kernelY;
                            }
                        }
                        else
                        {
                            for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                            {
                                const float* psrc = src + srcWidth * srcHeight * srcChannel;
                                for (size_t dy = 0; dy < kernelY; dy++)
                                {
                                    const float* ps = psrc + dy * srcWidth;
                                    LoadWeightsForward<kernelX>(weight, _weight);
                                    for (size_t row = 0; row < 16; ++row)
                                    {
                                        _dst[row] = _mm512_add_ps(_dst[row], (Convolution<kernelX, kernelY>::template RowConvolution<align, false>(ps, _weight)));
                                        ps += srcWidth;
                                    }
                                    weight += kernelX;
                                }
                            }
                        }
                        for (size_t row = 0; row < 16; ++row, dst += 16)
                            Avx512f::Store<align>(dst, _dst[row]);
                    }
                }

                template <bool align, size_t kernelX, size_t kernelY> void AddConvolution(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    if (dstWidth == 8 && dstHeight == 8)
                    {
                        AddConvolution8x8<align, kernelX, kernelY>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstDepth);
                        return;
                    }
                    if (dstWidth == 16 && dstHeight == 16)
                    {
                        AddConvolution16x16<align, kernelX, kernelY>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstDepth);
                        return;
                    }
                    size_t alignedWidth = AlignLo(dstWidth, F);
                    __mmask16 tailMask = TailMask16(dstWidth - alignedWidth);
                    __m512 _weight[kernelX * kernelY];
                    for (size_t dstChannel = 0; dstChannel < dstDepth; ++dstChannel)
                    {
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            const float* psrc = src + srcWidth * srcHeight * srcChannel;
                            const float* pweight = weight + (dstChannel * srcDepth + srcChannel) * kernelX * kernelY;
                            float* pdst = dst + dstWidth * dstHeight * dstChannel;
                            Avx512f::LoadWeightsForward<kernelX* kernelY>(pweight, _weight);
                            for (size_t row = 0; row < dstHeight; ++row)
                            {
                                size_t col = 0;
                                for (; col < alignedWidth; col += F)
                                {
                                    __m512 _dst = Avx512f::Load<align>(pdst + col);
                                    _dst = _mm512_add_ps(_dst, (Convolution<kernelX, kernelY>::template Forward<align, false>(psrc + col, srcWidth, _weight)));
                                    Avx512f::Store<align>(pdst + col, _dst);
                                }
                                if (col < dstWidth)
                                {
                                    __m512 _dst = Avx512f::Load<align, true>(pdst + col, tailMask);
                                    _dst = _mm512_add_ps(_dst, (Convolution<kernelX, kernelY>::template Forward<align, true>(psrc + col, srcWidth, _weight, tailMask)));
                                    Avx512f::Store<align, true>(pdst + col, _dst, tailMask);
                                }
                                psrc += srcWidth;
                                pdst += dstWidth;
                            }
                        }
                    }
                }

                void AddConvolution1x1x16(const float* src, size_t srcDepth, const float* weight, float* dst, size_t dstDepth)
                {
                    size_t dstDepth4 = dstDepth / 4 * 4;
                    size_t dstChannel = 0;
                    for (; dstChannel < dstDepth4; dstChannel += 4)
                    {
                        __m512 dst00 = _mm512_loadu_ps(dst + 0 * F);
                        __m512 dst10 = _mm512_loadu_ps(dst + 1 * F);
                        __m512 dst20 = _mm512_loadu_ps(dst + 2 * F);
                        __m512 dst30 = _mm512_loadu_ps(dst + 3 * F);
                        const float* psrc = src;
                        const float* pw0 = weight;
                        const float* pw1 = pw0 + srcDepth;
                        const float* pw2 = pw1 + srcDepth;
                        const float* pw3 = pw2 + srcDepth;
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            __m512 _weight;
                            __m512 src0 = _mm512_loadu_ps(psrc + 0 * F);
                            _weight = _mm512_set1_ps(pw0[srcChannel]);
                            dst00 = _mm512_fmadd_ps(_weight, src0, dst00);
                            _weight = _mm512_set1_ps(pw1[srcChannel]);
                            dst10 = _mm512_fmadd_ps(_weight, src0, dst10);
                            _weight = _mm512_set1_ps(pw2[srcChannel]);
                            dst20 = _mm512_fmadd_ps(_weight, src0, dst20);
                            _weight = _mm512_set1_ps(pw3[srcChannel]);
                            dst30 = _mm512_fmadd_ps(_weight, src0, dst30);
                            psrc += 16;
                        }
                        _mm512_storeu_ps(dst + 0 * F, dst00);
                        _mm512_storeu_ps(dst + 1 * F, dst10);
                        _mm512_storeu_ps(dst + 2 * F, dst20);
                        _mm512_storeu_ps(dst + 3 * F, dst30);
                        dst += 16 * 4;
                        weight += srcDepth * 4;
                    }
                    for (; dstChannel < dstDepth; ++dstChannel)
                    {
                        __m512 dst0 = _mm512_loadu_ps(dst + 0 * F);
                        const float* psrc = src;
                        for (size_t srcChannel = 0; srcChannel < srcDepth; ++srcChannel)
                        {
                            __m512 weight0 = _mm512_set1_ps(*weight++);
                            dst0 = _mm512_fmadd_ps(weight0, _mm512_loadu_ps(psrc + 0 * F), dst0);
                            psrc += 16;
                        }
                        _mm512_storeu_ps(dst + 0 * F, dst0);
                        dst += 16;
                    }
                }

                void Execute(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
                    const float* weight, size_t kernelX, size_t kernelY, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    assert(kernelX == kernelY);
                    if (kernelX == 1 && dstWidth * dstHeight == 16)
                        AddConvolution1x1x16(src, srcDepth, weight, dst, dstDepth);
                    else if (kernelX == 2)
                        AddConvolution<false, 2, 2>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 3)
                        AddConvolution<false, 3, 3>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 4)
                        AddConvolution<false, 4, 4>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else if (kernelX == 5)
                        AddConvolution<false, 5, 5>(src, srcWidth, srcHeight, srcDepth, weight, dst, dstWidth, dstHeight, dstDepth);
                    else
                        assert(0);
                }

                bool Preferable(size_t srcDepth, size_t kernelX, size_t kernelY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    if (kernelX == kernelY && strideX * strideY * dilationX * dilationY == 1)
                    {
                        if (kernelX >= 2 && kernelX <= 5)// && dstWidth*dstHeight*kernelX*kernelY >= 8 * 8 * 3 * 3)
                            return true;
                        if (kernelX == 1 && (dstWidth * dstHeight == 16))// || dstWidth * dstHeight == 64))
                            return true;
                    }
                    return false;
                }
            }

            struct Opt
            {
                enum Alg
                {
                    None,
                    Ver0,
                    Ver1,
                    Ver2,
                } alg;

                size_t sizeA;
                size_t sizeB;
                size_t sizeT;

                size_t cellA;
                size_t cellB;

                size_t M, N, K;
                size_t strideB;
                size_t paddedW;
                size_t paddedH;

                Opt(size_t srcWidth, size_t srcHeight, size_t srcDepth, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY, size_t dstWidth, size_t dstHeight, size_t dstDepth)
                {
                    alg = None;
                    sizeA = 0;
                    sizeB = 0;
                    sizeT = 0;
                    cellA = 1;
                    cellB = 1;

                    M = dstDepth;
                    N = dstHeight * dstWidth;
                    K = kernelX * kernelY * srcDepth;

                    if (dstWidth * dstHeight / kernelX <= 1000)
                        alg = Ver0;
                    else
                        alg = Ver1;
                    if (Ver2::Preferable(srcDepth, kernelX, kernelY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth))
                        alg = Ver2;

                    switch (alg)
                    {
                    case Ver0:
                        sizeB = N * K;
                        break;
                    case Ver1:
                        cellA = 4;
                        cellB = 48;
                        sizeA = M * K;
                        strideB = (N + cellB - 1) / cellB * cellB;
                        sizeB = strideB * K;
                        if (kernelX * kernelY > 1)
                            sizeT = sizeB;
                        break;
                    case Ver2:
                        if (padX > 0 || padY > 0)
                        {
                            paddedW = Simd::AlignHi(srcWidth + 2 * padX, F);
                            paddedH = srcHeight + 2 * padY;
                            sizeB = paddedW * paddedH * srcDepth;
                        }
                        else
                        {
                            paddedW = srcWidth;
                            paddedH = srcHeight;
                        }
                        break;
                    default:
                        assert(0);
                        break;
                    }
                }
            };

            struct Data
            {
                float* a;
                float* b;
                float* t;

                Data(size_t sizeA, size_t sizeB, size_t sizeT, void* externalData, size_t* externalSize)
                    : a(0)
                    , b(0)
                    , _data(0)
                {
                    sizeA = AlignHi(sizeA, F);
                    sizeB = AlignHi(sizeB, F);
                    sizeT = AlignHi(sizeT, F);
                    size_t size = (sizeA + sizeB + sizeT) * sizeof(float);
                    if (size == 0)
                        return;
                    if (externalData != AlignHi(externalData, SIMD_ALIGN))
                        size += SIMD_ALIGN;
                    float* data = NULL;
                    if (externalData == NULL || externalSize == NULL || *externalSize < size)
                    {
                        _data = Simd::Allocate(size);
                        if (externalSize)
                            *externalSize = size;
                        data = (float*)_data;
                    }
                    else
                        data = (float*)AlignHi(externalData, SIMD_ALIGN);
                    if (sizeA)
                        a = data;
                    if (sizeB)
                        b = data + sizeA;
                    if (sizeT)
                        t = data + sizeA + sizeB;
                }

                ~Data()
                {
                    if (_data)
                        Simd::Free(_data);
                }

            private:
                void* _data;
            };
        }

        void NeuralConvolutionForward(const float* src, size_t srcWidth, size_t srcHeight, size_t srcDepth,
            const float* weight, size_t kernelX, size_t kernelY, size_t padX, size_t padY, size_t strideX, size_t strideY, size_t dilationX, size_t dilationY,
            void* buffer, size_t* size, float* dst, size_t dstWidth, size_t dstHeight, size_t dstDepth, int add)
        {
            using namespace Ncf;

            assert(dstWidth == (srcWidth + 2 * padX - (dilationX * (kernelX - 1) + 1)) / strideX + 1);
            assert(dstHeight == (srcHeight + 2 * padY - (dilationY * (kernelY - 1) + 1)) / strideY + 1);

            if (dstWidth < F && srcDepth <= 32)
            {
                Avx2::NeuralConvolutionForward(src, srcWidth, srcHeight, srcDepth, weight, kernelX, kernelY, padX, padY,
                    strideX, strideY, dilationX, dilationY, buffer, size, dst, dstWidth, dstHeight, dstDepth, add);
                return;
            }

            if (!add)
                memset(dst, 0, dstWidth * dstHeight * dstDepth * sizeof(float));

            Opt opt(srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, dstDepth);

            Data data(opt.sizeA, opt.sizeB, opt.sizeT, buffer, size);

            if (opt.sizeA)
            {
                switch (opt.alg)
                {
                case Opt::Ver1: Ver1::PrepareA(weight, opt.M, opt.K, opt.cellA, data.a);
                default:
                    break;
                }
            }
            else
                data.a = (float*)weight;

            if (opt.sizeB)
            {
                switch (opt.alg)
                {
                case Opt::Ver0: Ver0::PrepareB(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, data.b); break;
                case Opt::Ver1: Ver1::PrepareB(src, srcWidth, srcHeight, srcDepth, kernelX, kernelY, padX, padY, strideX, strideY, dilationX, dilationY, dstWidth, dstHeight, opt.cellB, data.t, data.b); break;
                case Opt::Ver2: Ver2::PrepareB(src, srcWidth, srcHeight, srcDepth, padX, padY, data.b, opt.paddedW, opt.paddedH); break;
                default: break;
                }
            }
            else
                data.b = (float*)src;

            switch (opt.alg)
            {
            case Opt::Ver0: Ver0::Execute(opt.M, opt.N, opt.K, data.a, data.b, dst); break;
            case Opt::Ver1: Ver1::Execute(opt.M, opt.N, opt.K, data.a, data.b, dst, opt.cellA, opt.cellB); break;
            case Opt::Ver2: Ver2::Execute(data.b, opt.paddedW, opt.paddedH, srcDepth, weight, kernelX, kernelY, dst, dstWidth, dstHeight, dstDepth); break;
            default: break;
            }
        }
    }
#endif
}

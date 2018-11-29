/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2018 Yermalayeu Ihar.
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
#include "Simd/SimdConvolution.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdAvx512f.h"

namespace Simd
{
#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        static void BiasAndActivation(const float * bias, size_t count, size_t size, ::SimdConvolutionActivationType type, const float * params, float * dst)
        {
            size_t aligned = AlignLo(size, F);
            __mmask16 tail = __mmask16(-1) >> (F + aligned - size);
            if (type == ::SimdConvolutionActivationIdentity)
            {
                if (bias)
                    SynetAddBias(bias, count, size, dst, SimdFalse);
            }
            else if (type == ::SimdConvolutionActivationRelu)
            {
                if (bias)
                {
                    __m512 _0 = _mm512_set1_ps(0.0f);
                    for (size_t i = 0; i < count; ++i)
                    {
                        __m512 _shift = _mm512_set1_ps(bias[i]);
                        size_t j = 0;
                        for (; j < aligned; j += F)
                        {
                            __m512 _dst = _mm512_loadu_ps(dst + j);
                            _mm512_storeu_ps(dst + j, _mm512_max_ps(_0, _mm512_add_ps(_dst, _shift)));
                        }
                        if (j < size)
                        {
                            __m512 _dst = _mm512_maskz_loadu_ps(tail, dst + j);
                            _mm512_mask_storeu_ps(dst + j, tail, _mm512_max_ps(_0, _mm512_add_ps(_dst, _shift)));
                        }
                        dst += size;
                    }
                }
                else
                {
                    float slope = 0;
                    NeuralRelu(dst, size*count, &slope, dst);
                }
            }
            else if (type == ::SimdConvolutionActivationLeakyRelu)
            {
                float slope = params[0];
                if (bias)
                {
                    __m512 _0 = _mm512_set1_ps(0.0f);
                    __m512 _slope = _mm512_set1_ps(slope);
                    for (size_t i = 0; i < count; ++i)
                    {
                        __m512 _shift = _mm512_set1_ps(bias[i]);
                        size_t j = 0;
                        for (; j < aligned; j += F)
                        {
                            __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _shift);
                            _mm512_storeu_ps(dst + j, _mm512_add_ps(_mm512_max_ps(_0, value), _mm512_mul_ps(_slope, _mm512_min_ps(_0, value))));
                        }
                        if (j < size)
                        {
                            __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _shift);
                            _mm512_mask_storeu_ps(dst + j, tail, _mm512_add_ps(_mm512_max_ps(_0, value), _mm512_mul_ps(_slope, _mm512_min_ps(_0, value))));
                        }
                        dst += size;
                    }
                }
                else
                    NeuralRelu(dst, size*count, &slope, dst);
            }
            else if (type == ::SimdConvolutionActivationRestrictRange)
            {
                float lower = params[0];
                float upper = params[1];
                if (bias)
                {
                    __m512 _lower = _mm512_set1_ps(lower);
                    __m512 _upper = _mm512_set1_ps(upper);
                    for (size_t i = 0; i < count; ++i)
                    {
                        __m512 _shift = _mm512_set1_ps(bias[i]);
                        size_t j = 0;
                        for (; j < aligned; j += F)
                        {
                            __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _shift);
                            _mm512_storeu_ps(dst + j, _mm512_min_ps(_mm512_max_ps(_lower, value), _upper));
                        }
                        if (j < size)
                        {
                            __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _shift);
                            _mm512_mask_storeu_ps(dst + j, tail, _mm512_min_ps(_mm512_max_ps(_lower, value), _upper));
                        }
                        dst += size;
                    }
                }
                else
                    SynetRestrictRange(dst, size*count, &lower, &upper, dst);
            }
            else if (type == ::SimdConvolutionActivationPrelu)
            {
                if (bias)
                {
                    __m512 _0 = _mm512_set1_ps(0.0f);
                    for (size_t i = 0; i < count; ++i)
                    {
                        __m512 _shift = _mm512_set1_ps(bias[i]);
                        __m512 _slope = _mm512_set1_ps(params[i]);
                        size_t j = 0;
                        for (; j < aligned; j += F)
                        {
                            __m512 value = _mm512_add_ps(_mm512_loadu_ps(dst + j), _shift);
                            _mm512_storeu_ps(dst + j, _mm512_add_ps(_mm512_max_ps(_0, value), _mm512_mul_ps(_slope, _mm512_min_ps(_0, value))));
                        }
                        if (j < size)
                        {
                            __m512 value = _mm512_add_ps(_mm512_maskz_loadu_ps(tail, dst + j), _shift);
                            _mm512_mask_storeu_ps(dst + j, tail, _mm512_add_ps(_mm512_max_ps(_0, value), _mm512_mul_ps(_slope, _mm512_min_ps(_0, value))));
                        }
                        dst += size;
                    }
                }
                else
                {
                    for (size_t i = 0; i < count; ++i)
                        NeuralRelu(dst + i*size, size, params + i, dst + i*size);
                }
            }
        }

        //---------------------------------------------------------------------

        ConvolutionImgToCol::ConvolutionImgToCol(const ConvParam & p)
            : Avx2::ConvolutionImgToCol(p)
        {
            _index.Resize(F);
            for (size_t i = 0; i < F; ++i)
                _index[i] = int(i * p.strideX);
            _nose.Resize(p.kernelX);
            _tail.Resize(p.kernelX);
            ptrdiff_t aligned = AlignHi(p.dstW, F) - F;
            for (size_t kx = 0; kx < p.kernelX; ++kx)
            {
                _nose[kx] = 0;
                _tail[kx] = 0;
                ptrdiff_t sx = kx * p.dilationX - p.padX;
                for (size_t dx = 0; dx < p.dstW; ++dx)
                {
                    if (sx >= 0 && sx < ptrdiff_t(p.srcW) && dx < F)
                        _nose[kx] |= 1 << dx;
                    if (sx < ptrdiff_t(p.srcW) && ptrdiff_t(dx) >= aligned)
                        _tail[kx] |= 1 << (dx - aligned);
                    sx += p.strideX;
                }
            }
        }

        void ConvolutionImgToCol::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Avx512f::Gemm32fNN(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _N, &_0, dst + _dstStep * g, _N);
            Avx512f::BiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        void ConvolutionImgToCol::ImgToCol(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            size_t srcSize = p.srcW * p.srcH;
            if (p.dilationX == 1 && p.dilationY == 1 && p.strideX == 2 && p.strideY == 2 && p.padX == 0 && p.padY == 0 && p.padW == 0 && p.padH == 0 && p.kernelX == 1 && p.kernelY == 1)
            {
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t dy = 0; dy < p.dstH; ++dy)
                    {
                        const float * psrc = src + 2 * dy*p.srcW;
                        for (size_t dx = 0, sx = 0; dx < p.dstW; ++dx, sx += 2)
                            *(dst++) = psrc[sx];
                    }
                    src += srcSize;
                }
            }
            else if (p.dilationX*p.dilationY*p.strideX*p.strideY != 1)
            {
                __m512 _0 = _mm512_setzero_ps();
                __m512i index = _mm512_loadu_si512(_index.data);
                size_t aligned = AlignHi(p.dstW, F) - F;
                __mmask16 storeTail = TailMask16(p.dstW - aligned);
                __mmask16 storeNose = aligned ? __mmask16(-1) : storeTail;
                for (size_t c = 0; c < p.srcC; ++c)
                {
                    for (size_t ky = 0; ky < p.kernelY; ky++)
                    {
                        for (size_t kx = 0; kx < p.kernelX; kx++)
                        {
                            __mmask16 nose = _nose[kx];
                            __mmask16 tail = _tail[kx];
                            size_t sx0 = kx * p.dilationX - p.padX;
                            size_t sy = ky * p.dilationY - p.padY;
                            for (size_t dy = 0; dy < p.dstH; ++dy)
                            {
                                if (sy < p.srcH)
                                {
                                    size_t dx = 0, sx = sx0 + sy * p.srcW;
                                    _mm512_mask_storeu_ps(dst + dx, storeNose, _mm512_mask_i32gather_ps(_0, nose, index, (src + sx), 4));
                                    dx += F, sx += p.strideX*F;
                                    //if (p.strideX == 3)
                                    //{
                                    //    for (; dx < aligned; dx += F, sx += p.strideX*F)
                                    //        _mm512_storeu_ps(dst + dx, Avx512f::Gather<3>(src + sx));
                                    //}
                                    //else
                                    //{
                                        for (; dx < aligned; dx += F, sx += p.strideX*F)
                                            _mm512_storeu_ps(dst + dx, _mm512_i32gather_ps(index, (src + sx), 4));
                                    //}
                                    if (aligned)
                                        _mm512_mask_storeu_ps(dst + dx, storeTail, _mm512_mask_i32gather_ps(_0, tail, index, (src + sx), 4));
                                }
                                else
                                {
                                    memset(dst, 0, p.dstW * sizeof(float));
                                }
                                dst += p.dstW;
                                sy += p.strideY;
                            }
                        }
                    }
                    src += srcSize;
                }
            }
            else
            {
                Base::ConvolutionImgToCol::ImgToCol(src, dst);
            }
        }

        //---------------------------------------------------------------------

        ConvolutionImgToRow::ConvolutionImgToRow(const ConvParam & p)
            : Avx2::ConvolutionImgToRow(p)
        {
        }

        void ConvolutionImgToRow::GemmAndBias(const float * src, float * dst)
        {
            const ConvParam & p = _param;
            for (size_t g = 0; g < p.group; ++g)
                Avx512f::Gemm32fNT(_M, _N, _K, &_1, _weight + _weightStep * g, _K, src + _srcStep * g, _K, &_0, dst + _dstStep * g, _N);
            Avx512f::BiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionWinograd2x3p::ConvolutionWinograd2x3p(const ConvParam & p)
            : Avx2::ConvolutionWinograd2x3p(p)
        {
        }

        void ConvolutionWinograd2x3p::Forward(const float * src, float * buf, float * dst)
        {
            const ConvParam & p = _param;
            float * bufS = Buffer(buf);
            float * bufD = bufS + _strideS * _count;
            Avx512f::Winograd2x3pSetInput(src, p.srcC, p.srcH, p.srcW, buf, _pad);
            for (size_t i = 0; i < _count; ++i)
                Avx512f::Gemm32fNN(_M, _N, _K, &_1, _weight.data + i * _strideW, _K, bufS + i * _strideS, _N, &_0, bufD + i * _strideD, _N);
            Avx512f::Winograd2x3pSetOutput(bufD, dst, p.dstC, p.dstH, p.dstW);
            Avx512f::BiasAndActivation(_bias, p.dstC, p.dstH*p.dstW, _activationType, _activationParams, dst);
        }

        //---------------------------------------------------------------------

        ConvolutionDirect::ConvolutionDirect(const ConvParam & p)
            : Avx2::ConvolutionDirect(p)
        {
            _convolutionBiasActivation = SetConvolutionBiasActivation();
        }

        template <size_t size> SIMD_INLINE void LoadWeight(const float * src, __m512 * dst)
        {
            for (size_t i = 0; i < size; ++i)
                dst[i] = _mm512_set1_ps(src[i]);
        }

        template<int kernel, int stride> struct Kernel
        {
            static __m512 Convolution(const float * src, size_t step, const __m512  * weight);
        };

        template<> struct Kernel<1, 1>
        {
            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_mul_ps(_mm512_loadu_ps(src), weight[0]);
            }
        };

        template<> struct Kernel<1, 2>
        {
            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                __m512 s0 = _mm512_loadu_ps(src + 0);
                __m512 s1 = _mm512_loadu_ps(src + F);
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_mul_ps(_mm512_shuffle_ps(s0, s1, 0x88), weight[0]));
            }
        };

        template<> struct Kernel<2, 1>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0],
                    _mm512_mul_ps(_mm512_loadu_ps(src + 1), weight[1]));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2));
            }
        };

        template<> struct Kernel<2, 2>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 s0 = _mm512_loadu_ps(src + 0);
                __m512 s1 = _mm512_loadu_ps(src + F);
                return _mm512_fmadd_ps(_mm512_shuffle_ps(s0, s1, 0x88), weight[0],
                    _mm512_mul_ps(_mm512_shuffle_ps(s0, s1, 0xDD), weight[1]));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_add_ps(RowConv(src, weight), RowConv(src + step, weight + 2)));
            }
        };

        template<> struct Kernel<3, 1>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0],
                    _mm512_fmadd_ps(_mm512_loadu_ps(src + 1), weight[1],
                        _mm512_mul_ps(_mm512_loadu_ps(src + 2), weight[2])));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight),
                    _mm512_add_ps(RowConv(src + step, weight + 3),
                        RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<3, 2>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 s00 = _mm512_loadu_ps(src);
                __m512 s10 = _mm512_loadu_ps(src + F);
                __m512 s02 = _mm512_loadu_ps(src + 2);
                __m512 s12 = _mm512_loadu_ps(src + 2 + F);
                return _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0x88), weight[0],
                    _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0xDD), weight[1],
                        _mm512_mul_ps(_mm512_shuffle_ps(s02, s12, 0x88), weight[2])));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_add_ps(RowConv(src, weight),
                    _mm512_add_ps(RowConv(src + step, weight + 3), RowConv(src + 2 * step, weight + 6))));
            }
        };

        const __m512i K32_IDX_3_0A = SIMD_MM512_SETR_EPI32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
        const __m512i K32_IDX_3_0B = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 4, 7, 10, 13);
        const __m512i K32_IDX_3_1A = SIMD_MM512_SETR_EPI32(1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46);
        const __m512i K32_IDX_3_1B = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 5, 8, 11, 14);
        const __m512i K32_IDX_3_2A = SIMD_MM512_SETR_EPI32(2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, 44, 47);
        const __m512i K32_IDX_3_2B = SIMD_MM512_SETR_EPI32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 6, 9, 12, 15);

        template<> struct Kernel<3, 3>
        {

            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 src0 = _mm512_loadu_ps(src + 0 * F);
                __m512 src1 = _mm512_loadu_ps(src + 1 * F);
                __m512 src2 = _mm512_loadu_ps(src + 2 * F);
                __m512 s0 = _mm512_mask_permutexvar_ps(_mm512_maskz_permutex2var_ps(0xFFFF, src0, K32_IDX_3_0A, src1), 0xF800, K32_IDX_3_0B, src2);
                __m512 s1 = _mm512_mask_permutexvar_ps(_mm512_maskz_permutex2var_ps(0xFFFF, src0, K32_IDX_3_1A, src1), 0xF800, K32_IDX_3_1B, src2);
                __m512 s2 = _mm512_mask_permutexvar_ps(_mm512_maskz_permutex2var_ps(0xFFFF, src0, K32_IDX_3_2A, src1), 0xFC00, K32_IDX_3_2B, src2);
                return _mm512_fmadd_ps(s0, weight[0], _mm512_fmadd_ps(s1, weight[1], _mm512_mul_ps(s2, weight[2])));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight), _mm512_add_ps(RowConv(src + step, weight + 3), RowConv(src + 2 * step, weight + 6)));
            }
        };

        template<> struct Kernel<4, 1>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0], _mm512_fmadd_ps(_mm512_loadu_ps(src + 1), weight[1],
                        _mm512_fmadd_ps(_mm512_loadu_ps(src + 2), weight[2], _mm512_mul_ps(_mm512_loadu_ps(src + 3), weight[3]))));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight), _mm512_add_ps(RowConv(src + step, weight + 4),
                    _mm512_add_ps(RowConv(src + 2 * step, weight + 8), RowConv(src + 3 * step, weight + 12))));
            }
        };

        template<> struct Kernel<4, 2>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 s00 = _mm512_loadu_ps(src);
                __m512 s10 = _mm512_loadu_ps(src + F);
                __m512 s02 = _mm512_loadu_ps(src + 2);
                __m512 s12 = _mm512_loadu_ps(src + 2 + F);
                return _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0x88), weight[0], _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0xDD), weight[1],
                    _mm512_fmadd_ps(_mm512_shuffle_ps(s02, s12, 0x88), weight[2], _mm512_mul_ps(_mm512_shuffle_ps(s02, s12, 0xDD), weight[3]))));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_add_ps(RowConv(src, weight),
                    _mm512_add_ps(RowConv(src + step, weight + 4), _mm512_add_ps(RowConv(src + 2 * step, weight + 8), RowConv(src + 3 * step, weight + 12)))));
            }
        };

        template<> struct Kernel<5, 1>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                return _mm512_fmadd_ps(_mm512_loadu_ps(src), weight[0], _mm512_fmadd_ps(_mm512_loadu_ps(src + 1), weight[1],
                    _mm512_fmadd_ps(_mm512_loadu_ps(src + 2), weight[2], _mm512_fmadd_ps(_mm512_loadu_ps(src + 3), weight[3],
                        _mm512_mul_ps(_mm512_loadu_ps(src + 4), weight[4])))));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_add_ps(RowConv(src, weight), _mm512_add_ps(RowConv(src + step, weight + 5),
                    _mm512_add_ps(RowConv(src + 2 * step, weight + 10), _mm512_add_ps(RowConv(src + 3 * step, weight + 15), 
                        RowConv(src + 4 * step, weight + 20)))));
            }
        };

        template<> struct Kernel<5, 2>
        {
            static SIMD_INLINE __m512 RowConv(const float * src, const __m512  * weight)
            {
                __m512 s00 = _mm512_loadu_ps(src);
                __m512 s10 = _mm512_loadu_ps(src + F);
                __m512 s02 = _mm512_loadu_ps(src + 2);
                __m512 s12 = _mm512_loadu_ps(src + 2 + F);
                __m512 s04 = _mm512_loadu_ps(src + 4);
                __m512 s14 = _mm512_loadu_ps(src + 4 + F);
                return _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0x88), weight[0], _mm512_fmadd_ps(_mm512_shuffle_ps(s00, s10, 0xDD), weight[1],
                    _mm512_fmadd_ps(_mm512_shuffle_ps(s02, s12, 0x88), weight[2], _mm512_fmadd_ps(_mm512_shuffle_ps(s02, s12, 0xDD), weight[3],
                        _mm512_mul_ps(_mm512_shuffle_ps(s04, s14, 0x88), weight[4])))));
            }

            static SIMD_INLINE __m512 Convolution(const float * src, size_t step, const __m512  * weight)
            {
                return _mm512_permutexvar_ps(K32_PERMUTE_FOR_PACK, _mm512_add_ps(RowConv(src, weight), _mm512_add_ps(RowConv(src + step, weight + 5), 
                    _mm512_add_ps(RowConv(src + 2 * step, weight + 10), _mm512_add_ps(RowConv(src + 3 * step, weight + 15), RowConv(src + 4 * step, weight + 20))))));
            }
        };

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m512 Activate(__m512 value, const __m512 * params);

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationIdentity>(__m512 value, const __m512 * params)
        {
            return value;
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRelu>(__m512 value, const __m512 * params)
        {
            return _mm512_max_ps(_mm512_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationLeakyRelu>(__m512 value, const __m512 * params)
        {
            return _mm512_add_ps(_mm512_max_ps(_mm512_setzero_ps(), value), _mm512_mul_ps(params[0], _mm512_min_ps(_mm512_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRestrictRange>(__m512 value, const __m512 * params)
        {
            return _mm512_min_ps(_mm512_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationPrelu>(__m512 value, const __m512 * params)
        {
            return _mm512_add_ps(_mm512_max_ps(_mm512_setzero_ps(), value), _mm512_mul_ps(params[0], _mm512_min_ps(_mm512_setzero_ps(), value)));
        }

        template<int kernel, int stride, ::SimdConvolutionActivationType type> 
        void ConvolutionBiasActivation(const float * src, size_t srcC, size_t srcH, size_t srcW, const float * weight, 
            const float * bias, const float * params, float * dst, size_t dstC, size_t dstH, size_t dstW)
        {
            __m512 _weight[kernel*kernel];
            __m512 _params[2];
            _params[0] = _mm512_set1_ps(params[0]);
            if (type == ::SimdConvolutionActivationRestrictRange)
                _params[1] = _mm512_set1_ps(params[1]);
            size_t dstWF = Simd::AlignLo(dstW, F);
            __mmask16 tail = TailMask16(dstW - dstWF);
            for (size_t dc = 0; dc < dstC; ++dc)
            {
                if (type == ::SimdConvolutionActivationPrelu)
                    _params[0] = _mm512_set1_ps(params[dc]);
                if (srcC == 1)
                {
                    const float * ps = src;
                    float * pd = dst;
                    LoadWeight<kernel*kernel>(weight, _weight);
                    __m512 _bias = bias ? _mm512_set1_ps(bias[dc]) : _mm512_setzero_ps();
                    for (size_t y = 0; y < dstH; ++y)
                    {
                        size_t x = 0;
                        for (; x < dstWF; x += F)
                        {
                            __m512 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                            _mm512_storeu_ps(pd + x, Activate<type>(_mm512_add_ps(_bias, conv), _params));
                        }
                        if (x < dstW)
                        {
                            __m512 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                            _mm512_mask_storeu_ps(pd + x, tail, Activate<type>(_mm512_add_ps(_bias, conv), _params));
                        }
                        ps += srcW * stride;
                        pd += dstW;
                    }
                    weight += kernel * kernel;
                }
                else
                {
                    size_t sc = 0;
                    for (; sc < 1; ++sc)
                    {
                        const float * ps = src;
                        float * pd = dst;
                        LoadWeight<kernel*kernel>(weight, _weight);
                        __m512 _bias = bias ? _mm512_set1_ps(bias[dc]) : _mm512_setzero_ps();
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            size_t x = 0;
                            for (; x < dstWF; x += F)
                            {
                                __m512 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm512_storeu_ps(pd + x, _mm512_add_ps(_bias, conv));
                            }
                            if (x < dstW)
                            {
                                __m512 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm512_mask_storeu_ps(pd + x, tail, _mm512_add_ps(_bias, conv));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                    for (; sc < srcC - 1; ++sc)
                    {
                        const float * ps = src + sc * srcW * srcH;
                        float * pd = dst;
                        LoadWeight<kernel*kernel>(weight, _weight);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            size_t x = 0;
                            for (; x < dstWF; x += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(pd + x);
                                __m512 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm512_storeu_ps(pd + x, _mm512_add_ps(_dst, conv));
                            }
                            if (x < dstW)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, pd + x);
                                __m512 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm512_mask_storeu_ps(pd + x, tail, _mm512_add_ps(_dst, conv));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                    for (; sc < srcC; ++sc)
                    {
                        const float * ps = src + sc * srcW * srcH;
                        float * pd = dst;
                        LoadWeight<kernel*kernel>(weight, _weight);
                        for (size_t y = 0; y < dstH; ++y)
                        {
                            size_t x = 0;
                            for (; x < dstWF; x += F)
                            {
                                __m512 _dst = _mm512_loadu_ps(pd + x);
                                __m512 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm512_storeu_ps(pd + x, Activate<type>(_mm512_add_ps(_dst, conv), _params));
                            }
                            if (x < dstW)
                            {
                                __m512 _dst = _mm512_maskz_loadu_ps(tail, pd + x);
                                __m512 conv = Kernel<kernel, stride>::Convolution(ps + x * stride, srcW, _weight);
                                _mm512_mask_storeu_ps(pd + x, tail, Activate<type>(_mm512_add_ps(_dst, conv), _params));
                            }
                            ps += srcW * stride;
                            pd += dstW;
                        }
                        weight += kernel * kernel;
                    }
                }
                dst += dstH * dstW;
            }
        }

         bool ConvolutionDirect::Preferable(const ConvParam & p)
        {
            if (!p.IsDilation(1))
                return false;
            if (!(p.IsStride(1) || p.IsStride(2) || p.IsStride(3)))
                return false;
            double k = double(p.srcC) / p.group * p.strideX * p.strideY / p.kernelX / p.kernelY;
            return k < 2.0 && ((p.IsStride(1) && p.IsKernel(1)) || p.IsKernel(2) || p.IsKernel(3)
#if SIMD_ZMM_COUNT == 32
                    || p.IsKernel(4) || p.IsKernel(5)
#endif
                    );
        }

        template <int kernel, int stride> ConvolutionDirect::ConvolutionBiasActivationPtr SetConvolutionBiasActivation(::SimdConvolutionActivationType type)
        {
            switch (type)
            {
            case ::SimdConvolutionActivationIdentity: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationIdentity>;
            case ::SimdConvolutionActivationRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRelu>;
            case ::SimdConvolutionActivationLeakyRelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationLeakyRelu>;
            case ::SimdConvolutionActivationRestrictRange: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationRestrictRange>;
            case ::SimdConvolutionActivationPrelu: return ConvolutionBiasActivation<kernel, stride, ::SimdConvolutionActivationPrelu>;
            default:
                assert(0);
                return NULL;
            }
        }

        ConvolutionDirect::ConvolutionBiasActivationPtr ConvolutionDirect::SetConvolutionBiasActivation()
        {
            const ConvParam & p = _param;
            if (p.dstW <= HF)
                return Avx2::ConvolutionDirect::SetConvolutionBiasActivation();
            switch (p.strideX)
            {
            case 1:
                if (p.kernelX == 1)
                    return Avx512f::SetConvolutionBiasActivation<1, 1>(_activationType);
                if (p.kernelX == 2)
                    return Avx512f::SetConvolutionBiasActivation<2, 1>(_activationType);
                if (p.kernelX == 3)
                    return Avx512f::SetConvolutionBiasActivation<3, 1>(_activationType);
                if (p.kernelX == 4)
                    return Avx512f::SetConvolutionBiasActivation<4, 1>(_activationType);
                if (p.kernelX == 5)
                    return Avx512f::SetConvolutionBiasActivation<5, 1>(_activationType);
                break;
            case 2:
                if (p.kernelX == 2)
                    return Avx512f::SetConvolutionBiasActivation<2, 2>(_activationType);
                if (p.kernelX == 3)
                    return Avx512f::SetConvolutionBiasActivation<3, 2>(_activationType);
                if (p.kernelX == 4)
                    return Avx512f::SetConvolutionBiasActivation<4, 2>(_activationType);
                if (p.kernelX == 5)
                    return Avx512f::SetConvolutionBiasActivation<5, 2>(_activationType);
                break;
            case 3:
                if (p.kernelX == 3)
                    return Avx512f::SetConvolutionBiasActivation<3, 3>(_activationType);
                break;
            }
            return Avx2::ConvolutionDirect::SetConvolutionBiasActivation();
        }

        //---------------------------------------------------------------------

        void * ConvolutionInit(size_t srcC, size_t srcH, size_t srcW, size_t dstC, size_t kernelY, size_t kernelX, size_t dilationY, size_t dilationX, size_t strideY, size_t strideX, size_t padY, size_t padX, size_t padH, size_t padW, size_t group)
        {
            ConvParam param(srcC, srcH, srcW, dstC, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group);
            if (Avx::ConvolutionDepthwiseDotProduct::Preferable(param))
                return new Avx::ConvolutionDepthwiseDotProduct(param);
            else if (ConvolutionWinograd2x3p::Preferable(param))
                return new ConvolutionWinograd2x3p(param);
            else if (ConvolutionImgToRow::Preferable(param))
                return new ConvolutionImgToRow(param);
            else if (ConvolutionDirect::Preferable(param))
                return new Avx512f::ConvolutionDirect(param);
            else
                return new ConvolutionImgToCol(param);
        }
    }
#endif//SIMD_AVX512F_ENABLE
}

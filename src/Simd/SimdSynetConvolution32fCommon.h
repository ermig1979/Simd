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
#ifndef __SimdSynetConvolution32fCommon_h__
#define __SimdSynetConvolution32fCommon_h__

#include "Simd/SimdMath.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExp.h"

namespace Simd
{
    enum TermType
    {
        TermLast,
        TermInterim,
        TermSize
    };

    namespace Base
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE float Activate(float value, const float * params, size_t offset);

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationIdentity>(float value, const float * params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationLeakyRelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[0] * Simd::Min(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationRestrictRange>(float value, const float * params, size_t offset)
        {
            return Simd::Min(Simd::Max(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationPrelu>(float value, const float * params, size_t offset)
        {
            return Simd::Max(0.0f, value) + params[offset] * Simd::Min(0.0f, value);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationElu>(float value, const float * params, size_t offset)
        {
            return SynetElu32f(value, params[0]);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationHswish>(float value, const float * params, size_t offset)
        {
            return SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationMish>(float value, const float* params, size_t offset)
        {
            return SynetMish32f(value, params[0]);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationHardSigmoid>(float value, const float* params, size_t offset)
        {
            return SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationSwish>(float value, const float* params, size_t offset)
        {
            return SynetSwish32f(value, params[0]);
        }

        template<SimdConvolutionActivationType type> void DepthwiseConvolution(const float* src, const SimdConvolutionParameters& p,
            size_t maC, size_t yBeg, size_t yEnd, const size_t bufH[2], const float* weight, const float* bias, const float* params, float* dst, int first)
        {
            assert(p.group == p.srcC && p.group == p.dstC);
            size_t srcH = p.srcH, srcW = p.srcW, srcC = p.srcC, dstW = p.dstW;
            size_t kernelY = p.kernelY, kernelX = p.kernelX, strideY = p.strideY, strideX = p.strideX, padY = p.padY, padX = p.padX;
            for (size_t dy = yBeg; dy < yEnd; ++dy)
            {
                for (size_t dx = 0; dx < dstW; ++dx)
                {
                    for (size_t c = 0; c < srcC; ++c)
                    {
                        float sum = bias ? bias[c] : 0;
                        for (size_t ky = 0; ky < kernelY; ++ky)
                        {
                            size_t sy = dy * strideY + ky - padY;
                            if (sy < srcH)
                            {
                                for (size_t kx = 0; kx < kernelX; ++kx)
                                {
                                    size_t sx = dx * strideX + kx - padX;
                                    if (sx < srcW)
                                    {
                                        const float* pw = weight + (ky * kernelX + kx) * srcC + c;
                                        const float* ps = src + (sy * srcW + sx) * srcC + c;
                                        sum += ps[0] * pw[0];
                                    }
                                }
                            }
                        }
                        dst[c] = Activate<type>(sum, params, c);
                    }
                    dst += srcC;
                }
            }
        }
    }

#ifdef SIMD_SSE2_ENABLE    
    namespace Sse2
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 Activate(__m128 value, const float* params, size_t offset);

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationIdentity>(__m128 value, const float* params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRelu>(__m128 value, const float* params, size_t offset)
        {
            return _mm_max_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationLeakyRelu>(__m128 value, const float* params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_set1_ps(params[0]), _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRestrictRange>(__m128 value, const float* params, size_t offset)
        {
            return _mm_min_ps(_mm_max_ps(_mm_set1_ps(params[0]), value), _mm_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationPrelu>(__m128 value, const float* params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_loadu_ps(params + offset), _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationElu>(__m128 value, const float* params, size_t offset)
        {
            return Sse2::Elu(value, _mm_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHswish>(__m128 value, const float* params, size_t offset)
        {
            return SynetHswish32f(value, _mm_set1_ps(params[0]), _mm_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationMish>(__m128 value, const float* params, size_t offset)
        {
            return Sse2::Mish(value, _mm_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHardSigmoid>(__m128 value, const float* params, size_t offset)
        {
            return SynetHardSigmoid32f(value, _mm_set1_ps(params[0]), _mm_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationSwish>(__m128 value, const float* params, size_t offset)
        {
            return Swish(value, _mm_set1_ps(params[0]));
        }

        //---------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 Activate(__m128 value, const __m128 * params, size_t index);

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationIdentity>(__m128 value, const __m128 * params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRelu>(__m128 value, const __m128 * params, size_t index)
        {
            return _mm_max_ps(_mm_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationLeakyRelu>(__m128 value, const __m128 * params, size_t index)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(params[0], _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationRestrictRange>(__m128 value, const __m128 * params, size_t index)
        {
            return _mm_min_ps(_mm_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationPrelu>(__m128 value, const __m128 * params, size_t index)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(params[index], _mm_min_ps(_mm_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationElu>(__m128 value, const __m128 * params, size_t index)
        {
            return Sse2::Elu(value, params[0]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHswish>(__m128 value, const __m128 * params, size_t index)
        {
            return SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationMish>(__m128 value, const __m128* params, size_t index)
        {
            return Sse2::Mish(value, params[0]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHardSigmoid>(__m128 value, const __m128* params, size_t index)
        {
            return SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationSwish>(__m128 value, const __m128* params, size_t index)
        {
            return Sse2::Swish(value, params[0]);
        }

        //---------------------------------------------------------------------

        template <TermType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m128 value, const __m128 * bias, const __m128 * params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m128 value, const __m128 * bias, const __m128 * params, size_t tail);
        };

        template <> struct Term<TermLast>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m128 value, const __m128 * bias, const __m128 * params)
            {
                _mm_storeu_ps(ptr, Activate<type>(_mm_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m128 value, const __m128 * bias, const __m128 * params, size_t tail)
            {
                float tmp[F];
                _mm_storeu_ps(tmp, Activate<type>(_mm_add_ps(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template <> struct Term<TermInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m128 value, const __m128 * bias, const __m128 * params)
            {
                _mm_storeu_ps(ptr, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m128 value, const __m128 * bias, const __m128 * params, size_t tail)
            {
                float tmp[F];
                _mm_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m128 val0, const __m128* bias, const __m128* params)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m128 val0, const __m128* bias, const __m128* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m128 val0, __m128 val1, const __m128* bias, const __m128* params)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m128 val0, __m128 val1, const __m128* bias, const __m128* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, __m128 val0, __m128 val1, __m128 val2, const __m128* bias, const __m128* params)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, __m128 val0, __m128 val1, __m128 val2, const __m128* bias, const __m128* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params, tail);
        }
    }
#endif//SIMD_SSE2_ENABLE

#ifdef SIMD_AVX_ENABLE    
    namespace Avx
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m256 Activate(__m256 value, const __m256 * params, size_t index);

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationIdentity>(__m256 value, const __m256 * params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRelu>(__m256 value, const __m256 * params, size_t index)
        {
            return _mm256_max_ps(_mm256_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationLeakyRelu>(__m256 value, const __m256 * params, size_t index)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(params[0], _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRestrictRange>(__m256 value, const __m256 * params, size_t index)
        {
            return _mm256_min_ps(_mm256_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationPrelu>(__m256 value, const __m256 * params, size_t index)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(params[index], _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationHswish>(__m256 value, const __m256 * params, size_t index)
        {
            return Avx::SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationHardSigmoid>(__m256 value, const __m256* params, size_t index)
        {
            return Avx::SynetHardSigmoid32f(value, params[0], params[1]);
        }

        //---------------------------------------------------------------------

        template <TermType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params, size_t tail);
        };

        template <> struct Term<TermLast>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params)
            {
                _mm256_storeu_ps(ptr, Activate<type>(_mm256_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params, size_t tail)
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, Activate<type>(_mm256_add_ps(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template <> struct Term<TermInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params)
            {
                _mm256_storeu_ps(ptr, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params, size_t tail)
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m256 val0, const __m256* bias, const __m256* params)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m256 val0, const __m256* bias, const __m256* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m256 val0, __m256 val1, const __m256* bias, const __m256* params)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m256 val0, __m256 val1, const __m256* bias, const __m256* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, __m256 val0, __m256 val1, __m256 val2, const __m256* bias, const __m256* params)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, __m256 val0, __m256 val1, __m256 val2, const __m256* bias, const __m256* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params, tail);
        }
    }
#endif//SIMD_AVX_ENABLE

#ifdef SIMD_AVX2_ENABLE    
    namespace Avx2
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m256 Activate(__m256 value, const float* params, size_t offset);

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationIdentity>(__m256 value, const float* params, size_t offset)
        {
            return value;
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRelu>(__m256 value, const float* params, size_t offset)
        {
            return _mm256_max_ps(_mm256_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationLeakyRelu>(__m256 value, const float* params, size_t offset)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(_mm256_set1_ps(params[0]), _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRestrictRange>(__m256 value, const float* params, size_t offset)
        {
            return _mm256_min_ps(_mm256_max_ps(_mm256_set1_ps(params[0]), value), _mm256_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationPrelu>(__m256 value, const float* params, size_t offset)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(_mm256_loadu_ps(params + offset), _mm256_min_ps(_mm256_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationElu>(__m256 value, const float* params, size_t offset)
        {
            return Avx2::Elu(value, _mm256_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationHswish>(__m256 value, const float* params, size_t offset)
        {
            return Avx::SynetHswish32f(value, _mm256_set1_ps(params[0]), _mm256_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationMish>(__m256 value, const float* params, size_t offset)
        {
            return Avx2::Mish(value, _mm256_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationHardSigmoid>(__m256 value, const float* params, size_t offset)
        {
            return Avx::SynetHardSigmoid32f(value, _mm256_set1_ps(params[0]), _mm256_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationSwish>(__m256 value, const float* params, size_t offset)
        {
            return Avx2::Swish(value, _mm256_set1_ps(params[0]));
        }

        //---------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m256 Activate(__m256 value, const __m256 * params, size_t index);

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationIdentity>(__m256 value, const __m256 * params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRelu>(__m256 value, const __m256 * params, size_t index)
        {
            return _mm256_max_ps(_mm256_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationLeakyRelu>(__m256 value, const __m256 * params, size_t index)
        {
            return _mm256_fmadd_ps(params[0], _mm256_min_ps(_mm256_setzero_ps(), value), _mm256_max_ps(_mm256_setzero_ps(), value));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationRestrictRange>(__m256 value, const __m256 * params, size_t index)
        {
            return _mm256_min_ps(_mm256_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationPrelu>(__m256 value, const __m256 * params, size_t index)
        {
            return _mm256_fmadd_ps(params[index], _mm256_min_ps(_mm256_setzero_ps(), value), _mm256_max_ps(_mm256_setzero_ps(), value));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationElu>(__m256 value, const __m256 * params, size_t index)
        {
            return Avx2::Elu(value, params[0]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationHswish>(__m256 value, const __m256 * params, size_t index)
        {
            return Avx::SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationMish>(__m256 value, const __m256* params, size_t index)
        {
            return Avx2::Mish(value, params[0]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationHardSigmoid>(__m256 value, const __m256* params, size_t index)
        {
            return Avx::SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationSwish>(__m256 value, const __m256* params, size_t index)
        {
            return Avx2::Swish(value, params[0]);
        }

        //---------------------------------------------------------------------

        template <TermType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params, size_t tail);
        };

        template <> struct Term<TermLast>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params)
            {
                _mm256_storeu_ps(ptr, Activate<type>(_mm256_add_ps(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params, size_t tail)
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, Activate<type>(_mm256_add_ps(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template <> struct Term<TermInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params)
            {
                _mm256_storeu_ps(ptr, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m256 value, const __m256 * bias, const __m256 * params, size_t tail)
            {
                float tmp[F];
                _mm256_storeu_ps(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m256 val0, const __m256* bias, const __m256* params)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m256 val0, const __m256* bias, const __m256* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m256 val0, __m256 val1, const __m256* bias, const __m256* params)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m256 val0, __m256 val1, const __m256* bias, const __m256* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, __m256 val0, __m256 val1, __m256 val2, const __m256* bias, const __m256* params)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, __m256 val0, __m256 val1, __m256 val2, const __m256* bias, const __m256* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params, tail);
        }
    }
#endif//SIMD_AVX2_ENABLE

#ifdef SIMD_AVX512F_ENABLE    
    namespace Avx512f
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE __m512 Activate(__m512 value, const float* params, size_t offset, __mmask16 tail = -1);

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationIdentity>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return value;
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRelu>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return _mm512_max_ps(_mm512_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationLeakyRelu>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return _mm512_add_ps(_mm512_max_ps(_mm512_setzero_ps(), value), _mm512_mul_ps(_mm512_set1_ps(params[0]), _mm512_min_ps(_mm512_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRestrictRange>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return _mm512_min_ps(_mm512_max_ps(_mm512_set1_ps(params[0]), value), _mm512_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationPrelu>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return _mm512_add_ps(_mm512_max_ps(_mm512_setzero_ps(), value), _mm512_mul_ps(_mm512_maskz_loadu_ps(tail, params + offset), _mm512_min_ps(_mm512_setzero_ps(), value)));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationElu>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return Avx512f::Elu(value, _mm512_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHswish>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return Avx512f::SynetHswish32f(value, _mm512_set1_ps(params[0]), _mm512_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationMish>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return Avx512f::Mish(value, _mm512_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHardSigmoid>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return Avx512f::SynetHardSigmoid32f(value, _mm512_set1_ps(params[0]), _mm512_set1_ps(params[1]));
        }


        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationSwish>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return Avx512f::Swish(value, _mm512_set1_ps(params[0]));
        }

        //---------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m512 Activate(__m512 value, const __m512 * params, size_t index);

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationIdentity>(__m512 value, const __m512 * params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRelu>(__m512 value, const __m512 * params, size_t index)
        {
            return _mm512_max_ps(_mm512_setzero_ps(), value);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationLeakyRelu>(__m512 value, const __m512 * params, size_t index)
        {
            return _mm512_fmadd_ps(params[0], _mm512_min_ps(_mm512_setzero_ps(), value), _mm512_max_ps(_mm512_setzero_ps(), value));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationRestrictRange>(__m512 value, const __m512 * params, size_t index)
        {
            return _mm512_min_ps(_mm512_max_ps(params[0], value), params[1]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationPrelu>(__m512 value, const __m512 * params, size_t index)
        {
            return _mm512_fmadd_ps(params[index], _mm512_min_ps(_mm512_setzero_ps(), value), _mm512_max_ps(_mm512_setzero_ps(), value));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationElu>(__m512 value, const __m512 * params, size_t index)
        {
            return Avx512f::Elu(value, params[0]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHswish>(__m512 value, const __m512 * params, size_t index)
        {
            return Avx512f::SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationMish>(__m512 value, const __m512* params, size_t index)
        {
            return Avx512f::Mish(value, params[0]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHardSigmoid>(__m512 value, const __m512* params, size_t index)
        {
            return Avx512f::SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationSwish>(__m512 value, const __m512* params, size_t index)
        {
            return Avx512f::Swish(value, params[0]);
        }

        //---------------------------------------------------------------------

        template <TermType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m512 value, const __m512 * bias, const __m512 * params, __mmask16 tail = __mmask16(-1));
        };

        template <> struct Term<TermLast>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m512 value, const __m512 * bias, const __m512 * params, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps(ptr, tail, Activate<type>(_mm512_add_ps(value, bias[index]), params, index));
            }
        };

        template <> struct Term<TermInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, __m512 value, const __m512 * bias, const __m512 * params, __mmask16 tail = __mmask16(-1))
            {
                _mm512_mask_storeu_ps(ptr, tail, value);
            }
        };

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, __m512 val0, const __m512* bias, const __m512* params, const __mmask16 * tails)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params, tails[0]);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, __m512 val0, __m512 val1, const __m512* bias, const __m512* params, const __mmask16* tails)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params, tails[1]);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, __m512 val0, __m512 val1, __m512 val2, const __m512* bias, const __m512* params, const __mmask16* tails)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params, tails[2]);
        }
    }
#endif//SIMD_AVX512F_ENABLE

#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        template<::SimdConvolutionActivationType type> SIMD_INLINE float32x4_t Activate(float32x4_t value, const float32x4_t * params, size_t index);

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationIdentity>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return value;
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationRelu>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return vmaxq_f32(vdupq_n_f32(0.0f), value);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationLeakyRelu>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), value), params[0], vminq_f32(vdupq_n_f32(0.0f), value));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationRestrictRange>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return vminq_f32(vmaxq_f32(params[0], value), params[1]);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationPrelu>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return vmlaq_f32(vmaxq_f32(vdupq_n_f32(0.0f), value), params[index], vminq_f32(vdupq_n_f32(0.0f), value));
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationElu>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return Neon::Elu(value, params[0]);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationHswish>(float32x4_t value, const float32x4_t * params, size_t index)
        {
            return Neon::SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationMish>(float32x4_t value, const float32x4_t* params, size_t index)
        {
            return Neon::Mish<1>(value, params[0]);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationHardSigmoid>(float32x4_t value, const float32x4_t* params, size_t index)
        {
            return Neon::SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationSwish>(float32x4_t value, const float32x4_t* params, size_t index)
        {
            return Neon::Swish<1>(value, params[0]);
        }

        template <TermType term> struct Term
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, float32x4_t value, const float32x4_t * bias, const float32x4_t * params);
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, float32x4_t value, const float32x4_t * bias, const float32x4_t * params, size_t tail);
        };

        template <> struct Term<TermLast>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, float32x4_t value, const float32x4_t * bias, const float32x4_t * params)
            {
                Store<false>(ptr, Activate<type>(vaddq_f32(value, bias[index]), params, index));
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, float32x4_t value, const float32x4_t * bias, const float32x4_t * params, size_t tail)
            {
                float tmp[F];
                Store<false>(tmp, Activate<type>(vaddq_f32(value, bias[index]), params, index));
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template <> struct Term<TermInterim>
        {
            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, float32x4_t value, const float32x4_t * bias, const float32x4_t * params)
            {
                Store<false>(ptr, value);
            }

            template<SimdConvolutionActivationType type, int index> static SIMD_INLINE void Save(float * ptr, float32x4_t value, const float32x4_t * bias, const float32x4_t * params, size_t tail)
            {
                float tmp[F];
                Store<false>(tmp, value);
                for (size_t i = 0; i < tail; ++i)
                    ptr[i] = tmp[i];
            }
        };

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, float32x4_t val0, const float32x4_t* bias, const float32x4_t* params)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save1(float* dst, float32x4_t val0, const float32x4_t* bias, const float32x4_t* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst, val0, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, float32x4_t val0, float32x4_t val1, const float32x4_t* bias, const float32x4_t* params)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save2(float* dst, float32x4_t val0, float32x4_t val1, const float32x4_t* bias, const float32x4_t* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + F, val1, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, float32x4_t val0, float32x4_t val1, float32x4_t val2, const float32x4_t* bias, const float32x4_t* params)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save3(float* dst, float32x4_t val0, float32x4_t val1, float32x4_t val2, const float32x4_t* bias, const float32x4_t* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params, tail);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save4(float* dst, float32x4_t val0, float32x4_t val1, float32x4_t val2, float32x4_t val3, const float32x4_t* bias, const float32x4_t* params)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params);
            Term<term>::template Save<type, 3>(dst + 3 * F, val3, bias, params);
        }

        template<TermType term, SimdConvolutionActivationType type> SIMD_INLINE void Save4(float* dst, float32x4_t val0, float32x4_t val1, float32x4_t val2, float32x4_t val3, const float32x4_t* bias, const float32x4_t* params, size_t tail)
        {
            Term<term>::template Save<type, 0>(dst + 0 * F, val0, bias, params);
            Term<term>::template Save<type, 1>(dst + 1 * F, val1, bias, params);
            Term<term>::template Save<type, 2>(dst + 2 * F, val2, bias, params);
            Term<term>::template Save<type, 3>(dst + 3 * F, val3, bias, params, tail);
        }
    }
#endif//SIMD_NEON_ENABLE
}
#endif//__SimdSynetConvolution32fCommon_h__

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
#ifndef __SimdSynetActivation_h__
#define __SimdSynetActivation_h__

#include "Simd/SimdMath.h"
#include "Simd/SimdStore.h"
#include "Simd/SimdSynet.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdErf.h"

namespace Simd
{
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

        template<> SIMD_INLINE float Activate<SimdConvolutionActivationGelu>(float value, const float* params, size_t offset)
        {
            return Gelu(value);
        }
    }

#ifdef SIMD_SSE41_ENABLE    
    namespace Sse41
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
            return Elu(value, _mm_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHswish>(__m128 value, const float* params, size_t offset)
        {
            return SynetHswish32f(value, _mm_set1_ps(params[0]), _mm_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationMish>(__m128 value, const float* params, size_t offset)
        {
            return Mish(value, _mm_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHardSigmoid>(__m128 value, const float* params, size_t offset)
        {
            return SynetHardSigmoid32f(value, _mm_set1_ps(params[0]), _mm_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationSwish>(__m128 value, const float* params, size_t offset)
        {
            return Swish(value, _mm_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationGelu>(__m128 value, const float* params, size_t offset)
        {
            return Gelu(value);
        }

        //-------------------------------------------------------------------------------------------------

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
            return Elu(value, params[0]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHswish>(__m128 value, const __m128 * params, size_t index)
        {
            return SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationMish>(__m128 value, const __m128* params, size_t index)
        {
            return Mish(value, params[0]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationHardSigmoid>(__m128 value, const __m128* params, size_t index)
        {
            return SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationSwish>(__m128 value, const __m128* params, size_t index)
        {
            return Swish(value, params[0]);
        }

        template<> SIMD_INLINE __m128 Activate<::SimdConvolutionActivationGelu>(__m128 value, const __m128* params, size_t index)
        {
            return Gelu(value);
        }

        //-------------------------------------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m128 ActivateNchw(__m128 value, const float* params, size_t offset)
        {
            return Activate<type>(value, params, offset);
        }

        template<> SIMD_INLINE __m128 ActivateNchw<::SimdConvolutionActivationPrelu>(__m128 value, const float* params, size_t offset)
        {
            return _mm_add_ps(_mm_max_ps(_mm_setzero_ps(), value), _mm_mul_ps(_mm_set1_ps(params[offset]), _mm_min_ps(_mm_setzero_ps(), value)));
        }
    }
#endif

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
            return SynetHswish32f(value, _mm256_set1_ps(params[0]), _mm256_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationMish>(__m256 value, const float* params, size_t offset)
        {
            return Avx2::Mish(value, _mm256_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationHardSigmoid>(__m256 value, const float* params, size_t offset)
        {
            return SynetHardSigmoid32f(value, _mm256_set1_ps(params[0]), _mm256_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationSwish>(__m256 value, const float* params, size_t offset)
        {
            return Avx2::Swish(value, _mm256_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationGelu>(__m256 value, const float* params, size_t offset)
        {
            return Avx2::Gelu(value);
        }

        //-------------------------------------------------------------------------------------------------

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
            return SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationMish>(__m256 value, const __m256* params, size_t index)
        {
            return Avx2::Mish(value, params[0]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationHardSigmoid>(__m256 value, const __m256* params, size_t index)
        {
            return SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationSwish>(__m256 value, const __m256* params, size_t index)
        {
            return Avx2::Swish(value, params[0]);
        }

        template<> SIMD_INLINE __m256 Activate<::SimdConvolutionActivationGelu>(__m256 value, const __m256* params, size_t index)
        {
            return Avx2::Gelu(value);
        }

        //-------------------------------------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m256 ActivateNchw(__m256 value, const float* params, size_t offset)
        {
            return Activate<type>(value, params, offset);
        }

        template<> SIMD_INLINE __m256 ActivateNchw<::SimdConvolutionActivationPrelu>(__m256 value, const float* params, size_t offset)
        {
            return _mm256_add_ps(_mm256_max_ps(_mm256_setzero_ps(), value), _mm256_mul_ps(_mm256_set1_ps(params[offset]), _mm256_min_ps(_mm256_setzero_ps(), value)));
        }
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE    
    namespace Avx512bw
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
            return Elu(value, _mm512_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHswish>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return SynetHswish32f(value, _mm512_set1_ps(params[0]), _mm512_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationMish>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return Mish(value, _mm512_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHardSigmoid>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return SynetHardSigmoid32f(value, _mm512_set1_ps(params[0]), _mm512_set1_ps(params[1]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationSwish>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return Swish(value, _mm512_set1_ps(params[0]));
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationGelu>(__m512 value, const float* params, size_t offset, __mmask16 tail)
        {
            return Gelu(value);
        }

        //-------------------------------------------------------------------------------------------------

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
            return Elu(value, params[0]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHswish>(__m512 value, const __m512 * params, size_t index)
        {
            return SynetHswish32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationMish>(__m512 value, const __m512* params, size_t index)
        {
            return Mish(value, params[0]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationHardSigmoid>(__m512 value, const __m512* params, size_t index)
        {
            return SynetHardSigmoid32f(value, params[0], params[1]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationSwish>(__m512 value, const __m512* params, size_t index)
        {
            return Swish(value, params[0]);
        }

        template<> SIMD_INLINE __m512 Activate<::SimdConvolutionActivationGelu>(__m512 value, const __m512* params, size_t index)
        {
            return Gelu(value);
        }

        //-------------------------------------------------------------------------------------------------

        template<::SimdConvolutionActivationType type> SIMD_INLINE __m512 ActivateNchw(__m512 value, const float* params, size_t offset)
        {
            return Activate<type>(value, params, offset);
        }

        template<> SIMD_INLINE __m512 ActivateNchw<::SimdConvolutionActivationPrelu>(__m512 value, const float* params, size_t offset)
        {
            return _mm512_add_ps(_mm512_max_ps(_mm512_setzero_ps(), value), _mm512_mul_ps(_mm512_set1_ps(params[offset]), _mm512_min_ps(_mm512_setzero_ps(), value)));
        }
    }
#endif

#if defined(SIMD_AMXBF16_ENABLE) || (defined(SIMD_AVX512BW_ENABLE) && defined(SIMD_AMX_EMULATE)) 
    namespace AmxBf16
    {

    }
#endif

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

        template<> SIMD_INLINE float32x4_t Activate<::SimdConvolutionActivationGelu>(float32x4_t value, const float32x4_t* params, size_t index)
        {
            return Neon::Gelu<1>(value);
        }
    }
#endif

#ifdef SIMD_SVE2_ENABLE
    namespace Sve2
    {
        SIMD_INLINE svfloat32_t Poly5(const svbool_t& mask, svfloat32_t x)
        {
            svfloat32_t p = svdup_n_f32(1.8775767e-3f);
            p = svmla_f32_x(mask, svdup_n_f32(8.9893397e-3f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(5.5826318e-2f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(2.4015361e-1f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(6.9315308e-1f), x, p);
            p = svmla_f32_x(mask, svdup_n_f32(9.9999994e-1f), x, p);
            return p;
        }

        SIMD_INLINE svfloat32_t Exp2(const svbool_t& mask, svfloat32_t x)
        {
            x = svmax_f32_x(mask, svmin_f32_x(mask, x, svdup_n_f32(126.99999f)), svdup_n_f32(-126.99999f));
            svint32_t ipart = svcvt_s32_f32_x(mask, svsub_n_f32_x(mask, x, 0.5f));
            svfloat32_t fpart = svsub_f32_x(mask, x, svcvt_f32_s32_x(mask, ipart));
            svfloat32_t expipart = svreinterpret_f32_s32(svlsl_n_s32_x(mask, svadd_n_s32_x(mask, ipart, 127), 23));
            return svmul_f32_x(mask, expipart, Poly5(mask, fpart));
        }

        SIMD_INLINE svfloat32_t Exponent(const svbool_t& mask, svfloat32_t value)
        {
            return Exp2(mask, svmul_n_f32_x(mask, value, 1.44269504f));
        }

        SIMD_INLINE svfloat32_t Erf(const svbool_t& mask, svfloat32_t x)
        {
            const svfloat32_t _1 = svdup_n_f32(1.0f);
            svfloat32_t a = svmin_f32_x(mask, svabs_f32_x(mask, x), svdup_n_f32(9.0f));
            svfloat32_t p = svdup_n_f32(0.0000430638f);
            p = svmla_f32_x(mask, svdup_n_f32(0.0002765672f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0001520143f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0092705272f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0422820123f), a, p);
            p = svmla_f32_x(mask, svdup_n_f32(0.0705230784f), a, p);
            p = svmla_f32_x(mask, _1, a, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            p = svmul_f32_x(mask, p, p);
            svfloat32_t r = svsub_f32_x(mask, _1, svdiv_f32_x(mask, _1, p));
            return svsel_f32(svcmplt_n_f32(mask, x, 0.0f), svneg_f32_x(mask, r), r);
        }

        SIMD_INLINE svint32_t Round(const svfloat32_t& value, const svbool_t& mask)
        {
            return svcvt_s32_f32_x(mask, svrinta_f32_x(mask, value));
        }

        SIMD_INLINE svint32_t NearbyInt(const svfloat32_t& value, const svbool_t& mask)
        {
            return svcvt_s32_f32_x(mask, svrintn_f32_x(mask, value));
        }

        SIMD_INLINE svfloat32_t SynetHardSigmoid32f(const svbool_t& mask, svfloat32_t value, svfloat32_t scale, svfloat32_t shift)
        {
            return svmax_f32_x(mask, svmin_f32_x(mask, svmla_f32_x(mask, shift, value, scale), svdup_n_f32(1.0f)), svdup_n_f32(0.0f));
        }

        SIMD_INLINE svfloat32_t SynetHswish32f(const svbool_t& mask, svfloat32_t value, svfloat32_t shift, svfloat32_t scale)
        {
            svfloat32_t upper = svmin_f32_x(mask, value, shift);
            svfloat32_t positive = svmax_f32_x(mask, svadd_f32_x(mask, upper, shift), svdup_n_f32(0.0f));
            return svmul_f32_x(mask, svmul_f32_x(mask, positive, scale), value);
        }

        template<::SimdConvolutionActivationType type> SIMD_INLINE svfloat32_t Activate(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask);

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationIdentity>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            return value;
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationRelu>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            return svmax_n_f32_x(mask, value, 0.0f);
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationLeakyRelu>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), param0, svmin_n_f32_x(mask, value, 0.0f));
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationRestrictRange>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            return svmin_f32_x(mask, svmax_f32_x(mask, value, param0), param1);
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationPrelu>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            return svmla_f32_x(mask, svmax_n_f32_x(mask, value, 0.0f), index ? param1 : param0, svmin_n_f32_x(mask, value, 0.0f));
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationElu>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            svfloat32_t neg = svmul_f32_x(mask, param0, svsub_n_f32_x(mask, Exponent(mask, value), 1.0f));
            return svsel_f32(svcmplt_n_f32(mask, value, 0.0f), neg, value);
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationHswish>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            return SynetHswish32f(mask, value, param0, param1);
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationMish>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            svfloat32_t exp = svadd_n_f32_x(mask, Exponent(mask, value), 1.0f);
            svfloat32_t den = svadd_n_f32_x(mask, svmul_f32_x(mask, exp, exp), 1.0f);
            svfloat32_t tanh = svsub_f32_x(mask, svdup_n_f32(1.0f), svdiv_f32_x(mask, svdup_n_f32(2.0f), den));
            svfloat32_t mish = svmul_f32_x(mask, value, tanh);
            return svsel_f32(svcmpgt_f32(mask, value, param0), value, mish);
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationHardSigmoid>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            return SynetHardSigmoid32f(mask, value, param0, param1);
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationSwish>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            svfloat32_t exp = Exponent(mask, svneg_f32_x(mask, svmul_f32_x(mask, value, param0)));
            return svdiv_f32_x(mask, value, svadd_n_f32_x(mask, exp, 1.0f));
        }

        template<> SIMD_INLINE svfloat32_t Activate<::SimdConvolutionActivationGelu>(svfloat32_t value, const svfloat32_t& param0, const svfloat32_t& param1, size_t index, const svbool_t& mask)
        {
            svfloat32_t t = svmul_n_f32_x(mask, value, float(M_SQRT1_2));
            return svmul_f32_x(mask, svmul_n_f32_x(mask, t, float(M_SQRT1_2)), svadd_n_f32_x(mask, Erf(mask, t), 1.0f));
        }
    }
#endif
}
#endif

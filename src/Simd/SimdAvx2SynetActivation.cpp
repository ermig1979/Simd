/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#include "Simd/SimdArray.h"
#include "Simd/SimdPow.h"
#include "Simd/SimdExp.h"
#include "Simd/SimdBase.h"
#include "Simd/SimdSynet.h"

namespace Simd
{
#if defined(SIMD_AVX2_ENABLE) && defined(SIMD_SYNET_ENABLE)    
    namespace Avx2
    {
        template<bool align> SIMD_INLINE void SynetElu32f(const float * src, const Avx2::Exp & exp, __m256 alpha, float * dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, exp.Elu(Avx::Load<align>(src + offset), alpha));
        }

        template<bool align> void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            __m256 _alpha = _mm256_set1_ps(alpha[0]);
            Avx2::Exp exp;
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetElu32f<align>(src, exp, _alpha, dst, i + 0 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 1 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 2 * F);
                SynetElu32f<align>(src, exp, _alpha, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetElu32f<align>(src, exp, _alpha, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetElu32f(src[i], alpha[0]);
        }

        void SynetElu32f(const float * src, size_t size, const float * alpha, float * dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetElu32f<true>(src, size, alpha, dst);
            else
                SynetElu32f<false>(src, size, alpha, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetMish32f(const float* src, __m256 threshold, float* dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, Mish(Avx::Load<align>(src + offset), threshold));
        }

        template<bool align> void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            __m256 _threshold = _mm256_set1_ps(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetMish32f<align>(src, _threshold, dst, i + 0 * F);
                SynetMish32f<align>(src, _threshold, dst, i + 1 * F);
                SynetMish32f<align>(src, _threshold, dst, i + 2 * F);
                SynetMish32f<align>(src, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetMish32f<align>(src, _threshold, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetMish32f(src[i], threshold[0]);
        }

        void SynetMish32f(const float* src, size_t size, const float* threshold, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetMish32f<true>(src, size, threshold, dst);
            else
                SynetMish32f<false>(src, size, threshold, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetSigmoid32f(const float* src, const Avx2::Exp& exp, float* dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, exp.Sigmoid(Avx::Load<align>(src + offset)));
        }

        template<bool align> void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSigmoid32f<align>(src, exp, dst, i + 0 * F);
                SynetSigmoid32f<align>(src, exp, dst, i + 1 * F);
                SynetSigmoid32f<align>(src, exp, dst, i + 2 * F);
                SynetSigmoid32f<align>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSigmoid32f<align>(src, exp, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSigmoid32f(src[i], slope[0]);
        }

        void SynetSigmoid32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSigmoid32f<true>(src, size, slope, dst);
            else
                SynetSigmoid32f<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetSoftplus32f(const float* src, __m256 beta, __m256 threshold, float* dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, Softplus(Avx::Load<align>(src + offset), beta, threshold));
        }

        template<bool align> void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            __m256 _beta = _mm256_set1_ps(beta[0]);
            __m256 _threshold = _mm256_set1_ps(threshold[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i + 0 * F);
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i + 1 * F);
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i + 2 * F);
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSoftplus32f<align>(src, _beta, _threshold, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSoftplus32f(src[i], beta[0], threshold[0]);
        }

        void SynetSoftplus32f(const float* src, size_t size, const float* beta, const float* threshold, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSoftplus32f<true>(src, size, beta, threshold, dst);
            else
                SynetSoftplus32f<false>(src, size, beta, threshold, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetSwish32f(const float* src, const Avx2::Exp& exp, float* dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, exp.Swish(Avx::Load<align>(src + offset)));
        }

        template<bool align> void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            Exp exp(-slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetSwish32f<align>(src, exp, dst, i + 0 * F);
                SynetSwish32f<align>(src, exp, dst, i + 1 * F);
                SynetSwish32f<align>(src, exp, dst, i + 2 * F);
                SynetSwish32f<align>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetSwish32f<align>(src, exp, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetSwish32f(src[i], slope[0]);
        }

        void SynetSwish32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetSwish32f<true>(src, size, slope, dst);
            else
                SynetSwish32f<false>(src, size, slope, dst);
        }

        //---------------------------------------------------------------------

        template<bool align> SIMD_INLINE void SynetTanh32f(const float* src, const Avx2::Exp& exp, float* dst, size_t offset)
        {
            Avx::Store<align>(dst + offset, exp.Tanh(Avx::Load<align>(src + offset)));
        }

        template<bool align> void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (align)
                assert(Aligned(src) && Aligned(dst));

            Exp exp(-2.0f*slope[0]);
            size_t sizeF = AlignLo(size, F);
            size_t sizeQF = AlignLo(size, QF);
            size_t i = 0;
            for (; i < sizeQF; i += QF)
            {
                SynetTanh32f<align>(src, exp, dst, i + 0 * F);
                SynetTanh32f<align>(src, exp, dst, i + 1 * F);
                SynetTanh32f<align>(src, exp, dst, i + 2 * F);
                SynetTanh32f<align>(src, exp, dst, i + 3 * F);
            }
            for (; i < sizeF; i += F)
                SynetTanh32f<align>(src, exp, dst, i);
            for (; i < size; ++i)
                dst[i] = Base::SynetTanh32f(src[i], slope[0]);
        }

        void SynetTanh32f(const float* src, size_t size, const float* slope, float* dst)
        {
            if (Aligned(src) && Aligned(dst))
                SynetTanh32f<true>(src, size, slope, dst);
            else
                SynetTanh32f<false>(src, size, slope, dst);
        }
    }
#endif// SIMD_AVX2_ENABLE
}

/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2019 Yermalayeu Ihar.
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
#include "Simd/SimdDescrInt.h"
#include "Simd/SimdDescrIntCommon.h"
#include "Simd/SimdCpu.h"

namespace Simd
{
#ifdef SIMD_NEON_ENABLE    
    namespace Neon
    {
        static void MinMax32f(const float* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            float32x4_t _min = vdupq_n_f32(FLT_MAX);
            float32x4_t _max = vdupq_n_f32(-FLT_MAX);
            size_t i = 0;
            if (Aligned(src))
            {
                for (; i < size; i += 4)
                {
                    float32x4_t _src = Load<true>(src + i);
                    _min = vminq_f32(_src, _min);
                    _max = vmaxq_f32(_src, _max);
                }
            }
            else
            {
                for (; i < size; i += 4)
                {
                    float32x4_t _src = Load<false>(src + i);
                    _min = vminq_f32(_src, _min);
                    _max = vmaxq_f32(_src, _max);
                }
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        static void MinMax16f(const uint16_t* src, size_t size, float& min, float& max)
        {
            assert(size % 8 == 0);
            float32x4_t _min = vdupq_n_f32(FLT_MAX);
            float32x4_t _max = vdupq_n_f32(-FLT_MAX);
            size_t i = 0;
            if (Aligned(src))
            {
                for (; i < size; i += 4)
                {
                    float32x4_t _src = vcvt_f32_f16((float16x4_t)LoadHalf<true>(src + i));
                    _min = vminq_f32(_src, _min);
                    _max = vmaxq_f32(_src, _max);
                }
            }
            else
            {
                for (; i < size; i += 4)
                {
                    float32x4_t _src = vcvt_f32_f16((float16x4_t)LoadHalf<false>(src + i));
                    _min = vminq_f32(_src, _min);
                    _max = vmaxq_f32(_src, _max);
                }
            }
            MinVal32f(_min, min);
            MaxVal32f(_max, max);
        }

        //-------------------------------------------------------------------------------------------------

        DescrInt::DescrInt(size_t size, size_t depth)
            : Base::DescrInt(size, depth)
        {
            _minMax32f = MinMax32f;
            _minMax16f = MinMax16f;
            _encode32f = GetEncode32f(_depth);
            if (_depth >= 8)
                _encode16f = GetEncode16f(_depth);

            //_decode32f = GetDecode32f(_depth);
            //_decode16f = GetDecode16f(_depth);

            //_cosineDistance = GetCosineDistance(_depth);
            //_macroCosineDistancesDirect = GetMacroCosineDistancesDirect(_depth);
            //_microMd = 2;
            //_microNd = 4;

            //_unpackNormA = UnpackNormA;
            //_unpackNormB = UnpackNormB;
            //_unpackDataA = GetUnpackData(_depth, false);
            //_unpackDataB = GetUnpackData(_depth, true);
            //_macroCosineDistancesUnpack = GetMacroCosineDistancesUnpack(_depth);
            _unpSize = _size * (_depth == 8 ? 2 : 1);
            _microMu = _depth == 8 ? 6 : 5;
            _microNu = 8;
}

        void DescrInt::CosineDistancesMxNa(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
        {
            Base::DescrInt::CosineDistancesMxNa(M, N, A, B, distances);
            //if (_unpSize * _microNu > Base::AlgCacheL1() || N * 2 < _microNu || _depth == 8)
            //    CosineDistancesDirect(M, N, A, B, distances);
            //else
            //    CosineDistancesUnpack(M, N, A, B, distances);
        }

        void DescrInt::CosineDistancesMxNp(size_t M, size_t N, const uint8_t* A, const uint8_t* B, float* distances) const
        {
            Array8ucp a(M);
            for (size_t i = 0; i < M; ++i)
                a[i] = A + i * _encSize;
            Array8ucp b(N);
            for (size_t j = 0; j < N; ++j)
                b[j] = B + j * _encSize;
            CosineDistancesMxNa(M, N, a.data, b.data, distances);
        }

        void DescrInt::CosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
        {
            const size_t L2 = Base::AlgCacheL2();
            size_t mN = AlignLoAny(L2 / _encSize, _microNd);
            size_t mM = AlignLoAny(L2 / _encSize, _microMd);
            for (size_t i = 0; i < M; i += mM)
            {
                size_t dM = Simd::Min(M, i + mM) - i;
                for (size_t j = 0; j < N; j += mN)
                {
                    size_t dN = Simd::Min(N, j + mN) - j;
                    _macroCosineDistancesDirect(dM, dN, A + i, B + j, _size, distances + i * N + j, N);
                }
            }
        }

        void DescrInt::CosineDistancesUnpack(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const
        {
            size_t macroM = AlignLoAny(Base::AlgCacheL2() / _unpSize, _microMu);
            size_t macroN = AlignLoAny(Base::AlgCacheL3() / _unpSize, _microNu);
            size_t sizeA = Min(macroM, M), sizeB = AlignHi(Min(macroN, N), _microNu);
            Array8u dA(sizeA * _unpSize), dB(sizeB * _unpSize);
            Array32f nA(sizeA * 4), nB(sizeB * 4);
            for (size_t i = 0; i < M; i += macroM)
            {
                size_t dM = Simd::Min(M, i + macroM) - i;
                _unpackNormA(dM, A + i, nA.data, 1);
                _unpackDataA(dM, A + i, _size, dA.data, _unpSize);
                for (size_t j = 0; j < N; j += macroN)
                {
                    size_t dN = Simd::Min(N, j + macroN) - j;
                    _unpackNormB(dN, B + j, nB.data, dN);
                    _unpackDataB(dN, B + j, _size, dB.data, 1);
                    _macroCosineDistancesUnpack(dM, dN, _size, dA.data, nA.data, dB.data, nB.data, distances + i * N + j, N);
                }
            }
        }

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth)
        {
            if (!Base::DescrInt::Valid(size, depth))
                return NULL;
            return new Neon::DescrInt(size, depth);
        }
    }
#endif// SIMD_NEON_ENABLE
}

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
#ifndef __SimdDescrInt_h__
#define __SimdDescrInt_h__

#include "Simd/SimdMemory.h"

#define SIMD_DESCR_INT_EPS 0.000001f

namespace Simd
{
    namespace Base
    {
        class DescrInt : Deletable
        {
        public:
            static bool Valid(size_t size, size_t depth);

            DescrInt(size_t size, size_t depth);

            size_t DecodedSize() const { return _size; }
            size_t EncodedSize() const { return _encSize; }

            virtual void Encode(const float* src, uint8_t* dst) const;
            virtual void Decode(const uint8_t* src, float* dst) const;

            virtual void CosineDistance(const uint8_t* a, const uint8_t* b, float* distance) const;
            virtual void CosineDistancesMxNa(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const;
            virtual void CosineDistancesMxNp(size_t M, size_t N, const uint8_t* A, const uint8_t* B, float* distances) const;

            virtual void VectorNorm(const uint8_t* a, float* norm) const;
            virtual void VectorNormsNa(size_t N, const uint8_t* const* A, float* norms) const;
            virtual void VectorNormsNp(size_t N, const uint8_t* A, float* norms) const;

        protected:
            typedef void (*MinMaxPtr)(const float* src, size_t size, float &min, float &max);
            typedef void (*EncodePtr)(const float* src, float scale, float min, size_t size, uint8_t* dst);
            typedef void (*DecodePtr)(const uint8_t * src, float scale, float shift, size_t size, float* dst);
            typedef void (*CosineDistancePtr)(const uint8_t* a, float aScale, float aShift, const uint8_t* b, float bScale, float bShift, size_t size, float* distance);
            typedef void (*VectorNormPtr)(const uint8_t* src, float scale, float shift, size_t size, float* norm);

            MinMaxPtr _minMax;
            EncodePtr _encode;
            DecodePtr _decode;
            CosineDistancePtr _cosineDistance;
            VectorNormPtr _vectorNorm;
            size_t _size, _depth, _encSize;
            float _range;
        };

        //-------------------------------------------------------------------------------------------------

        void * DescrIntInit(size_t size, size_t depth);
    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
    }
#endif
}
#endif//__SimdDescrInt_h__

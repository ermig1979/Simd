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
        class DescrInt : public Deletable
        {
        public:
            static bool Valid(size_t size, size_t depth);

            DescrInt(size_t size, size_t depth);

            size_t DecodedSize() const { return _size; }
            size_t EncodedSize() const { return _encSize; }

            void Encode32f(const float* src, uint8_t* dst) const;
            void Encode16f(const uint16_t* src, uint8_t* dst) const;
            void Decode32f(const uint8_t* src, float* dst) const;
            void Decode16f(const uint8_t* src, uint16_t* dst) const;

            void CosineDistance(const uint8_t* a, const uint8_t* b, float* distance) const;
            virtual void CosineDistancesMxNa(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const;
            virtual void CosineDistancesMxNp(size_t M, size_t N, const uint8_t* A, const uint8_t* B, float* distances) const;

            void VectorNorm(const uint8_t* a, float* norm) const;

            typedef void (*Encode32fPtr)(const float* src, float scale, float min, size_t size, int32_t &sum, int32_t& sqsum, uint8_t* dst);
            typedef void (*Encode16fPtr)(const uint16_t* src, float scale, float min, size_t size, int32_t& sum, int32_t& sqsum, uint8_t* dst);
            typedef void (*Decode32fPtr)(const uint8_t * src, float scale, float shift, size_t size, float* dst);
            typedef void (*Decode16fPtr)(const uint8_t* src, float scale, float shift, size_t size, uint16_t* dst);
            typedef void (*CosineDistancePtr)(const uint8_t* a, const uint8_t* b, size_t size, float* distance);

        protected:
            typedef void (*MinMax32fPtr)(const float* src, size_t size, float &min, float &max);
            typedef void (*MinMax16fPtr)(const uint16_t* src, size_t size, float& min, float& max);

            MinMax32fPtr _minMax32f;
            MinMax16fPtr _minMax16f;
            Encode32fPtr _encode32f;
            Encode16fPtr _encode16f;
            Decode32fPtr _decode32f;
            Decode16fPtr _decode16f;
            CosineDistancePtr _cosineDistance;
            size_t _size, _depth, _encSize;
            float _range;
        };

        //-------------------------------------------------------------------------------------------------

        void * DescrIntInit(size_t size, size_t depth);
    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        class DescrInt : public Base::DescrInt
        {
        public:
            DescrInt(size_t size, size_t depth);

            virtual void CosineDistancesMxNa(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const;
            virtual void CosineDistancesMxNp(size_t M, size_t N, const uint8_t* A, const uint8_t* B, float* distances) const;

            typedef void (*MacroCosineDistancesDirectPtr)(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, size_t size, float* distances, size_t stride);
            typedef void (*UnpackDataPtr)(size_t count, const uint8_t* const* src, size_t size, uint8_t* dst, size_t stride);
            typedef void (*MacroCosineDistancesUnpackPtr)(size_t M, size_t N, size_t K, const uint8_t* ad, const float * an, const uint8_t* bd, const float* bn, float* distances, size_t stride);

        protected:
            typedef void (*UnpackNormPtr)(size_t count, const uint8_t* const* src, float* dst, size_t stride);

            void CosineDistancesDirect(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const;

            MacroCosineDistancesDirectPtr _macroCosineDistancesDirect;
            size_t _microMd, _microNd;

            void CosineDistancesUnpack(size_t M, size_t N, const uint8_t* const* A, const uint8_t* const* B, float* distances) const;

            UnpackNormPtr _unpackNormA, _unpackNormB;
            UnpackDataPtr _unpackDataA, _unpackDataB;
            MacroCosineDistancesUnpackPtr _macroCosineDistancesUnpack;
            size_t _microMu, _microNu, _unpSize;
        };

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Encode32fPtr GetEncode32f(size_t depth);
        Base::DescrInt::Encode16fPtr GetEncode16f(size_t depth);

        Base::DescrInt::Decode32fPtr GetDecode32f(size_t depth);
        Base::DescrInt::Decode16fPtr GetDecode16f(size_t depth);

        Base::DescrInt::CosineDistancePtr GetCosineDistance(size_t depth);
        Sse41::DescrInt::MacroCosineDistancesDirectPtr GetMacroCosineDistancesDirect(size_t depth);

        Sse41::DescrInt::UnpackDataPtr GetUnpackData(size_t depth, bool transpose);
        Sse41::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth);

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth);
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        class DescrInt : public Sse41::DescrInt
        {
        public:
            DescrInt(size_t size, size_t depth);
        };

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Encode32fPtr GetEncode32f(size_t depth);
        Base::DescrInt::Encode16fPtr GetEncode16f(size_t depth);

        Base::DescrInt::Decode32fPtr GetDecode32f(size_t depth);
        Base::DescrInt::Decode16fPtr GetDecode16f(size_t depth);

        Base::DescrInt::CosineDistancePtr GetCosineDistance(size_t depth);
        Sse41::DescrInt::MacroCosineDistancesDirectPtr GetMacroCosineDistancesDirect(size_t depth);

        Sse41::DescrInt::UnpackDataPtr GetUnpackData(size_t depth, bool transpose);
        Sse41::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth);

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        class DescrInt : public Avx2::DescrInt
        {
        public:
            DescrInt(size_t size, size_t depth);
        };

        //-------------------------------------------------------------------------------------------------

        Base::DescrInt::Encode32fPtr GetEncode32f(size_t depth);
        Base::DescrInt::Encode16fPtr GetEncode16f(size_t depth);

        Base::DescrInt::Decode32fPtr GetDecode32f(size_t depth);
        Base::DescrInt::Decode16fPtr GetDecode16f(size_t depth);

        Base::DescrInt::CosineDistancePtr GetCosineDistance(size_t depth);
        Sse41::DescrInt::MacroCosineDistancesDirectPtr GetMacroCosineDistancesDirect(size_t depth);

        Sse41::DescrInt::UnpackDataPtr GetUnpackData(size_t depth, bool transpose);
        Sse41::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth);

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth);
    }
#endif

#ifdef SIMD_AVX512VNNI_ENABLE
    namespace Avx512vnni
    {
        class DescrInt : public Avx512bw::DescrInt
        {
        public:
            DescrInt(size_t size, size_t depth);
        };

        //-------------------------------------------------------------------------------------------------

        Sse41::DescrInt::MacroCosineDistancesUnpackPtr GetMacroCosineDistancesUnpack(size_t depth);

        //-------------------------------------------------------------------------------------------------

        void* DescrIntInit(size_t size, size_t depth);
    }
#endif
}
#endif//__SimdDescrInt_h__

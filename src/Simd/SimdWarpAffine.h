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
#ifndef __SimdWarpAffine_h__
#define __SimdWarpAffine_h__

#include "Simd/SimdArray.h"
#include "Simd/SimdMath.h"

#include "Simd/SimdPoint.hpp"

namespace Simd
{
    struct WarpAffParam
    {
        static const int BorderSizeMax = 4 * 1;

        SimdWarpAffineFlags flags;
        float mat[6], inv[6];
        uint8_t border[BorderSizeMax];
        size_t srcW, srcH, srcS, dstW, dstH, dstS, channels, align;

        WarpAffParam(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border, size_t align);

        bool Valid() const
        {
            return channels >= 1 && channels <= 4 && srcH * srcS <= 0x100000000 &&
                (inv[0] != 0.0f || inv[1] != 0.0f || inv[3] != 0.0f || inv[4] != 0.0f);
        }

        bool IsNearest() const
        {
            return (flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpNearest;
        }

        bool IsByteBilinear() const
        {
            return (flags & SimdWarpAffineInterpMask) == SimdWarpAffineInterpBilinear && (SimdWarpAffineChannelMask & flags) == SimdWarpAffineChannelByte;
        }

        bool NeedFill() const
        {
            return (flags & SimdWarpAffineBorderMask) == SimdWarpAffineBorderConstant;
        }

        size_t ChannelSize() const
        {
            switch (SimdWarpAffineChannelMask & flags)
            {
            case SimdWarpAffineChannelByte: return 1;
            default:
                assert(0); return 0;
            }
        }

        size_t PixelSize() const
        {
            return ChannelSize() * channels;
        }
    };

    //-------------------------------------------------------------------------------------------------

    class WarpAffine : Deletable
    {
    public:
        WarpAffine(const WarpAffParam & param)
            : _param(param)
            , _first(true)
        {
        }

        virtual void Run(const uint8_t * src, uint8_t * dst) = 0;

    protected:
        WarpAffParam _param;
        bool _first;
    };

    //-------------------------------------------------------------------------------------------------

    namespace Base
    {
        typedef Simd::Point<float> Point;
        
        //-------------------------------------------------------------------------------------------------

        class WarpAffineNearest : public WarpAffine
        {
        public:
            typedef void(*RunPtr)(const WarpAffParam& p, const int32_t* beg, const int32_t* end, const uint8_t* src, uint8_t* dst, uint32_t* offs);

            WarpAffineNearest(const WarpAffParam& param);

            virtual void Run(const uint8_t* src, uint8_t* dst);

        protected:
            void Init();

            virtual void SetRange(const Base::Point * points);

            Array32i _beg, _end;
            Array32u _buf;
            RunPtr _run;
        };

        //-------------------------------------------------------------------------------------------------

        class WarpAffineByteBilinear : public WarpAffine
        {
        public:
            typedef void(*RunPtr)(const WarpAffParam& p, const int* ib, const int* ie, const int* ob, const int* oe, const uint8_t* src, uint8_t* dst, uint8_t* buf);

            WarpAffineByteBilinear(const WarpAffParam & param);

            virtual void Run(const uint8_t * src, uint8_t * dst);

        protected:
            void Init();

            virtual void SetRange(const Base::Point* rect, int* beg, int* end, const int* lo, const int* hi);

            Array32i _range;
            int *_ib, *_ie, *_ob, *_oe;
            Array8u _buf;
            RunPtr _run;
        };

        //-------------------------------------------------------------------------------------------------

        void * WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border);
    }

#ifdef SIMD_SSE41_ENABLE
    namespace Sse41
    {
        class WarpAffineNearest : public Base::WarpAffineNearest
        {
        public:
            WarpAffineNearest(const WarpAffParam& param);

        protected:
            virtual void SetRange(const Base::Point* points);
        };

        //-------------------------------------------------------------------------------------------------

        class WarpAffineByteBilinear : public Base::WarpAffineByteBilinear
        {
        public:
            WarpAffineByteBilinear(const WarpAffParam& param);

        protected:
            virtual void SetRange(const Base::Point* rect, int* beg, int* end, const int* lo, const int* hi);
        };

        //-------------------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border);
    }
#endif

#ifdef SIMD_AVX2_ENABLE
    namespace Avx2
    {
        class WarpAffineNearest : public Sse41::WarpAffineNearest
        {
        public:
            WarpAffineNearest(const WarpAffParam& param);

        protected:
            virtual void SetRange(const Base::Point* points);
        };

        //-------------------------------------------------------------------------------------------------
 
        class WarpAffineByteBilinear : public Sse41::WarpAffineByteBilinear
        {
        public:
            WarpAffineByteBilinear(const WarpAffParam& param);

        protected:
            virtual void SetRange(const Base::Point* rect, int* beg, int* end, const int* lo, const int* hi);
        };

        //-------------------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border);
    }
#endif

#ifdef SIMD_AVX512BW_ENABLE
    namespace Avx512bw
    {
        class WarpAffineNearest : public Avx2::WarpAffineNearest
        {
        public:
            WarpAffineNearest(const WarpAffParam& param);

        protected:
            virtual void SetRange(const Base::Point* points);
        };

        //-------------------------------------------------------------------------------------------------

        void* WarpAffineInit(size_t srcW, size_t srcH, size_t srcS, size_t dstW, size_t dstH, size_t dstS, size_t channels, const float* mat, SimdWarpAffineFlags flags, const uint8_t* border);
    }
#endif
}
#endif//__SimdWarpAffine_h__

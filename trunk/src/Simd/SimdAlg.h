/*
* Simd Library.
*
* Copyright (c) 2011-2013 Yermalayeu Ihar.
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
#ifndef __SimdAlg_h__
#define __SimdAlg_h__

#include "Simd/SimdLib.h"
#include "Simd/SimdTypes.h"

namespace Simd
{
    SIMD_INLINE void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
        size_t width, size_t height, uint64_t * sum)
    {
        SimdAbsDifferenceSum(a, aStride, b, bStride, width, height, sum);
    }

    SIMD_INLINE void AbsDifferenceSum(const uchar *a, size_t aStride, const uchar *b, size_t bStride, 
        const uchar *mask, size_t maskStride, uchar index, size_t width, size_t height, uint64_t * sum)
    {
        SimdAbsDifferenceSumMasked(a, aStride, b, bStride, mask, maskStride, index, width, height, sum);
    }

    SIMD_INLINE void AbsGradientSaturatedSum(const uchar * src, size_t srcStride, size_t width, size_t height, uchar * dst, size_t dstStride)
    {
        SimdAbsGradientSaturatedSum(src, srcStride, width, height, dst, dstStride);
    }

    SIMD_INLINE void AddFeatureDifference(const uchar * value, size_t valueStride, size_t width, size_t height, 
        const uchar * lo, size_t loStride, const uchar * hi, size_t hiStride,
        ushort weight, uchar * difference, size_t differenceStride)
    {
        SimdAddFeatureDifference(value, valueStride, width, height, lo, loStride, hi, hiStride, weight, difference, differenceStride);
    }

    SIMD_INLINE void BackgroundGrowRangeSlow(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
    {
        SimdBackgroundGrowRangeSlow(value, valueStride, width, height, lo, loStride, hi, hiStride);
    }

    SIMD_INLINE void BackgroundGrowRangeFast(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
    {
        SimdBackgroundGrowRangeFast(value, valueStride, width, height, lo, loStride, hi, hiStride);
    }

    SIMD_INLINE void BackgroundIncrementCount(const uchar * value, size_t valueStride, size_t width, size_t height,
        const uchar * loValue, size_t loValueStride, const uchar * hiValue, size_t hiValueStride,
        uchar * loCount, size_t loCountStride, uchar * hiCount, size_t hiCountStride)
    {
        SimdBackgroundIncrementCount(value, valueStride, width, height, loValue, loValueStride, hiValue, hiValueStride, 
            loCount, loCountStride, hiCount, hiCountStride);
    }

    SIMD_INLINE void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
        uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
        uchar * hiValue, size_t hiValueStride, uchar threshold)
    {
        SimdBackgroundAdjustRange(loCount, loCountStride, width, height, loValue, loValueStride,
                hiCount, hiCountStride, hiValue, hiValueStride, threshold);
    }

    SIMD_INLINE void BackgroundAdjustRange(uchar * loCount, size_t loCountStride, size_t width, size_t height, 
        uchar * loValue, size_t loValueStride, uchar * hiCount, size_t hiCountStride, 
        uchar * hiValue, size_t hiValueStride, uchar threshold, const uchar * mask, size_t maskStride)
    {
        SimdBackgroundAdjustRangeMasked(loCount, loCountStride, width, height, loValue, loValueStride, 
                hiCount, hiCountStride, hiValue, hiValueStride, threshold, mask, maskStride);
    }

    SIMD_INLINE void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * lo, size_t loStride, uchar * hi, size_t hiStride)
    {
        SimdBackgroundShiftRange(value, valueStride, width, height, lo, loStride, hi, hiStride);
    }

    SIMD_INLINE void BackgroundShiftRange(const uchar * value, size_t valueStride, size_t width, size_t height,
        uchar * lo, size_t loStride, uchar * hi, size_t hiStride, const uchar * mask, size_t maskStride)
    {
        SimdBackgroundShiftRangeMasked(value, valueStride, width, height, lo, loStride, hi, hiStride, mask, maskStride);
    }

    SIMD_INLINE void BackgroundInitMask(const uchar * src, size_t srcStride, size_t width, size_t height,
        uchar index, uchar value, uchar * dst, size_t dstStride)
    {
        SimdBackgroundInitMask(src, srcStride, width, height, index, value, dst, dstStride);
    }
}

#include "Simd/SimdBgraToBgr.h"
#include "Simd/SimdBgraToGray.h"
#include "Simd/SimdBgrToBgra.h"
#include "Simd/SimdBgrToGray.h"
#include "Simd/SimdBinarization.h"
#include "Simd/SimdCopy.h"
#include "Simd/SimdCrc32.h"
#include "Simd/SimdDeinterleaveUv.h"
#include "Simd/SimdDrawing.h"
#include "Simd/SimdEdgeBackground.h"
#include "Simd/SimdFill.h"
#include "Simd/SimdGaussianBlur3x3.h"
#include "Simd/SimdGrayToBgra.h"
#include "Simd/SimdHistogram.h"
#include "Simd/SimdInterleaveBgra.h"
#include "Simd/SimdLbp.h"
#include "Simd/SimdMedianFilterSquare3x3.h"
#include "Simd/SimdMedianFilterSquare5x5.h"
#include "Simd/SimdOperation.h"
#include "Simd/SimdReduceGray2x2.h"
#include "Simd/SimdReduceGray3x3.h"
#include "Simd/SimdReduceGray4x4.h"
#include "Simd/SimdReduceGray5x5.h"
#include "Simd/SimdResizeBilinear.h"
#include "Simd/SimdShiftBilinear.h"
#include "Simd/SimdSquaredDifferenceSum.h"
#include "Simd/SimdStatistic.h"
#include "Simd/SimdStretchGray2x2.h"
#include "Simd/SimdTexture.h"
#include "Simd/SimdYuvToBgra.h"
#include "Simd/SimdYuvToBgr.h"
#include "Simd/SimdYuvToHue.h"

#endif//__SimdAlg_h__

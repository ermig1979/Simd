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

#include "Simd/SimdView.h"

#ifndef __SimdUtils_h__
#define __SimdUtils_h__

namespace Simd
{
    void AbsDifferenceSum(const View & a, const View & b, uint64_t & sum);
    void AbsDifferenceSum(const View & a, const View & b, const View & mask, uchar index, uint64_t & sum);

    void AbsGradientSaturatedSum(const View & src, View & dst);

    void AbsSecondDerivativeHistogram(const View & src, size_t step, size_t indent, uint * histogram);

    void AddFeatureDifference(const View & value, const View & lo, const View & hi, ushort weight, View & difference);

    void BackgroundGrowRangeSlow(const View & value, View & lo, View & hi);
    void BackgroundGrowRangeFast(const View & value, View & lo, View & hi);
    void BackgroundIncrementCount(const View & value, const View & loValue, const View & hiValue, View & loCount, View & hiCount);
    void BackgroundAdjustRange(View & loCount, View & loValue, View & hiCount, View & hiValue, uchar threshold);
    void BackgroundAdjustRange(View & loCount, View & loValue, View & hiCount, View & hiValue, uchar threshold, const View & mask);
    void BackgroundShiftRange(const View & value, View & lo, View & hi);
    void BackgroundShiftRange(const View & value, View & lo, View & hi, const View & mask);
    void BackgroundInitMask(const View & src, uchar index, uchar value, View & dst);

    void BgraToBgr(const View & bgra, View & bgr);

    void BgraToGray(const View & bgra, View & gray);

    void BgrToBgra(const View & bgr, View & bgra, uchar alpha = 0xFF);

    void BgrToGray(const View & bgr, View & gray);

    void Binarization(const View & src, uchar value, uchar positive, uchar negative, View & dst, CompareType compareType);
    void AveragingBinarization(const View & src, uchar value, size_t neighborhood, uchar threshold, uchar positive, uchar negative, View & dst, CompareType compareType);

    void Copy(const View & src, View & dst);
    void CopyFrame(const View & src, const Rectangle<ptrdiff_t> & frame, View & dst);

    void DeinterleaveUv(const View & uv, View & u, View & v);

    void EdgeBackgroundGrowRangeSlow(const View & value, View & background);
    void EdgeBackgroundGrowRangeFast(const View & value, View & background);
    void EdgeBackgroundIncrementCount(const View & value, const View & backgroundValue, View & backgroundCount);
    void EdgeBackgroundAdjustRange(View & backgroundCount, View & backgroundValue, uchar threshold);
    void EdgeBackgroundAdjustRange(View & backgroundCount, View & backgroundValue, uchar threshold, const View & mask);
    void EdgeBackgroundShiftRange(const View & value, View & background);
    void EdgeBackgroundShiftRange(const View & value, View & background, const View & mask);

    void Fill(View & dst, uchar value);
    void FillBgra(View & dst, uchar blue, uchar green, uchar red, uchar alpha = 0xFF);

    void GaussianBlur3x3(const View & src, View & dst);

    void MedianFilterSquare3x3(const View & src, View & dst);

    void MedianFilterSquare5x5(const View & src, View & dst);

    void Operation(const View & a, const View & b, View & dst, OperationType type);

    void ReduceGray2x2(const View & src, View & dst);

    void ReduceGray3x3(const View & src, View & dst, bool compensation = true);

    void ReduceGray4x4(const View & src, View & dst);

    void ReduceGray5x5(const View & src, View & dst, bool compensation = true);

    void ResizeBilinear(const View & src, View & dst);

    void ShiftBilinear(const View & src, const View & bkg, const Point<double> & shift, const Rectangle<ptrdiff_t> & crop, View & dst);

    void SquaredDifferenceSum(const View & a, const View & b, uint64_t & sum);
    void SquaredDifferenceSum(const View & a, const View & b, const View & mask, uchar index, uint64_t & sum);

    void GetStatistic(const View & src, uchar * min, uchar * max, uchar * average);
    void GetMoments(const View & mask, uchar index, uint64_t * area, uint64_t * x, uint64_t * y, uint64_t * xx, uint64_t * xy, uint64_t * yy);
    void GetRowSums(const View & src, uint * sums);
    void GetColSums(const View & src, uint * sums);
    void GetAbsDyRowSums(const View & src, uint * sums);
    void GetAbsDxColSums(const View & src, uint * sums);

    void StretchGray2x2(const View & src, View & dst);

    void TextureBoostedSaturatedGradient(const View & src, uchar saturation, uchar boost, View &  dx, View & dy);
    void TextureBoostedUv(const View & src, uchar boost, View & dst);
    void TextureGetDifferenceSum(const View & src, const View & lo, const View & hi, int64_t * sum);
    void TexturePerformCompensation(const View & src, int shift, View & dst);

    void Yuv444ToBgr(const View & y, const View & u, const View & v, View & bgr);
    void Yuv420ToBgr(const View & y, const View & u, const View & v, View & bgr);

    void Yuv444ToBgra(const View & y, const View & u, const View & v, View & bgra, uchar alpha = 0xFF);
    void Yuv420ToBgra(const View & y, const View & u, const View & v, View & bgra, uchar alpha = 0xFF);

    void Yuv444ToHue(const View & y, const View & u, const View & v, View & hue);
    void Yuv420ToHue(const View & y, const View & u, const View & v, View & hue);
}

#endif//__SimdUtils_h__


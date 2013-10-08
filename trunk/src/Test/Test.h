/*
* Simd Library Tests.
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
#ifndef __Test_h__
#define __Test_h__

namespace Test
{
    bool ReduceGray2x2Test();
    bool ReduceGray3x3Test();
    bool ReduceGray4x4Test();
    bool ReduceGray5x5Test();

    bool Crc32Test();

    bool BgraToGrayTest();

    bool BgrToGrayTest();

    bool Yuv444ToHueTest();
    bool Yuv420ToHueTest();

    bool Yuv444ToBgraTest();
    bool Yuv420ToBgraTest();

    bool MedianFilterSquare3x3Test();
    bool MedianFilterSquare5x5Test();
    bool GaussianBlur3x3Test();
    bool AbsGradientSaturatedSumTest();
    bool LbpEstimateTest();

    bool SquareDifferenceSumTest();
    bool MaskedSquareDifferenceSumTest();
    bool AbsDifferenceSumTest();
    bool MaskedAbsDifferenceSumTest();

    bool ResizeBilinearTest();

    bool DeinterleaveUvTest();

    bool OperationTest();

    bool AbsSecondDerivativeHistogramTest();

    bool BinarizationTest();
    bool AveragingBinarizationTest();

    bool ShiftBilinearTest();

    bool GetStatisticTest();
    bool GetMomentsTest();
    bool GetRowSumsTest();
    bool GetColSumsTest();
    bool GetAbsDyRowSumsTest();
    bool GetAbsDxColSumsTest();

    bool StretchGray2x2Test();

    bool BackgroundGrowRangeSlowTest();
    bool BackgroundGrowRangeFastTest();
    bool BackgroundIncrementCountTest();
    bool BackgroundAdjustRangeTest();
    bool MaskedBackgroundAdjustRangeTest();
    bool BackgroundShiftRangeTest();
    bool MaskedBackgroundShiftRangeTest();
    bool BackgroundInitMaskTest();

    bool EdgeBackgroundGrowRangeSlowTest();
    bool EdgeBackgroundGrowRangeFastTest();
    bool EdgeBackgroundIncrementCountTest();
    bool EdgeBackgroundAdjustRangeTest();
    bool MaskedEdgeBackgroundAdjustRangeTest();
    bool EdgeBackgroundShiftRangeTest();
    bool MaskedEdgeBackgroundShiftRangeTest();

    bool AddFeatureDifferenceTest();

    bool TextureBoostedSaturatedGradientTest();
    bool TextureBoostedUvTest();
    bool TextureGetDifferenceSumTest();
    bool TexturePerformCompensationTest();

    bool FillBgraTest();

    bool GrayToBgraTest();
}
#endif//__Test_h__

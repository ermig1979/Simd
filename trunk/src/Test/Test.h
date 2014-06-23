/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar.
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
    /***************************** Auto Tests: *******************************/

    bool ReduceGray2x2AutoTest();
    bool ReduceGray3x3AutoTest();
    bool ReduceGray4x4AutoTest();
    bool ReduceGray5x5AutoTest();

    bool Crc32cAutoTest();

    bool BayerToBgrAutoTest();

    bool BayerToBgraAutoTest();

    bool BgraToBayerAutoTest();

    bool BgraToBgrAutoTest();

    bool BgraToGrayAutoTest();

    bool BgrToBayerAutoTest();

    bool BgrToBgraAutoTest();
    bool Bgr48pToBgra32AutoTest();

    bool BgrToGrayAutoTest();

    bool Yuv444pToHueAutoTest();
    bool Yuv420pToHueAutoTest();

    bool Yuv444pToBgrAutoTest();
    bool Yuv420pToBgrAutoTest();

    bool Yuv444pToBgraAutoTest();
    bool Yuv420pToBgraAutoTest();

    bool MedianFilterRhomb3x3AutoTest();
    bool MedianFilterRhomb5x5AutoTest();
    bool MedianFilterSquare3x3AutoTest();
    bool MedianFilterSquare5x5AutoTest();
    bool GaussianBlur3x3AutoTest();
    bool AbsGradientSaturatedSumAutoTest();
    bool LbpEstimateAutoTest();

    bool SquareDifferenceSumAutoTest();
    bool SquareDifferenceSumMaskedAutoTest();
    bool AbsDifferenceSumAutoTest();
    bool AbsDifferenceSumMaskedAutoTest();
    bool AbsDifferenceSums3x3AutoTest();
    bool AbsDifferenceSums3x3MaskedAutoTest();

    bool ResizeBilinearAutoTest();

    bool DeinterleaveUvAutoTest();

    bool OperationBinary8uAutoTest();
    bool OperationBinary16iAutoTest();
    bool VectorProductAutoTest();

    bool AbsSecondDerivativeHistogramAutoTest();

    bool BinarizationAutoTest();
    bool AveragingBinarizationAutoTest();

    bool ShiftBilinearAutoTest();

    bool GetStatisticAutoTest();
    bool GetMomentsAutoTest();
    bool GetRowSumsAutoTest();
    bool GetColSumsAutoTest();
    bool GetAbsDyRowSumsAutoTest();
    bool GetAbsDxColSumsAutoTest();
    bool ValueSumAutoTest();
    bool SquareSumAutoTest();

    bool StretchGray2x2AutoTest();

    bool BackgroundGrowRangeSlowAutoTest();
    bool BackgroundGrowRangeFastAutoTest();
    bool BackgroundIncrementCountAutoTest();
    bool BackgroundAdjustRangeAutoTest();
    bool BackgroundAdjustRangeMaskedAutoTest();
    bool BackgroundShiftRangeAutoTest();
    bool BackgroundShiftRangeMaskedAutoTest();
    bool BackgroundInitMaskAutoTest();

    bool EdgeBackgroundGrowRangeSlowAutoTest();
    bool EdgeBackgroundGrowRangeFastAutoTest();
    bool EdgeBackgroundIncrementCountAutoTest();
    bool EdgeBackgroundAdjustRangeAutoTest();
    bool EdgeBackgroundAdjustRangeMaskedAutoTest();
    bool EdgeBackgroundShiftRangeAutoTest();
    bool EdgeBackgroundShiftRangeMaskedAutoTest();

    bool AddFeatureDifferenceAutoTest();

    bool TextureBoostedSaturatedGradientAutoTest();
    bool TextureBoostedUvAutoTest();
    bool TextureGetDifferenceSumAutoTest();
    bool TexturePerformCompensationAutoTest();

    bool FillBgraAutoTest();
    bool FillBgrAutoTest();

    bool GrayToBgrAutoTest();

    bool GrayToBgraAutoTest();

    bool AlphaBlendingAutoTest();

    bool ConditionalCountAutoTest();
    bool ConditionalSumAutoTest();
    bool ConditionalSquareSumAutoTest();
    bool ConditionalSquareGradientSumAutoTest();

    bool SobelDxAutoTest();
    bool SobelDxAbsAutoTest();
    bool SobelDyAutoTest();
    bool SobelDyAbsAutoTest();
    bool ContourMetricsAutoTest();
    bool ContourMetricsMaskedAutoTest();
    bool ContourAnchorsAutoTest();

    bool SegmentationShrinkRegionAutoTest();
    bool SegmentationFillSingleHolesAutoTest();

    /***************************** Data Tests: *******************************/

    bool AddFeatureDifferenceDataTest(bool create);

    bool BackgroundGrowRangeSlowDataTest(bool create);
    bool BackgroundGrowRangeFastDataTest(bool create);
    bool BackgroundIncrementCountDataTest(bool create);
    bool BackgroundAdjustRangeDataTest(bool create);

    bool BgraToGrayDataTest(bool create);

    bool BgrToGrayDataTest(bool create);

    bool ConditionalCountDataTest(bool create);

    bool DeinterleaveUvDataTest(bool create);

    bool GrayToBgraDataTest(bool create);

    bool AbsSecondDerivativeHistogramDataTest(bool create);

    bool AbsGradientSaturatedSumDataTest(bool create);

    bool ReduceGray2x2DataTest(bool create);
    bool ReduceGray4x4DataTest(bool create);

    bool OperationBinary8uDataTest(bool create);

    bool ShiftBilinearDataTest(bool create);

    bool GetAbsDyRowSumsDataTest(bool create);
    bool GetAbsDxColSumsDataTest(bool create);

    bool TextureBoostedSaturatedGradientDataTest(bool create);
}
#endif//__Test_h__

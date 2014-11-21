/*
* Simd Library Tests.
*
* Copyright (c) 2011-2014 Yermalayeu Ihar,
*               2014-2014 Antonenka Mikhail.
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

    bool AbsDifferenceSumAutoTest();
    bool AbsDifferenceSumMaskedAutoTest();
    bool AbsDifferenceSums3x3AutoTest();
    bool AbsDifferenceSums3x3MaskedAutoTest();
    bool SquaredDifferenceSumAutoTest();
    bool SquaredDifferenceSumMaskedAutoTest();
    bool SquaredDifferenceSum32fAutoTest();

    bool AddFeatureDifferenceAutoTest();

    bool BackgroundGrowRangeSlowAutoTest();
    bool BackgroundGrowRangeFastAutoTest();
    bool BackgroundIncrementCountAutoTest();
    bool BackgroundAdjustRangeAutoTest();
    bool BackgroundAdjustRangeMaskedAutoTest();
    bool BackgroundShiftRangeAutoTest();
    bool BackgroundShiftRangeMaskedAutoTest();
    bool BackgroundInitMaskAutoTest();

    bool BayerToBgrAutoTest();

    bool BayerToBgraAutoTest();

    bool BgraToBayerAutoTest();

    bool BgraToBgrAutoTest();

    bool BgraToGrayAutoTest();

    bool BgrToBayerAutoTest();

    bool BgrToBgraAutoTest();
    bool Bgr48pToBgra32AutoTest();

    bool BgrToGrayAutoTest();

    bool BgraToYuv420pAutoTest();
    bool BgraToYuv444pAutoTest();
    bool BgrToYuv420pAutoTest();
    bool BgrToYuv444pAutoTest();

    bool BinarizationAutoTest();
    bool AveragingBinarizationAutoTest();

    bool ConditionalCount8uAutoTest();
    bool ConditionalCount16iAutoTest();
    bool ConditionalSumAutoTest();
    bool ConditionalSquareSumAutoTest();
    bool ConditionalSquareGradientSumAutoTest();

    bool CopyAutoTest();
    bool CopyFrameAutoTest();

    bool Crc32cAutoTest();

    bool DeinterleaveUvAutoTest();

    bool AlphaBlendingAutoTest();

    bool EdgeBackgroundGrowRangeSlowAutoTest();
    bool EdgeBackgroundGrowRangeFastAutoTest();
    bool EdgeBackgroundIncrementCountAutoTest();
    bool EdgeBackgroundAdjustRangeAutoTest();
    bool EdgeBackgroundAdjustRangeMaskedAutoTest();
    bool EdgeBackgroundShiftRangeAutoTest();
    bool EdgeBackgroundShiftRangeMaskedAutoTest();

    bool FillAutoTest();
    bool FillFrameAutoTest();
    bool FillBgraAutoTest();
    bool FillBgrAutoTest();

    bool GrayToBgrAutoTest();

    bool GrayToBgraAutoTest();

    bool HistogramAutoTest();
    bool HistogramMaskedAutoTest();
    bool AbsSecondDerivativeHistogramAutoTest();

    bool IntegralAutoTest();

    bool InterferenceIncrementAutoTest();
    bool InterferenceIncrementMaskedAutoTest();
    bool InterferenceDecrementAutoTest();
    bool InterferenceDecrementMaskedAutoTest();

    bool MedianFilterRhomb3x3AutoTest();
    bool MedianFilterRhomb5x5AutoTest();
    bool MedianFilterSquare3x3AutoTest();
    bool MedianFilterSquare5x5AutoTest();
    bool GaussianBlur3x3AutoTest();
    bool AbsGradientSaturatedSumAutoTest();
    bool LbpEstimateAutoTest();

    bool OperationBinary8uAutoTest();
    bool OperationBinary16iAutoTest();
    bool VectorProductAutoTest();

    bool ReduceGray2x2AutoTest();
    bool ReduceGray3x3AutoTest();
    bool ReduceGray4x4AutoTest();
    bool ReduceGray5x5AutoTest();

    bool Reorder16bitAutoTest();
    bool Reorder32bitAutoTest();

    bool ResizeBilinearAutoTest();

    bool SegmentationShrinkRegionAutoTest();
    bool SegmentationFillSingleHolesAutoTest();
    bool SegmentationChangeIndexAutoTest();
    bool SegmentationPropagate2x2AutoTest();

    bool ShiftBilinearAutoTest();

    bool SobelDxAutoTest();
    bool SobelDxAbsAutoTest();
    bool SobelDyAutoTest();
    bool SobelDyAbsAutoTest();
    bool ContourMetricsAutoTest();
    bool ContourMetricsMaskedAutoTest();
    bool ContourAnchorsAutoTest();

    bool GetStatisticAutoTest();
    bool GetMomentsAutoTest();
    bool GetRowSumsAutoTest();
    bool GetColSumsAutoTest();
    bool GetAbsDyRowSumsAutoTest();
    bool GetAbsDxColSumsAutoTest();
    bool ValueSumAutoTest();
    bool SquareSumAutoTest();

    bool StretchGray2x2AutoTest();

    bool TextureBoostedSaturatedGradientAutoTest();
    bool TextureBoostedUvAutoTest();
    bool TextureGetDifferenceSumAutoTest();
    bool TexturePerformCompensationAutoTest();

    bool Yuv444pToHueAutoTest();
    bool Yuv420pToHueAutoTest();

    bool Yuv444pToBgrAutoTest();
    bool Yuv420pToBgrAutoTest();

    bool Yuv444pToBgraAutoTest();
    bool Yuv420pToBgraAutoTest();

    /***************************** Data Tests: *******************************/

    bool AbsDifferenceSumDataTest(bool create);
    bool AbsDifferenceSumMaskedDataTest(bool create);
    bool AbsDifferenceSums3x3DataTest(bool create);
    bool AbsDifferenceSums3x3MaskedDataTest(bool create);
    bool SquaredDifferenceSumDataTest(bool create);
    bool SquaredDifferenceSumMaskedDataTest(bool create);
    bool SquaredDifferenceSum32fDataTest(bool create);

    bool AddFeatureDifferenceDataTest(bool create);

    bool BackgroundGrowRangeSlowDataTest(bool create);
    bool BackgroundGrowRangeFastDataTest(bool create);
    bool BackgroundIncrementCountDataTest(bool create);
    bool BackgroundAdjustRangeDataTest(bool create);
    bool BackgroundAdjustRangeMaskedDataTest(bool create);
    bool BackgroundShiftRangeDataTest(bool create);
    bool BackgroundShiftRangeMaskedDataTest(bool create);
    bool BackgroundInitMaskDataTest(bool create);

    bool BayerToBgrDataTest(bool create);

    bool BayerToBgraDataTest(bool create);

    bool BgraToBayerDataTest(bool create);

    bool BgraToBgrDataTest(bool create);

    bool BgraToGrayDataTest(bool create);

    bool BgrToBayerDataTest(bool create);

    bool BgrToBgraDataTest(bool create);
    bool Bgr48pToBgra32DataTest(bool create);

    bool BgrToGrayDataTest(bool create);

    bool BgraToYuv420pDataTest(bool create);
    bool BgraToYuv444pDataTest(bool create);
    bool BgrToYuv420pDataTest(bool create);
    bool BgrToYuv444pDataTest(bool create);

    bool BinarizationDataTest(bool create);
    bool AveragingBinarizationDataTest(bool create);

    bool ConditionalCount8uDataTest(bool create);
    bool ConditionalCount16iDataTest(bool create);
    bool ConditionalSumDataTest(bool create);
    bool ConditionalSquareSumDataTest(bool create);
    bool ConditionalSquareGradientSumDataTest(bool create);

    bool CopyDataTest(bool create);
    bool CopyFrameDataTest(bool create);

    bool Crc32cDataTest(bool create);

    bool DeinterleaveUvDataTest(bool create);

    bool AlphaBlendingDataTest(bool create);

    bool EdgeBackgroundGrowRangeSlowDataTest(bool create);
    bool EdgeBackgroundGrowRangeFastDataTest(bool create);
    bool EdgeBackgroundIncrementCountDataTest(bool create);
    bool EdgeBackgroundAdjustRangeDataTest(bool create);
    bool EdgeBackgroundAdjustRangeMaskedDataTest(bool create);
    bool EdgeBackgroundShiftRangeDataTest(bool create);
    bool EdgeBackgroundShiftRangeMaskedDataTest(bool create);

    bool FillDataTest(bool create);
    bool FillFrameDataTest(bool create);
    bool FillBgrDataTest(bool create);
    bool FillBgraDataTest(bool create);

    bool GrayToBgraDataTest(bool create);

    bool HistogramDataTest(bool create);
    bool HistogramMaskedDataTest(bool create);
    bool AbsSecondDerivativeHistogramDataTest(bool create);

    bool IntegralDataTest(bool create);

    bool InterferenceIncrementDataTest(bool create);
    bool InterferenceIncrementMaskedDataTest(bool create);
    bool InterferenceDecrementDataTest(bool create);
    bool InterferenceDecrementMaskedDataTest(bool create);

    bool AbsGradientSaturatedSumDataTest(bool create);
    bool GaussianBlur3x3DataTest(bool create);
    bool LbpEstimateDataTest(bool create);
    bool MedianFilterRhomb3x3DataTest(bool create);
    bool MedianFilterRhomb5x5DataTest(bool create);
    bool MedianFilterSquare3x3DataTest(bool create);
    bool MedianFilterSquare5x5DataTest(bool create);

    bool OperationBinary8uDataTest(bool create);
    bool OperationBinary16iDataTest(bool create);

    bool ReduceGray2x2DataTest(bool create);
    bool ReduceGray3x3DataTest(bool create);
    bool ReduceGray4x4DataTest(bool create);
    bool ReduceGray5x5DataTest(bool create);

    bool Reorder16bitDataTest(bool create);
    bool Reorder32bitDataTest(bool create);

    bool ResizeBilinearDataTest(bool create);

    bool SegmentationShrinkRegionDataTest(bool create);
    bool SegmentationFillSingleHolesDataTest(bool create);
    bool SegmentationChangeIndexDataTest(bool create);
    bool SegmentationPropagate2x2DataTest(bool create);

    bool ShiftBilinearDataTest(bool create);

    bool SobelDxDataTest(bool create);
    bool SobelDxAbsDataTest(bool create);
    bool SobelDyDataTest(bool create);
    bool SobelDyAbsDataTest(bool create);
    bool ContourMetricsDataTest(bool create);
    bool ContourMetricsMaskedDataTest(bool create);

    bool GetStatisticDataTest(bool create);
    bool GetMomentsDataTest(bool create);
    bool GetRowSumsDataTest(bool create);
    bool GetColSumsDataTest(bool create);
    bool GetAbsDyRowSumsDataTest(bool create);
    bool GetAbsDxColSumsDataTest(bool create);
    bool ValueSumDataTest(bool create);
    bool SquareSumDataTest(bool create);

    bool StretchGray2x2DataTest(bool create);

    bool TextureBoostedSaturatedGradientDataTest(bool create);
    bool TextureGetDifferenceSumDataTest(bool create);
    bool TexturePerformCompensationDataTest(bool create);
    bool TextureBoostedUvDataTest(bool create);

    bool Yuv420pToBgrDataTest(bool create);
    bool Yuv444pToBgrDataTest(bool create);

    bool Yuv420pToBgraDataTest(bool create);
    bool Yuv444pToBgraDataTest(bool create);

    bool Yuv420pToHueDataTest(bool create);
    bool Yuv444pToHueDataTest(bool create);
}
#endif//__Test_h__

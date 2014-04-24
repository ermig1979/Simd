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
#include "Test/TestPerformance.h"

#include "Test/Test.h"

#define EXECUTE_TEST(test)\
if(argc < 2 || std::string(#test).find(argv[1]) != std::string::npos) \
{\
	std::cout << #test << " is started :" << std::endl; \
	bool result = test(); \
	std::cout << #test << " is finished "  << (result ? "successfully." : "with errors!") << std::endl << std::endl; \
	if(!result) \
	{ \
		std::cout << "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl << std::endl; \
		return 1; \
	} \
}

int main(int argc, char* argv[])
{
    using namespace Test;

    EXECUTE_TEST(ReduceGray2x2Test);
    EXECUTE_TEST(ReduceGray3x3Test);
    EXECUTE_TEST(ReduceGray4x4Test);
    EXECUTE_TEST(ReduceGray5x5Test);

    EXECUTE_TEST(Crc32cTest);

    EXECUTE_TEST(BgraToGrayTest);

    EXECUTE_TEST(BgrToBgraTest);
    EXECUTE_TEST(Bgr48pToBgra32Test);

    EXECUTE_TEST(BgrToGrayTest);

    EXECUTE_TEST(Yuv444pToHueTest);
    EXECUTE_TEST(Yuv420pToHueTest);

    EXECUTE_TEST(Yuv444pToBgrTest);
    EXECUTE_TEST(Yuv420pToBgrTest);

    EXECUTE_TEST(Yuv444pToBgraTest);
    EXECUTE_TEST(Yuv420pToBgraTest);

    EXECUTE_TEST(MedianFilterRhomb3x3Test);
    EXECUTE_TEST(MedianFilterRhomb5x5Test);
    EXECUTE_TEST(MedianFilterSquare3x3Test);
    EXECUTE_TEST(MedianFilterSquare5x5Test);
    EXECUTE_TEST(GaussianBlur3x3Test);
    EXECUTE_TEST(AbsGradientSaturatedSumTest);
    EXECUTE_TEST(LbpEstimateTest);

    EXECUTE_TEST(SquareDifferenceSumTest);
    EXECUTE_TEST(SquareDifferenceSumMaskedTest);
    EXECUTE_TEST(AbsDifferenceSumTest);
    EXECUTE_TEST(AbsDifferenceSumMaskedTest);
    EXECUTE_TEST(AbsDifferenceSums3x3Test);
    EXECUTE_TEST(AbsDifferenceSums3x3MaskedTest);

    EXECUTE_TEST(ResizeBilinearTest);

    EXECUTE_TEST(DeinterleaveUvTest);

    EXECUTE_TEST(OperationBinary8uTest);
    EXECUTE_TEST(OperationBinary16iTest);
    EXECUTE_TEST(VectorProductTest);

    EXECUTE_TEST(AbsSecondDerivativeHistogramTest);

    EXECUTE_TEST(BinarizationTest);
    EXECUTE_TEST(AveragingBinarizationTest);

    EXECUTE_TEST(ShiftBilinearTest);

    EXECUTE_TEST(GetStatisticTest);
    //EXECUTE_TEST(GetMomentsTest);
    EXECUTE_TEST(GetRowSumsTest);
    EXECUTE_TEST(GetColSumsTest);
    EXECUTE_TEST(GetAbsDyRowSumsTest);
    EXECUTE_TEST(GetAbsDxColSumsTest);
    EXECUTE_TEST(ValueSumTest);
    EXECUTE_TEST(SquareSumTest);

    EXECUTE_TEST(StretchGray2x2Test);

    EXECUTE_TEST(BackgroundGrowRangeSlowTest);
    EXECUTE_TEST(BackgroundGrowRangeFastTest);
    EXECUTE_TEST(BackgroundIncrementCountTest);
    EXECUTE_TEST(BackgroundAdjustRangeTest);
    EXECUTE_TEST(BackgroundAdjustRangeMaskedTest);
    EXECUTE_TEST(BackgroundShiftRangeTest);
    EXECUTE_TEST(BackgroundShiftRangeMaskedTest);
    EXECUTE_TEST(BackgroundInitMaskTest);

    EXECUTE_TEST(EdgeBackgroundGrowRangeSlowTest);
    EXECUTE_TEST(EdgeBackgroundGrowRangeFastTest);
    EXECUTE_TEST(EdgeBackgroundIncrementCountTest);
    EXECUTE_TEST(EdgeBackgroundAdjustRangeTest);
    EXECUTE_TEST(EdgeBackgroundAdjustRangeMaskedTest);
    EXECUTE_TEST(EdgeBackgroundShiftRangeTest);
    EXECUTE_TEST(EdgeBackgroundShiftRangeMaskedTest);

    EXECUTE_TEST(AddFeatureDifferenceTest);

    EXECUTE_TEST(TextureBoostedSaturatedGradientTest);
    EXECUTE_TEST(TextureBoostedUvTest);
    EXECUTE_TEST(TextureGetDifferenceSumTest);
    EXECUTE_TEST(TexturePerformCompensationTest);

    EXECUTE_TEST(FillBgraTest);
    EXECUTE_TEST(FillBgrTest);

    EXECUTE_TEST(GrayToBgrTest);

    EXECUTE_TEST(GrayToBgraTest);

    EXECUTE_TEST(AlphaBlendingTest);

    EXECUTE_TEST(ConditionalCountTest);
    EXECUTE_TEST(ConditionalSumTest);
    EXECUTE_TEST(ConditionalSquareSumTest);
    EXECUTE_TEST(ConditionalSquareGradientSumTest);

    EXECUTE_TEST(SobelDxTest);
    EXECUTE_TEST(SobelDxAbsTest);
    EXECUTE_TEST(SobelDyTest);
    EXECUTE_TEST(SobelDyAbsTest);
    EXECUTE_TEST(ContourMetricsTest);
    EXECUTE_TEST(ContourMetricsMaskedTest);
    EXECUTE_TEST(ContourAnchorsTest);

    EXECUTE_TEST(BgraToBgrTest);

    EXECUTE_TEST(BgraToBayerTest);

    EXECUTE_TEST(BayerToBgrTest);

    EXECUTE_TEST(BayerToBgraTest);

    EXECUTE_TEST(BgrToBayerTest);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
    std::cout << "Performance report:" << std::endl;
    std::cout << Test::PerformanceMeasurerStorage::s_storage.Report(false, true, false) << std::endl;
#endif//TEST_PERFORMANCE_TEST_ENABLE

	return 0;
}

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
#include "Test/TestPerformance.h"

#include "Test/Test.h"

#define EXECUTE_TEST(test)\
if(argc < 2 || std::string(argv[1]) == std::string(#test)) \
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

    EXECUTE_TEST(Crc32Test);

    EXECUTE_TEST(BgraToGrayTest);

    EXECUTE_TEST(BgrToGrayTest);

    EXECUTE_TEST(Yuv444ToHueTest);
    EXECUTE_TEST(Yuv420ToHueTest);

    EXECUTE_TEST(Yuv444ToBgraTest);
    EXECUTE_TEST(Yuv420ToBgraTest);

    EXECUTE_TEST(MedianFilterSquare3x3Test);
    EXECUTE_TEST(MedianFilterSquare5x5Test);
    EXECUTE_TEST(GaussianBlur3x3Test);
    EXECUTE_TEST(AbsGradientSaturatedSumTest);

    EXECUTE_TEST(SquareDifferenceSumTest);
    EXECUTE_TEST(MaskedSquareDifferenceSumTest);
    EXECUTE_TEST(AbsDifferenceSumTest);
    EXECUTE_TEST(MaskedAbsDifferenceSumTest);

    EXECUTE_TEST(ResizeBilinearTest);

    EXECUTE_TEST(DeinterleaveUvTest);

    EXECUTE_TEST(OperationTest);

    EXECUTE_TEST(AbsSecondDerivativeHistogramTest);

    EXECUTE_TEST(BinarizationTest);
    EXECUTE_TEST(AveragingBinarizationTest);

    EXECUTE_TEST(ShiftBilinearTest);

    EXECUTE_TEST(GetStatisticTest);
    EXECUTE_TEST(GetMomentsTest);
    EXECUTE_TEST(GetRowSumsTest);
    EXECUTE_TEST(GetColSumsTest);

    EXECUTE_TEST(StretchGray2x2Test);

    EXECUTE_TEST(BackgroundGrowRangeSlowTest);
    EXECUTE_TEST(BackgroundGrowRangeFastTest);
    EXECUTE_TEST(BackgroundIncrementCountTest);
    EXECUTE_TEST(BackgroundAdjustRangeTest);
    EXECUTE_TEST(MaskedBackgroundAdjustRangeTest);
    EXECUTE_TEST(BackgroundShiftRangeTest);
    EXECUTE_TEST(MaskedBackgroundShiftRangeTest);
    EXECUTE_TEST(BackgroundInitMaskTest);

    EXECUTE_TEST(EdgeBackgroundGrowRangeSlowTest);
    EXECUTE_TEST(EdgeBackgroundGrowRangeFastTest);
    EXECUTE_TEST(EdgeBackgroundIncrementCountTest);
    EXECUTE_TEST(EdgeBackgroundAdjustRangeTest);
    EXECUTE_TEST(MaskedEdgeBackgroundAdjustRangeTest);
    EXECUTE_TEST(EdgeBackgroundShiftRangeTest);
    EXECUTE_TEST(MaskedEdgeBackgroundShiftRangeTest);

    EXECUTE_TEST(AddFeatureDifferenceTest);

    EXECUTE_TEST(TextureBoostedSaturatedGradientTest);
    EXECUTE_TEST(TextureBoostedUvTest);
    EXECUTE_TEST(TextureGetDifferenceSumTest);
    EXECUTE_TEST(TexturePerformCompensationTest);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
	std::cout << "Function execution times:" << std::endl;
	std::cout << Test::PerformanceMeasurerStorage::s_storage.Statistic() << std::endl;
#endif//TEST_PERFORMANCE_TEST_ENABLE

	return 0;
}

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

#include "Test/TestAverage.h"
#include "Test/TestBackground.h"
#include "Test/TestBgraToGray.h"
#include "Test/TestBgrToGray.h"
#include "Test/TestBinarization.h"
#include "Test/TestCrc32.h"
#include "Test/TestDeinterleave.h"
#include "Test/TestDifferenceSum.h"
#include "Test/TestFilter.h"
#include "Test/TestHistogram.h"
#include "Test/TestReduceGray.h"
#include "Test/TestResize.h"
#include "Test/TestShift.h"
#include "Test/TestStatistic.h"
#include "Test/TestStretchGray.h"
#include "Test/TestYuvToBgra.h"
#include "Test/TestYuvToHue.h"

#define EXECUTE_TEST(test)\
{\
	std::cout << #test << " is started :" << std::endl; \
	bool result = test(); \
	std::cout << #test << " is finished "  << (result ? " successfully." : " with errors!") << std::endl << std::endl; \
	if(!result) \
	{ \
		std::cout << "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl << std::endl; \
		return 1; \
	} \
}

int main(int argc, char* argv[])
{
    EXECUTE_TEST(Test::ReduceGray2x2Test);
    EXECUTE_TEST(Test::ReduceGray3x3Test);
    EXECUTE_TEST(Test::ReduceGray4x4Test);
    EXECUTE_TEST(Test::ReduceGray5x5Test);

    EXECUTE_TEST(Test::Crc32Test);

    EXECUTE_TEST(Test::BgraToGrayTest);

    EXECUTE_TEST(Test::BgrToGrayTest);

    EXECUTE_TEST(Test::Yuv444ToHueTest);
    EXECUTE_TEST(Test::Yuv420ToHueTest);

    EXECUTE_TEST(Test::Yuv444ToBgraTest);
    EXECUTE_TEST(Test::Yuv420ToBgraTest);

    EXECUTE_TEST(Test::MedianFilterSquare3x3Test);
    EXECUTE_TEST(Test::MedianFilterSquare5x5Test);
    EXECUTE_TEST(Test::GaussianBlur3x3Test);
    EXECUTE_TEST(Test::AbsGradientSaturatedSumTest);

    EXECUTE_TEST(Test::SquareDifferenceSumTest);
    EXECUTE_TEST(Test::MaskedSquareDifferenceSumTest);
    EXECUTE_TEST(Test::AbsDifferenceSumTest);
    EXECUTE_TEST(Test::MaskedAbsDifferenceSumTest);

    EXECUTE_TEST(Test::ResizeBilinearTest);

    EXECUTE_TEST(Test::DeinterleaveUvTest);

    EXECUTE_TEST(Test::AverageTest);

    EXECUTE_TEST(Test::AbsSecondDerivativeHistogramTest);

    EXECUTE_TEST(Test::GreaterThenBinarizationTest);
    EXECUTE_TEST(Test::LesserThenBinarizationTest);
    EXECUTE_TEST(Test::EqualToBinarizationTest);

    EXECUTE_TEST(Test::ShiftBilinearTest);

    EXECUTE_TEST(Test::GetStatisticTest);

	EXECUTE_TEST(Test::StretchGray2x2Test);

	EXECUTE_TEST(Test::BackgroundGrowRangeSlowTest);
	EXECUTE_TEST(Test::BackgroundGrowRangeFastTest);
	EXECUTE_TEST(Test::BackgroundIncrementCountTest);
	EXECUTE_TEST(Test::BackgroundAdjustRangeTest);
	EXECUTE_TEST(Test::MaskedBackgroundAdjustRangeTest);
	EXECUTE_TEST(Test::BackgroundShiftRangeTest);
	EXECUTE_TEST(Test::MaskedBackgroundShiftRangeTest);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
	std::cout << "Function execution times:" << std::endl;
	std::cout << Test::PerformanceMeasurerStorage::s_storage.Statistic() << std::endl;
#endif//TEST_PERFORMANCE_TEST_ENABLE

	return 0;
}

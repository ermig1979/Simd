/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar,
*               2014-2015 Antonenka Mikhail.
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

namespace Test
{
    struct Options
    {
        enum Type
        {
            Auto,
            Create, 
            Verify,
        } type;
        std::string filter;

        Options(int argc, char* argv[])
            : type(Auto)
        {
            if(argc > 1)
            {
                std::string first(argv[1]); 
                if(first == "a" || first == "c" || first == "v")
                {
                    if(first == "c")
                        type = Create;
                    if(first == "v")
                        type = Verify;
                    if(argc > 2)
                        filter = argv[2];
                }
                else
                    filter = first;
            }
        }

        bool NeedToPerform(const std::string & name) const
        {
            return filter.empty() || name.find(filter) != std::string::npos;
        }
    };

    typedef bool (*AutoTestPtr)(); 
    typedef bool (*DataTestPtr)(bool create); 

    struct Test
    {
        std::string name;
        AutoTestPtr autoTest;
        DataTestPtr dataTest;
        Test(const std::string & n, const AutoTestPtr & a, const DataTestPtr & d)
            : name(n)
            , autoTest(a)
            , dataTest(d)
        {
        }
    };
    std::list<Test> g_tests;

#define ADD_TEST(name) \
    bool name##AutoTest(); \
    bool name##DataTest(bool create); \
    bool name##AddToList(){ g_tests.push_back(Test(#name, name##AutoTest, name##DataTest)); return true; } \
    bool name##AtList = name##AddToList();

    ADD_TEST(AbsDifferenceSum);
    ADD_TEST(AbsDifferenceSumMasked);
    ADD_TEST(AbsDifferenceSums3x3);
    ADD_TEST(AbsDifferenceSums3x3Masked);
    ADD_TEST(SquaredDifferenceSum);
    ADD_TEST(SquaredDifferenceSumMasked);
    ADD_TEST(SquaredDifferenceSum32f);

    ADD_TEST(AddFeatureDifference);

    ADD_TEST(BgraToBgr);
    ADD_TEST(BgraToGray);
    ADD_TEST(BgrToGray);
    ADD_TEST(BgrToHsl);
    ADD_TEST(BgrToHsv);
    ADD_TEST(GrayToBgr);

    ADD_TEST(BgraToBayer);
    ADD_TEST(BgrToBayer);

    ADD_TEST(BgrToBgra);
    ADD_TEST(GrayToBgra);

    ADD_TEST(BgraToYuv420p);
    ADD_TEST(BgraToYuv422p);
    ADD_TEST(BgraToYuv444p);
    ADD_TEST(BgrToYuv420p);
    ADD_TEST(BgrToYuv422p);
    ADD_TEST(BgrToYuv444p);

    ADD_TEST(BackgroundGrowRangeSlow);
    ADD_TEST(BackgroundGrowRangeFast);
    ADD_TEST(BackgroundIncrementCount);
    ADD_TEST(BackgroundAdjustRange);
    ADD_TEST(BackgroundAdjustRangeMasked);
    ADD_TEST(BackgroundShiftRange);
    ADD_TEST(BackgroundShiftRangeMasked);
    ADD_TEST(BackgroundInitMask);

    ADD_TEST(BayerToBgr);

    ADD_TEST(BayerToBgra);

    ADD_TEST(Bgr48pToBgra32);

    ADD_TEST(Binarization);
    ADD_TEST(AveragingBinarization);

    ADD_TEST(ConditionalCount8u);
    ADD_TEST(ConditionalCount16i);
    ADD_TEST(ConditionalSum);
    ADD_TEST(ConditionalSquareSum);
    ADD_TEST(ConditionalSquareGradientSum);

    ADD_TEST(Copy);
    ADD_TEST(CopyFrame);

    ADD_TEST(Crc32c);

    ADD_TEST(DeinterleaveUv);

    ADD_TEST(AlphaBlending);

    ADD_TEST(EdgeBackgroundGrowRangeSlow);
    ADD_TEST(EdgeBackgroundGrowRangeFast);
    ADD_TEST(EdgeBackgroundIncrementCount);
    ADD_TEST(EdgeBackgroundAdjustRange);
    ADD_TEST(EdgeBackgroundAdjustRangeMasked);
    ADD_TEST(EdgeBackgroundShiftRange);
    ADD_TEST(EdgeBackgroundShiftRangeMasked);

    ADD_TEST(Fill);
    ADD_TEST(FillFrame);
    ADD_TEST(FillBgra);
    ADD_TEST(FillBgr);

    ADD_TEST(Histogram);
    ADD_TEST(HistogramMasked);
    ADD_TEST(AbsSecondDerivativeHistogram);

    ADD_TEST(HogDirectionHistograms);

    ADD_TEST(Integral);

    ADD_TEST(InterferenceIncrement);
    ADD_TEST(InterferenceIncrementMasked);
    ADD_TEST(InterferenceDecrement);
    ADD_TEST(InterferenceDecrementMasked);

    ADD_TEST(MedianFilterRhomb3x3);
    ADD_TEST(MedianFilterRhomb5x5);
    ADD_TEST(MedianFilterSquare3x3);
    ADD_TEST(MedianFilterSquare5x5);
    ADD_TEST(GaussianBlur3x3);
    ADD_TEST(AbsGradientSaturatedSum);
    ADD_TEST(LbpEstimate);

    ADD_TEST(OperationBinary8u);
    ADD_TEST(OperationBinary16i);
    ADD_TEST(VectorProduct);

    ADD_TEST(ReduceGray2x2);
    ADD_TEST(ReduceGray3x3);
    ADD_TEST(ReduceGray4x4);
    ADD_TEST(ReduceGray5x5);

    ADD_TEST(Reorder16bit);
    ADD_TEST(Reorder32bit);
    ADD_TEST(Reorder64bit);

    ADD_TEST(ResizeBilinear);

    ADD_TEST(SegmentationShrinkRegion);
    ADD_TEST(SegmentationFillSingleHoles);
    ADD_TEST(SegmentationChangeIndex);
    ADD_TEST(SegmentationPropagate2x2);

    ADD_TEST(ShiftBilinear);

    ADD_TEST(SobelDx);
    ADD_TEST(SobelDxAbs);
    ADD_TEST(SobelDy);
    ADD_TEST(SobelDyAbs);
    ADD_TEST(ContourMetrics);
    ADD_TEST(ContourMetricsMasked);
    ADD_TEST(ContourAnchors);

    ADD_TEST(GetStatistic);
    ADD_TEST(GetMoments);
    ADD_TEST(GetRowSums);
    ADD_TEST(GetColSums);
    ADD_TEST(GetAbsDyRowSums);
    ADD_TEST(GetAbsDxColSums);
    ADD_TEST(ValueSum);
    ADD_TEST(SquareSum);
    ADD_TEST(SobelDxAbsSum);
    ADD_TEST(SobelDyAbsSum);
    ADD_TEST(CorrelationSum);

    ADD_TEST(StretchGray2x2);

    ADD_TEST(SvmSumLinear);

    ADD_TEST(TextureBoostedSaturatedGradient);
    ADD_TEST(TextureBoostedUv);
    ADD_TEST(TextureGetDifferenceSum);
    ADD_TEST(TexturePerformCompensation);

    ADD_TEST(Yuv444pToBgr);
    ADD_TEST(Yuv422pToBgr);
    ADD_TEST(Yuv420pToBgr);
    ADD_TEST(Yuv444pToHsl);
    ADD_TEST(Yuv444pToHsv);
    ADD_TEST(Yuv444pToHue);
    ADD_TEST(Yuv420pToHue);

    ADD_TEST(Yuv444pToBgra);
    ADD_TEST(Yuv422pToBgra);
    ADD_TEST(Yuv420pToBgra);
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);
    if(options.type == Test::Options::Auto)
    {
        for(const Test::Test & test : Test::g_tests)
        {
            if(options.NeedToPerform(test.name)) 
            {
                std::cout << test.name << "AutoTest is started :" << std::endl; 
                bool result = test.autoTest(); 
                std::cout << test.name << "AutoTest  is finished "  << (result ? "successfully." : "with errors!") << std::endl << std::endl; 
                if(!result) 
                { 
                    std::cout << "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl << std::endl; 
                    return 1;
                } 
            }
        }
#ifdef TEST_PERFORMANCE_TEST_ENABLE
        std::cout << Test::PerformanceMeasurerStorage::s_storage.Report(true, true, false) << std::endl;
#endif
    }
    else
    {
        for(const Test::Test & test : Test::g_tests)
        {
            if(options.NeedToPerform(test.name)) 
            {
                bool create = options.type == Test::Options::Create;
                std::cout << test.name << "DataTest - data " << (create ? "creation" : "verification") << " is started :" << std::endl; 
                bool result = test.dataTest(create); 
                std::cout << test.name << "DataTest - data " << (create ? "creation" : "verification") << " is finished " << (result ? "successfully." : "with errors!") << std::endl << std::endl;
                if(!result) 
                { 
                    std::cout << "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl << std::endl; 
                    return 1;
                } 
            }
        }
    }
}

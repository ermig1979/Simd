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

#define EXECUTE_AUTO_TEST(test)\
    if(options.NeedToPerform(#test)) \
{\
    std::cout << #test << " is started :" << std::endl; \
    result = test(); \
    std::cout << #test << " is finished "  << (result ? "successfully." : "with errors!") << std::endl << std::endl; \
    if(!result) \
    { \
        std::cout << "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl << std::endl; \
        goto end; \
    } \
}

int ExecuteAutoTest(const Options & options)
{
    using namespace Test;

    bool result = true;

#ifdef CUDA_ENABLE
    if(::cudaSetDevice(0) != ::cudaSuccess)
    {
        std::cout << "Operation ::cudaSetDevice(0) is failed!" << std::endl;
        goto end;
    }
#endif

    EXECUTE_AUTO_TEST(ReduceGray2x2AutoTest);
    EXECUTE_AUTO_TEST(ReduceGray3x3AutoTest);
    EXECUTE_AUTO_TEST(ReduceGray4x4AutoTest);
    EXECUTE_AUTO_TEST(ReduceGray5x5AutoTest);

    EXECUTE_AUTO_TEST(Crc32cAutoTest);

    EXECUTE_AUTO_TEST(BgraToGrayAutoTest);

    EXECUTE_AUTO_TEST(BgrToBgraAutoTest);
    EXECUTE_AUTO_TEST(Bgr48pToBgra32AutoTest);

    EXECUTE_AUTO_TEST(BgrToGrayAutoTest);

    EXECUTE_AUTO_TEST(Yuv444pToHueAutoTest);
    EXECUTE_AUTO_TEST(Yuv420pToHueAutoTest);

    EXECUTE_AUTO_TEST(Yuv444pToBgrAutoTest);
    EXECUTE_AUTO_TEST(Yuv420pToBgrAutoTest);

    EXECUTE_AUTO_TEST(Yuv444pToBgraAutoTest);
    EXECUTE_AUTO_TEST(Yuv420pToBgraAutoTest);

    EXECUTE_AUTO_TEST(MedianFilterRhomb3x3AutoTest);
    EXECUTE_AUTO_TEST(MedianFilterRhomb5x5AutoTest);
    EXECUTE_AUTO_TEST(MedianFilterSquare3x3AutoTest);
    EXECUTE_AUTO_TEST(MedianFilterSquare5x5AutoTest);
    EXECUTE_AUTO_TEST(GaussianBlur3x3AutoTest);
    EXECUTE_AUTO_TEST(AbsGradientSaturatedSumAutoTest);
    EXECUTE_AUTO_TEST(LbpEstimateAutoTest);

    EXECUTE_AUTO_TEST(SquaredDifferenceSumAutoTest);
    EXECUTE_AUTO_TEST(SquaredDifferenceSumMaskedAutoTest);
    EXECUTE_AUTO_TEST(AbsDifferenceSumAutoTest);
    EXECUTE_AUTO_TEST(AbsDifferenceSumMaskedAutoTest);
    EXECUTE_AUTO_TEST(AbsDifferenceSums3x3AutoTest);
    EXECUTE_AUTO_TEST(AbsDifferenceSums3x3MaskedAutoTest);

    EXECUTE_AUTO_TEST(ResizeBilinearAutoTest);

    EXECUTE_AUTO_TEST(DeinterleaveUvAutoTest);

    EXECUTE_AUTO_TEST(OperationBinary8uAutoTest);
    EXECUTE_AUTO_TEST(OperationBinary16iAutoTest);
    EXECUTE_AUTO_TEST(VectorProductAutoTest);

    EXECUTE_AUTO_TEST(AbsSecondDerivativeHistogramAutoTest);

    EXECUTE_AUTO_TEST(BinarizationAutoTest);
    EXECUTE_AUTO_TEST(AveragingBinarizationAutoTest);

    EXECUTE_AUTO_TEST(ShiftBilinearAutoTest);

    EXECUTE_AUTO_TEST(GetStatisticAutoTest);
    EXECUTE_AUTO_TEST(GetMomentsAutoTest);
    EXECUTE_AUTO_TEST(GetRowSumsAutoTest);
    EXECUTE_AUTO_TEST(GetColSumsAutoTest);
    EXECUTE_AUTO_TEST(GetAbsDyRowSumsAutoTest);
    EXECUTE_AUTO_TEST(GetAbsDxColSumsAutoTest);
    EXECUTE_AUTO_TEST(ValueSumAutoTest);
    EXECUTE_AUTO_TEST(SquareSumAutoTest);

    EXECUTE_AUTO_TEST(StretchGray2x2AutoTest);

    EXECUTE_AUTO_TEST(BackgroundGrowRangeSlowAutoTest);
    EXECUTE_AUTO_TEST(BackgroundGrowRangeFastAutoTest);
    EXECUTE_AUTO_TEST(BackgroundIncrementCountAutoTest);
    EXECUTE_AUTO_TEST(BackgroundAdjustRangeAutoTest);
    EXECUTE_AUTO_TEST(BackgroundAdjustRangeMaskedAutoTest);
    EXECUTE_AUTO_TEST(BackgroundShiftRangeAutoTest);
    EXECUTE_AUTO_TEST(BackgroundShiftRangeMaskedAutoTest);
    EXECUTE_AUTO_TEST(BackgroundInitMaskAutoTest);

    EXECUTE_AUTO_TEST(EdgeBackgroundGrowRangeSlowAutoTest);
    EXECUTE_AUTO_TEST(EdgeBackgroundGrowRangeFastAutoTest);
    EXECUTE_AUTO_TEST(EdgeBackgroundIncrementCountAutoTest);
    EXECUTE_AUTO_TEST(EdgeBackgroundAdjustRangeAutoTest);
    EXECUTE_AUTO_TEST(EdgeBackgroundAdjustRangeMaskedAutoTest);
    EXECUTE_AUTO_TEST(EdgeBackgroundShiftRangeAutoTest);
    EXECUTE_AUTO_TEST(EdgeBackgroundShiftRangeMaskedAutoTest);

    EXECUTE_AUTO_TEST(AddFeatureDifferenceAutoTest);

    EXECUTE_AUTO_TEST(TextureBoostedSaturatedGradientAutoTest);
    EXECUTE_AUTO_TEST(TextureBoostedUvAutoTest);
    EXECUTE_AUTO_TEST(TextureGetDifferenceSumAutoTest);
    EXECUTE_AUTO_TEST(TexturePerformCompensationAutoTest);

    EXECUTE_AUTO_TEST(FillBgraAutoTest);
    EXECUTE_AUTO_TEST(FillBgrAutoTest);

    EXECUTE_AUTO_TEST(GrayToBgrAutoTest);

    EXECUTE_AUTO_TEST(GrayToBgraAutoTest);

    EXECUTE_AUTO_TEST(AlphaBlendingAutoTest);

    EXECUTE_AUTO_TEST(ConditionalCountAutoTest);
    EXECUTE_AUTO_TEST(ConditionalSumAutoTest);
    EXECUTE_AUTO_TEST(ConditionalSquareSumAutoTest);
    EXECUTE_AUTO_TEST(ConditionalSquareGradientSumAutoTest);

    EXECUTE_AUTO_TEST(SobelDxAutoTest);
    EXECUTE_AUTO_TEST(SobelDxAbsAutoTest);
    EXECUTE_AUTO_TEST(SobelDyAutoTest);
    EXECUTE_AUTO_TEST(SobelDyAbsAutoTest);
    EXECUTE_AUTO_TEST(ContourMetricsAutoTest);
    EXECUTE_AUTO_TEST(ContourMetricsMaskedAutoTest);
    EXECUTE_AUTO_TEST(ContourAnchorsAutoTest);

    EXECUTE_AUTO_TEST(BgraToBgrAutoTest);

    EXECUTE_AUTO_TEST(BgraToBayerAutoTest);

    EXECUTE_AUTO_TEST(BayerToBgrAutoTest);

    EXECUTE_AUTO_TEST(BayerToBgraAutoTest);

    EXECUTE_AUTO_TEST(BgrToBayerAutoTest);

    EXECUTE_AUTO_TEST(SegmentationShrinkRegionAutoTest);
    EXECUTE_AUTO_TEST(SegmentationFillSingleHolesAutoTest);

end:

#ifdef TEST_PERFORMANCE_TEST_ENABLE
    std::cout << Test::PerformanceMeasurerStorage::s_storage.Report(false, true, true) << std::endl;
#endif//TEST_PERFORMANCE_TEST_ENABLE

#ifdef CUDA_ENABLE
    if(::cudaDeviceReset() != ::cudaSuccess)
        std::cout << "Operation ::cudaDeviceReset() is failed!" << std::endl;
#endif

    return result ? 1 : 0;
}

#define EXECUTE_DATA_TEST(test)\
if(options.NeedToPerform(#test)) \
{ \
    bool create = options.type == Options::Create; \
    std::cout << #test << " - data " << (create ? "creation" : "verification") << " is started :" << std::endl; \
    result = test(create); \
    std::cout << #test << " - data " << (create ? "creation" : "verification") << " is finished "  << (result ? "successfully." : "with errors!") << std::endl << std::endl; \
    if(!result) \
    { \
        std::cout << "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl << std::endl; \
        goto end; \
    } \
}

int ExecuteDataTest(const Options & options)
{
    using namespace Test;

    bool result = true;

    EXECUTE_DATA_TEST(AbsDifferenceSumDataTest);

    EXECUTE_DATA_TEST(AddFeatureDifferenceDataTest);

    EXECUTE_DATA_TEST(BackgroundGrowRangeSlowDataTest);
    EXECUTE_DATA_TEST(BackgroundGrowRangeFastDataTest);
    EXECUTE_DATA_TEST(BackgroundIncrementCountDataTest);
    EXECUTE_DATA_TEST(BackgroundAdjustRangeDataTest);

    EXECUTE_DATA_TEST(BgraToGrayDataTest);

    EXECUTE_DATA_TEST(BgrToGrayDataTest);

    EXECUTE_DATA_TEST(ConditionalCountDataTest);

    EXECUTE_DATA_TEST(DeinterleaveUvDataTest);

    EXECUTE_DATA_TEST(GrayToBgraDataTest);

    EXECUTE_DATA_TEST(AbsSecondDerivativeHistogramDataTest);

    EXECUTE_DATA_TEST(AbsGradientSaturatedSumDataTest);

    EXECUTE_DATA_TEST(OperationBinary8uDataTest);

    EXECUTE_DATA_TEST(ReduceGray2x2DataTest);
    EXECUTE_DATA_TEST(ReduceGray4x4DataTest);

    EXECUTE_DATA_TEST(ShiftBilinearDataTest);

    EXECUTE_DATA_TEST(GetAbsDyRowSumsDataTest);
    EXECUTE_DATA_TEST(GetAbsDxColSumsDataTest);
    EXECUTE_DATA_TEST(GetStatisticDataTest);

    EXECUTE_DATA_TEST(TextureBoostedSaturatedGradientDataTest);

end:

    return result ? 1 : 0;
}

int main(int argc, char* argv[])
{
    Options options(argc, argv);

    if(options.type == Options::Auto)
        return ExecuteAutoTest(options);
    else
        return ExecuteDataTest(options);
}

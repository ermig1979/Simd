/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar,
*               2014-2017 Antonenka Mikhail,
*               2019-2019 Facundo Galan.
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
#include "Test/TestUtils.h"
#include "Test/TestLog.h"
#include "Test/TestString.h"

namespace Test
{
    typedef bool(*AutoTestPtr)();
    typedef bool(*DataTestPtr)(bool create);
    typedef bool(*SpecialTestPtr)();

    struct Group
    {
        String name;
        AutoTestPtr autoTest;
        DataTestPtr dataTest;
        SpecialTestPtr specialTest;
        Group(const String & n, const AutoTestPtr & a, const DataTestPtr & d, const SpecialTestPtr & s)
            : name(n)
            , autoTest(a)
            , dataTest(d)
            , specialTest(s)
        {
        }
    };
    typedef std::vector<Group> Groups;
    Groups g_groups;

#define TEST_ADD_GROUP_00S(name) \
    bool name##SpecialTest(); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, NULL, NULL, name##SpecialTest)); return true; } \
    bool name##AtList = name##AddToList();

#define TEST_ADD_GROUP_A00(name) \
    bool name##AutoTest(); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, name##AutoTest, NULL, NULL)); return true; } \
    bool name##AtList = name##AddToList();

#define TEST_ADD_GROUP_A0S(name) \
    bool name##AutoTest(); \
    bool name##SpecialTest(); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, name##AutoTest, NULL, name##SpecialTest)); return true; } \
    bool name##AtList = name##AddToList();

#define TEST_ADD_GROUP_AD0(name) \
    bool name##AutoTest(); \
    bool name##DataTest(bool create); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, name##AutoTest, name##DataTest, NULL)); return true; } \
    bool name##AtList = name##AddToList();

#define TEST_ADD_GROUP_ADS(name) \
    bool name##AutoTest(); \
    bool name##DataTest(bool create); \
    bool name##SpecialTest(); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, name##AutoTest, name##DataTest, name##SpecialTest)); return true; } \
    bool name##AtList = name##AddToList();

    TEST_ADD_GROUP_AD0(AbsDifference);

    TEST_ADD_GROUP_AD0(AbsDifferenceSum);
    TEST_ADD_GROUP_AD0(AbsDifferenceSumMasked);
    TEST_ADD_GROUP_AD0(AbsDifferenceSums3x3);
    TEST_ADD_GROUP_AD0(AbsDifferenceSums3x3Masked);
    TEST_ADD_GROUP_AD0(SquaredDifferenceSum);
    TEST_ADD_GROUP_AD0(SquaredDifferenceSumMasked);
    TEST_ADD_GROUP_AD0(SquaredDifferenceSum32f);
    TEST_ADD_GROUP_AD0(SquaredDifferenceKahanSum32f);
    TEST_ADD_GROUP_AD0(CosineDistance32f);

    TEST_ADD_GROUP_AD0(AddFeatureDifference);

    TEST_ADD_GROUP_AD0(BgraToBgr);
    TEST_ADD_GROUP_AD0(BgraToGray);
    TEST_ADD_GROUP_A00(BgraToRgb);
    TEST_ADD_GROUP_A00(BgraToRgba);
    TEST_ADD_GROUP_AD0(BgrToGray);
    TEST_ADD_GROUP_AD0(BgrToHsl);
    TEST_ADD_GROUP_AD0(BgrToHsv);
    TEST_ADD_GROUP_A00(BgrToRgb);
    TEST_ADD_GROUP_AD0(GrayToBgr);
    TEST_ADD_GROUP_AD0(Int16ToGray);
    TEST_ADD_GROUP_A00(RgbToGray);
    TEST_ADD_GROUP_A00(RgbaToGray);
    TEST_ADD_GROUP_00S(ConvertImage);

    TEST_ADD_GROUP_AD0(BgraToBayer);
    TEST_ADD_GROUP_AD0(BgrToBayer);

    TEST_ADD_GROUP_AD0(BgrToBgra);
    TEST_ADD_GROUP_AD0(GrayToBgra);
    TEST_ADD_GROUP_A00(RgbToBgra);

    TEST_ADD_GROUP_AD0(BgraToYuv420p);
    TEST_ADD_GROUP_AD0(BgraToYuv422p);
    TEST_ADD_GROUP_AD0(BgraToYuv444p);
    TEST_ADD_GROUP_AD0(BgrToYuv420p);
    TEST_ADD_GROUP_AD0(BgrToYuv422p);
    TEST_ADD_GROUP_AD0(BgrToYuv444p);
    TEST_ADD_GROUP_A00(BgraToYuva420p);

    TEST_ADD_GROUP_AD0(BackgroundGrowRangeSlow);
    TEST_ADD_GROUP_AD0(BackgroundGrowRangeFast);
    TEST_ADD_GROUP_AD0(BackgroundIncrementCount);
    TEST_ADD_GROUP_AD0(BackgroundAdjustRange);
    TEST_ADD_GROUP_AD0(BackgroundAdjustRangeMasked);
    TEST_ADD_GROUP_AD0(BackgroundShiftRange);
    TEST_ADD_GROUP_AD0(BackgroundShiftRangeMasked);
    TEST_ADD_GROUP_AD0(BackgroundInitMask);

    TEST_ADD_GROUP_AD0(BayerToBgr);

    TEST_ADD_GROUP_AD0(BayerToBgra);

    TEST_ADD_GROUP_AD0(Bgr48pToBgra32);

    TEST_ADD_GROUP_AD0(Binarization);
    TEST_ADD_GROUP_AD0(AveragingBinarization);
    TEST_ADD_GROUP_A00(AveragingBinarizationV2);

    TEST_ADD_GROUP_AD0(ConditionalCount8u);
    TEST_ADD_GROUP_AD0(ConditionalCount16i);
    TEST_ADD_GROUP_AD0(ConditionalSum);
    TEST_ADD_GROUP_AD0(ConditionalSquareSum);
    TEST_ADD_GROUP_AD0(ConditionalSquareGradientSum);
    TEST_ADD_GROUP_AD0(ConditionalFill);

    TEST_ADD_GROUP_AD0(ContourMetricsMasked);
    TEST_ADD_GROUP_AD0(ContourAnchors);
    TEST_ADD_GROUP_00S(ContourDetector);

    TEST_ADD_GROUP_AD0(Copy);
    TEST_ADD_GROUP_AD0(CopyFrame);

    TEST_ADD_GROUP_A00(Crc32);
    TEST_ADD_GROUP_AD0(Crc32c);

    TEST_ADD_GROUP_AD0(DeinterleaveUv);
    TEST_ADD_GROUP_AD0(DeinterleaveBgr);
    TEST_ADD_GROUP_AD0(DeinterleaveBgra);

    TEST_ADD_GROUP_AD0(DetectionHaarDetect32fp);
    TEST_ADD_GROUP_AD0(DetectionHaarDetect32fi);
    TEST_ADD_GROUP_AD0(DetectionLbpDetect32fp);
    TEST_ADD_GROUP_AD0(DetectionLbpDetect32fi);
    TEST_ADD_GROUP_AD0(DetectionLbpDetect16ip);
    TEST_ADD_GROUP_AD0(DetectionLbpDetect16ii);
    TEST_ADD_GROUP_00S(Detection);

    TEST_ADD_GROUP_AD0(AlphaBlending);
    TEST_ADD_GROUP_AD0(AlphaFilling);
    TEST_ADD_GROUP_A00(AlphaPremultiply);
    TEST_ADD_GROUP_A00(AlphaUnpremultiply);
    TEST_ADD_GROUP_00S(DrawLine);
    TEST_ADD_GROUP_00S(DrawRectangle);
    TEST_ADD_GROUP_00S(DrawFilledRectangle);
    TEST_ADD_GROUP_00S(DrawPolygon);
    TEST_ADD_GROUP_00S(DrawFilledPolygon);
    TEST_ADD_GROUP_00S(DrawEllipse);
    TEST_ADD_GROUP_00S(DrawCircle);

    TEST_ADD_GROUP_00S(FontDraw);

    TEST_ADD_GROUP_AD0(EdgeBackgroundGrowRangeSlow);
    TEST_ADD_GROUP_AD0(EdgeBackgroundGrowRangeFast);
    TEST_ADD_GROUP_AD0(EdgeBackgroundIncrementCount);
    TEST_ADD_GROUP_AD0(EdgeBackgroundAdjustRange);
    TEST_ADD_GROUP_AD0(EdgeBackgroundAdjustRangeMasked);
    TEST_ADD_GROUP_AD0(EdgeBackgroundShiftRange);
    TEST_ADD_GROUP_AD0(EdgeBackgroundShiftRangeMasked);

    TEST_ADD_GROUP_AD0(Fill);
    TEST_ADD_GROUP_AD0(FillFrame);
    TEST_ADD_GROUP_AD0(FillBgra);
    TEST_ADD_GROUP_AD0(FillBgr);
    TEST_ADD_GROUP_AD0(FillPixel);
    TEST_ADD_GROUP_A00(Fill32f);

    TEST_ADD_GROUP_AD0(Float32ToFloat16);
    TEST_ADD_GROUP_AD0(Float16ToFloat32);
    TEST_ADD_GROUP_AD0(SquaredDifferenceSum16f);
    TEST_ADD_GROUP_AD0(CosineDistance16f);
    TEST_ADD_GROUP_A00(CosineDistancesMxNa16f);
    TEST_ADD_GROUP_A00(CosineDistancesMxNp16f);

    TEST_ADD_GROUP_AD0(Float32ToUint8);
    TEST_ADD_GROUP_AD0(Uint8ToFloat32);

    TEST_ADD_GROUP_A00(Gemm32fNN);
    TEST_ADD_GROUP_A00(Gemm32fNT);

    TEST_ADD_GROUP_A00(ImageSaveToMemory);
    TEST_ADD_GROUP_A00(ImageLoadFromMemory);

    TEST_ADD_GROUP_AD0(MeanFilter3x3);
    TEST_ADD_GROUP_AD0(MedianFilterRhomb3x3);
    TEST_ADD_GROUP_AD0(MedianFilterRhomb5x5);
    TEST_ADD_GROUP_AD0(MedianFilterSquare3x3);
    TEST_ADD_GROUP_AD0(MedianFilterSquare5x5);
    TEST_ADD_GROUP_AD0(GaussianBlur3x3);
    TEST_ADD_GROUP_AD0(AbsGradientSaturatedSum);
    TEST_ADD_GROUP_AD0(LbpEstimate);
    TEST_ADD_GROUP_AD0(NormalizeHistogram);
    TEST_ADD_GROUP_AD0(SobelDx);
    TEST_ADD_GROUP_AD0(SobelDxAbs);
    TEST_ADD_GROUP_AD0(SobelDy);
    TEST_ADD_GROUP_AD0(SobelDyAbs);
    TEST_ADD_GROUP_AD0(ContourMetrics);
    TEST_ADD_GROUP_AD0(Laplace);
    TEST_ADD_GROUP_AD0(LaplaceAbs);
    TEST_ADD_GROUP_A0S(GaussianBlur);

    TEST_ADD_GROUP_AD0(Histogram);
    TEST_ADD_GROUP_AD0(HistogramMasked);
    TEST_ADD_GROUP_AD0(HistogramConditional);
    TEST_ADD_GROUP_AD0(AbsSecondDerivativeHistogram);
    TEST_ADD_GROUP_AD0(ChangeColors);

    TEST_ADD_GROUP_AD0(HogDirectionHistograms);
    TEST_ADD_GROUP_AD0(HogExtractFeatures);
    TEST_ADD_GROUP_AD0(HogDeinterleave);
    TEST_ADD_GROUP_AD0(HogFilterSeparable);

    TEST_ADD_GROUP_AD0(HogLiteExtractFeatures);
    TEST_ADD_GROUP_AD0(HogLiteFilterFeatures);
    TEST_ADD_GROUP_AD0(HogLiteResizeFeatures);
    TEST_ADD_GROUP_AD0(HogLiteCompressFeatures);
    TEST_ADD_GROUP_AD0(HogLiteFilterSeparable);
    TEST_ADD_GROUP_AD0(HogLiteFindMax7x7);
    TEST_ADD_GROUP_AD0(HogLiteCreateMask);

    TEST_ADD_GROUP_00S(ImageMatcher);

    TEST_ADD_GROUP_AD0(Integral);

    TEST_ADD_GROUP_AD0(InterferenceIncrement);
    TEST_ADD_GROUP_AD0(InterferenceIncrementMasked);
    TEST_ADD_GROUP_AD0(InterferenceDecrement);
    TEST_ADD_GROUP_AD0(InterferenceDecrementMasked);

    TEST_ADD_GROUP_AD0(InterleaveUv);
    TEST_ADD_GROUP_AD0(InterleaveBgr);
    TEST_ADD_GROUP_AD0(InterleaveBgra);

    TEST_ADD_GROUP_00S(Motion);

    TEST_ADD_GROUP_AD0(NeuralConvert);
    TEST_ADD_GROUP_AD0(NeuralProductSum);
    TEST_ADD_GROUP_AD0(NeuralAddVectorMultipliedByValue);
    TEST_ADD_GROUP_AD0(NeuralAddVector);
    TEST_ADD_GROUP_AD0(NeuralAddValue);
    TEST_ADD_GROUP_AD0(NeuralRoughSigmoid);
    TEST_ADD_GROUP_AD0(NeuralRoughSigmoid2);
    TEST_ADD_GROUP_AD0(NeuralDerivativeSigmoid);
    TEST_ADD_GROUP_AD0(NeuralRoughTanh);
    TEST_ADD_GROUP_AD0(NeuralDerivativeTanh);
    TEST_ADD_GROUP_AD0(NeuralDerivativeRelu);
    TEST_ADD_GROUP_AD0(NeuralPow);
    TEST_ADD_GROUP_AD0(NeuralUpdateWeights);
    TEST_ADD_GROUP_AD0(NeuralAdaptiveGradientUpdate);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution2x2Forward);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution3x3Forward);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution4x4Forward);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution5x5Forward);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution2x2Backward);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution3x3Backward);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution4x4Backward);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution5x5Backward);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution2x2Sum);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution3x3Sum);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution4x4Sum);
    TEST_ADD_GROUP_AD0(NeuralAddConvolution5x5Sum);
    TEST_ADD_GROUP_AD0(NeuralPooling1x1Max3x3);
    TEST_ADD_GROUP_AD0(NeuralPooling2x2Max2x2);
    TEST_ADD_GROUP_AD0(NeuralPooling2x2Max3x3);
    TEST_ADD_GROUP_AD0(NeuralConvolutionForward);
    TEST_ADD_GROUP_00S(NeuralPredict);
    TEST_ADD_GROUP_00S(NeuralTrain);

    TEST_ADD_GROUP_AD0(OperationBinary8u);
    TEST_ADD_GROUP_AD0(OperationBinary16i);
    TEST_ADD_GROUP_AD0(VectorProduct);

    TEST_ADD_GROUP_AD0(ReduceColor2x2);
    TEST_ADD_GROUP_AD0(ReduceGray2x2);
    TEST_ADD_GROUP_AD0(ReduceGray3x3);
    TEST_ADD_GROUP_AD0(ReduceGray4x4);
    TEST_ADD_GROUP_AD0(ReduceGray5x5);

    TEST_ADD_GROUP_AD0(Reorder16bit);
    TEST_ADD_GROUP_AD0(Reorder32bit);
    TEST_ADD_GROUP_AD0(Reorder64bit);

    TEST_ADD_GROUP_ADS(ResizeBilinear);
    TEST_ADD_GROUP_A00(Resizer);

    TEST_ADD_GROUP_AD0(SegmentationShrinkRegion);
    TEST_ADD_GROUP_AD0(SegmentationFillSingleHoles);
    TEST_ADD_GROUP_AD0(SegmentationChangeIndex);
    TEST_ADD_GROUP_AD0(SegmentationPropagate2x2);

    TEST_ADD_GROUP_AD0(ShiftBilinear);
    TEST_ADD_GROUP_00S(ShiftDetectorRand);
    TEST_ADD_GROUP_00S(ShiftDetectorFile);

    TEST_ADD_GROUP_AD0(GetStatistic);
    TEST_ADD_GROUP_AD0(GetMoments);
    TEST_ADD_GROUP_A00(GetObjectMoments);
    TEST_ADD_GROUP_AD0(GetRowSums);
    TEST_ADD_GROUP_AD0(GetColSums);
    TEST_ADD_GROUP_AD0(GetAbsDyRowSums);
    TEST_ADD_GROUP_AD0(GetAbsDxColSums);
    TEST_ADD_GROUP_AD0(ValueSum);
    TEST_ADD_GROUP_AD0(SquareSum);
    TEST_ADD_GROUP_AD0(SobelDxAbsSum);
    TEST_ADD_GROUP_AD0(SobelDyAbsSum);
    TEST_ADD_GROUP_AD0(LaplaceAbsSum);
    TEST_ADD_GROUP_AD0(ValueSquareSum);
    TEST_ADD_GROUP_AD0(CorrelationSum);

    TEST_ADD_GROUP_AD0(StretchGray2x2);

    TEST_ADD_GROUP_AD0(SvmSumLinear);

#if defined(SIMD_SYNET_ENABLE)
    TEST_ADD_GROUP_A00(SynetAddBias);
    TEST_ADD_GROUP_A00(SynetAdd8i);
    TEST_ADD_GROUP_AD0(SynetEltwiseLayerForward);
    TEST_ADD_GROUP_A00(SynetLrnLayerCrossChannels);
    TEST_ADD_GROUP_A00(SynetShuffleLayerForward);
    TEST_ADD_GROUP_A00(SynetSoftmaxLayerForward);
    TEST_ADD_GROUP_A00(SynetUnaryOperation32fLayerForward);

    TEST_ADD_GROUP_A00(SynetElu32f);
    TEST_ADD_GROUP_A00(SynetHswish32f);
    TEST_ADD_GROUP_A00(SynetMish32f);
    TEST_ADD_GROUP_A00(SynetPreluLayerForward);
    TEST_ADD_GROUP_A00(SynetRelu32f);
    TEST_ADD_GROUP_A00(SynetRestrictRange32f);
    TEST_ADD_GROUP_A00(SynetSigmoid32f);
    TEST_ADD_GROUP_A00(SynetSoftplus32f);
    TEST_ADD_GROUP_A00(SynetTanh32f);

    TEST_ADD_GROUP_A00(SynetConvert32fTo8u);
    TEST_ADD_GROUP_A00(SynetConvert8uTo32f);
    TEST_ADD_GROUP_A00(SynetSetInput);
    TEST_ADD_GROUP_A00(SynetReorderImage);
    TEST_ADD_GROUP_A00(SynetReorderFilter);

    TEST_ADD_GROUP_A00(SynetConvolution8iForward);

    TEST_ADD_GROUP_A00(SynetConvolution32fForward);

    TEST_ADD_GROUP_A00(SynetDeconvolution32fForward);

    TEST_ADD_GROUP_A00(SynetFusedLayerForward0);
    TEST_ADD_GROUP_A00(SynetFusedLayerForward1);
    TEST_ADD_GROUP_A00(SynetFusedLayerForward2);
    TEST_ADD_GROUP_A00(SynetFusedLayerForward3);
    TEST_ADD_GROUP_A00(SynetFusedLayerForward4);
    TEST_ADD_GROUP_A00(SynetFusedLayerForward8);
    TEST_ADD_GROUP_A00(SynetFusedLayerForward9);

    TEST_ADD_GROUP_A00(SynetInnerProduct32fForward);
    TEST_ADD_GROUP_A00(SynetInnerProductLayerForward);
    TEST_ADD_GROUP_A00(SynetInnerProduct8i);

    TEST_ADD_GROUP_A00(SynetMergedConvolution8iForward);

    TEST_ADD_GROUP_A00(SynetMergedConvolution32fForward);

    TEST_ADD_GROUP_A00(SynetPoolingForwardAverage);
    TEST_ADD_GROUP_A00(SynetPoolingForwardMax32f);
    TEST_ADD_GROUP_A00(SynetPoolingForwardMax8u);

    TEST_ADD_GROUP_A00(SynetScaleLayerForward);
    TEST_ADD_GROUP_A00(SynetScale8iForward);
#endif

    TEST_ADD_GROUP_AD0(TextureBoostedSaturatedGradient);
    TEST_ADD_GROUP_AD0(TextureBoostedUv);
    TEST_ADD_GROUP_AD0(TextureGetDifferenceSum);
    TEST_ADD_GROUP_AD0(TexturePerformCompensation);

    TEST_ADD_GROUP_A00(TransformImage);

#if defined(SIMD_SYNET_ENABLE)
    TEST_ADD_GROUP_A00(WinogradKernel1x3Block1x4SetFilter);
    TEST_ADD_GROUP_A00(WinogradKernel1x3Block1x4SetInput);
    TEST_ADD_GROUP_A00(WinogradKernel1x3Block1x4SetOutput);
    TEST_ADD_GROUP_A00(WinogradKernel1x5Block1x4SetFilter);
    TEST_ADD_GROUP_A00(WinogradKernel1x5Block1x4SetInput);
    TEST_ADD_GROUP_A00(WinogradKernel1x5Block1x4SetOutput);
    TEST_ADD_GROUP_A00(WinogradKernel2x2Block2x2SetFilter);
    TEST_ADD_GROUP_A00(WinogradKernel2x2Block2x2SetInput);
    TEST_ADD_GROUP_A00(WinogradKernel2x2Block2x2SetOutput);
    TEST_ADD_GROUP_A00(WinogradKernel2x2Block4x4SetFilter);
    TEST_ADD_GROUP_A00(WinogradKernel2x2Block4x4SetInput);
    TEST_ADD_GROUP_A00(WinogradKernel2x2Block4x4SetOutput);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block2x2SetFilter);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block2x2SetInput);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block2x2SetOutput);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block3x3SetFilter);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block3x3SetInput);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block3x3SetOutput);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block4x4SetFilter);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block4x4SetInput);
    TEST_ADD_GROUP_A00(WinogradKernel3x3Block4x4SetOutput);

    TEST_ADD_GROUP_00S(WinogradKernel1x3Block1x4);
    TEST_ADD_GROUP_00S(WinogradKernel1x5Block1x4);
    TEST_ADD_GROUP_00S(WinogradKernel2x2Block2x2);
    TEST_ADD_GROUP_00S(WinogradKernel2x2Block4x4);
    TEST_ADD_GROUP_00S(WinogradKernel3x3Block2x2);
    TEST_ADD_GROUP_00S(WinogradKernel3x3Block3x3);
    TEST_ADD_GROUP_00S(WinogradKernel3x3Block4x4);
#endif

    TEST_ADD_GROUP_AD0(Yuv444pToBgr);
    TEST_ADD_GROUP_AD0(Yuv422pToBgr);
    TEST_ADD_GROUP_AD0(Yuv420pToBgr);
    TEST_ADD_GROUP_AD0(Yuv444pToHsl);
    TEST_ADD_GROUP_AD0(Yuv444pToHsv);
    TEST_ADD_GROUP_AD0(Yuv444pToHue);
    TEST_ADD_GROUP_AD0(Yuv420pToHue);
    TEST_ADD_GROUP_A00(Yuv444pToRgb);
    TEST_ADD_GROUP_A00(Yuv422pToRgb);
    TEST_ADD_GROUP_A00(Yuv420pToRgb);

    TEST_ADD_GROUP_A00(Yuva420pToBgra);
    TEST_ADD_GROUP_AD0(Yuv444pToBgra);
    TEST_ADD_GROUP_AD0(Yuv422pToBgra);
    TEST_ADD_GROUP_AD0(Yuv420pToBgra);

    class Task
    {
        Groups _groups;
        std::thread _thread;
        volatile double _progress;
    public:
        static volatile bool s_stopped;

        Task(Groups::const_iterator begin, Groups::const_iterator end, bool start)
            : _groups(begin, end)
            , _progress(0)
        {
            if (start)
                _thread = std::thread(&Task::Run, this);
        }

        ~Task()
        {
            if (_thread.joinable())
            {
                _thread.join();
            }
        }

        double Progress() const
        {
            return _progress;
        }

        void Run()
        {
            for (size_t i = 0; i < _groups.size() && !s_stopped; ++i)
            {
                _progress = double(i) / double(_groups.size());
                const Group & group = _groups[i];
                TEST_LOG_SS(Info, group.name << "AutoTest is started :");
                bool result = group.autoTest();
                TEST_LOG_SS(Info, group.name << "AutoTest is finished " << (result ? "successfully." : "with errors!") << std::endl);
                if (!result)
                {
                    s_stopped = true;
                    TEST_LOG_SS(Error, "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl);
                    return;
                }
            }
            _progress = 1.0;
        }
    };
    volatile bool Task::s_stopped = false;
    typedef std::shared_ptr<Task> TaskPtr;
    typedef std::vector<TaskPtr> TaskPtrs;

    inline void Sleep(unsigned int miliseconds)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(miliseconds));
    }

    struct Options
    {
        enum Mode
        {
            Auto,
            Create,
            Verify,
            Special,
        } mode;

        bool help;

        Strings include, exclude;

        String text, html;

        size_t testThreads, workThreads;

        bool printAlign;

        Options(int argc, char* argv[])
            : mode(Auto)
            , help(false)
            , testThreads(0)
            , workThreads(1)
            , printAlign(false)
        {
            for (int i = 1; i < argc; ++i)
            {
                String arg = argv[i];
                if (arg.substr(0, 5) == "-help" || arg.substr(0, 2) == "-?")
                {
                    help = true;
                    break;
                }
                else if (arg.find("-m=") == 0)
                {
                    switch (arg[3])
                    {
                    case 'a': mode = Auto; break;
                    case 'c': mode = Create; break;
                    case 'v': mode = Verify; break;
                    case 's': mode = Special; break;
                    default:
                        TEST_LOG_SS(Error, "Unknown command line options: '" << arg << "'!" << std::endl);
                        exit(1);
                    }
                }
                else if (arg.find("-tt=") == 0)
                {
#if defined(NDEBUG)
                    testThreads = FromString<size_t>(arg.substr(4, arg.size() - 4));
#endif
                }
                else if (arg.find("-fi=") == 0)
                {
                    include.push_back(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-fe=") == 0)
                {
                    exclude.push_back(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-ot=") == 0)
                {
                    text = arg.substr(4, arg.size() - 4);
                }
                else if (arg.find("-oh=") == 0)
                {
                    html = arg.substr(4, arg.size() - 4);
                }
                else if (arg.find("-r=") == 0)
                {
                    ROOT_PATH = arg.substr(3, arg.size() - 3);
                }
                else if (arg.find("-s=") == 0)
                {
                    SOURCE = arg.substr(3, arg.size() - 3);
                }
                else if (arg.find("-o=") == 0)
                {
                    OUTPUT = arg.substr(3, arg.size() - 3);
                }
                else if (arg.find("-c=") == 0)
                {
                    C = FromString<int>(arg.substr(3, arg.size() - 3));
                }
                else if (arg.find("-h=") == 0)
                {
                    H = FromString<int>(arg.substr(3, arg.size() - 3));
                }                
                else if (arg.find("-w=") == 0)
                {
                    W = FromString<int>(arg.substr(3, arg.size() - 3));
                }
                else if (arg.find("-pa=") == 0)
                {
                    printAlign = FromString<bool>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-wt=") == 0)
                {
                    workThreads = FromString<size_t>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-mt=") == 0)
                {
                    MINIMAL_TEST_EXECUTION_TIME = FromString<int>(arg.substr(4, arg.size() - 4))*0.001;
                }
                else if (arg.find("-lc=") == 0)
                {
                    LITTER_CPU_CACHE = FromString<int>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-ri=") == 0)
                {
                    REAL_IMAGE = arg.substr(4, arg.size() - 4);
                }
                else
                {
                    TEST_LOG_SS(Error, "Unknown command line options: '" << arg << "'!" << std::endl);
                    exit(1);
                }
            }
        }

        bool Required(const Group & group) const
        {
            if (mode == Auto && group.autoTest == NULL)
                return false;
            if ((mode == Create || mode == Verify) && group.dataTest == NULL)
                return false;
            if (mode == Special && group.specialTest == NULL)
                return false;
            bool required = include.empty();
            for (size_t i = 0; i < include.size() && !required; ++i)
                if (group.name.find(include[i]) != std::string::npos)
                    required = true;
            for (size_t i = 0; i < exclude.size() && required; ++i)
                if (group.name.find(exclude[i]) != std::string::npos)
                    required = false;
            return required;
        }
    };

    int MakeAutoTests(const Groups & groups, const Options & options)
    {
        if (options.testThreads > 0)
        {
            TEST_LOG_SS(Info, "Test threads count = " << options.testThreads);

            Test::Log::s_log.SetLevel(Test::Log::Error);

            Test::TaskPtrs tasks;
            size_t n = options.testThreads;
            size_t total = groups.size();
            size_t block = (total + n - 1) / n;
            for (size_t i = 0; i < n; ++i)
            {
                size_t begin = i * block;
                size_t end = std::min(total, begin + block);
                tasks.push_back(Test::TaskPtr(new Test::Task(groups.begin() + begin, groups.begin() + end, true)));
            }

            std::cout << std::endl;
            double progress;
            do
            {
                progress = 0;
                for (size_t i = 0; i < tasks.size(); ++i)
                    progress += tasks[i]->Progress();
                progress /= double(tasks.size());
                std::cout << "\rTest progress: " << std::fixed << std::setprecision(1) << progress*100.0 << "%" << std::flush;
                Test::Sleep(40);
            } while (progress < 1.0 && !Test::Task::s_stopped);
            std::cout << std::endl << std::endl;

            Test::Log::s_log.SetLevel(Test::Log::Info);
        }
        else
        {
            Test::Task task(groups.begin(), groups.end(), false);
            task.Run();
        }

        if (Test::Task::s_stopped)
            return 1;

        TEST_LOG_SS(Info, "ALL TESTS ARE FINISHED SUCCESSFULLY!" << std::endl);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, Test::PerformanceMeasurerStorage::s_storage.TextReport(options.printAlign, false) << SimdPerformanceStatistic());
        if (!options.html.empty())
            Test::PerformanceMeasurerStorage::s_storage.HtmlReport(options.html, options.printAlign);
#endif
        return 0;
    }

    int MakeDataTests(const Groups & groups, const Options & options)
    {
        for (const Test::Group & group : groups)
        {
            bool create = options.mode == Test::Options::Create;
            TEST_LOG_SS(Info, group.name << "DataTest - data " << (create ? "creation" : "verification") << " is started :");
            bool result = group.dataTest(create);
            TEST_LOG_SS(Info, group.name << "DataTest - data " << (create ? "creation" : "verification") << " is finished " << (result ? "successfully." : "with errors!") << std::endl);
            if (!result)
            {
                TEST_LOG_SS(Error, "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl);
                return 1;
            }
        }
        TEST_LOG_SS(Info, "ALL TESTS ARE FINISHED SUCCESSFULLY!" << std::endl);

        return 0;
    }

    int MakeSpecialTests(const Groups & groups, const Options & options)
    {
        for (const Test::Group & group : groups)
        {
            TEST_LOG_SS(Info, group.name << "SpecialTest is started :");
            bool result = group.specialTest();
            TEST_LOG_SS(Info, group.name << "SpecialTest is finished " << (result ? "successfully." : "with errors!") << std::endl);
            if (!result)
            {
                TEST_LOG_SS(Error, "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl);
                return 1;
            }
        }
        TEST_LOG_SS(Info, "ALL TESTS ARE FINISHED SUCCESSFULLY!" << std::endl);

        return 0;
    }

    int PrintHelp()
    {
        std::cout << "Test framework of Simd Library." << std::endl << std::endl;
        std::cout << "Using example:" << std::endl << std::endl;
        std::cout << "  ./Test -m=a -tt=1 -fi=Sobel -ot=log.txt" << std::endl << std::endl;
        std::cout << "Where next parameters were used:" << std::endl << std::endl;
        std::cout << "-m=a         - a auto checking mode which includes performance testing" << std::endl;
        std::cout << "               (only for library built in Release mode)." << std::endl;
        std::cout << "               In this case different implementations of each functions" << std::endl;
        std::cout << "               will be compared between themselves " << std::endl;
        std::cout << "               (for example a scalar implementation and implementations" << std::endl;
        std::cout << "               with using of different SIMD instructions such as SSE2, " << std::endl;
        std::cout << "               AVX2, and other). Also it can be: " << std::endl;
        std::cout << "               -m=c - creation of test data for cross-platform testing), " << std::endl;
        std::cout << "               -m=v - cross - platform testing with using of early " << std::endl;
        std::cout << "               prepared test data)," << std::endl;
        std::cout << "               -m=s - running of special tests." << std::endl << std::endl;
        std::cout << "-tt=1        - a number of test threads." << std::endl;
        std::cout << "-fi=Sobel    - an include filter. In current case will be tested only" << std::endl;
        std::cout << "               functions which contain word 'Sobel' in their names." << std::endl;
        std::cout << "               If you miss this parameter then full testing will be" << std::endl;
        std::cout << "               performed. You can use several include filters - " << std::endl;
        std::cout << "               function name has to satisfy at least one of them. " << std::endl << std::endl;
        std::cout << "-ot=log.txt  - a file name with test report (in TEXT format)." << std::endl;
        std::cout << "               The test's report also will be output to console." << std::endl << std::endl;
        std::cout << "Also you can use parameters: " << std::endl << std::endl;
        std::cout << "    -help or -?   to print this help message." << std::endl << std::endl;
        std::cout << "    -r=../..      to set project root directory." << std::endl << std::endl;
        std::cout << "    -pa=1         to print alignment statistics." << std::endl << std::endl;
        std::cout << "    -c=512        a number of channels in test image for performance testing." << std::endl << std::endl;
        std::cout << "    -h=1080       a height of test image for performance testing." << std::endl << std::endl;
        std::cout << "    -w=1920       a width of test image for performance testing." << std::endl << std::endl;
        std::cout << "    -oh=log.html  a file name with test report (in HTML format)." << std::endl << std::endl;
        std::cout << "    -s=sample.avi a video source (Simd::Motion test)." << std::endl << std::endl;
        std::cout << "    -o=output.avi an annotated video output (Simd::Motion test)." << std::endl << std::endl;
        std::cout << "    -wt=1         a thread number used to parallelize algorithms." << std::endl << std::endl;
        std::cout << "    -fe=Abs       an exclude filter to exclude some tests." << std::endl << std::endl;
        std::cout << "    -mt=100       a minimal test execution time (in milliseconds)." << std::endl << std::endl;
        std::cout << "    -lc=1         to litter CPU cache between test runs." << std::endl << std::endl;
        std::cout << "    -ri=city.jpg  a name of real image used in some tests." << std::endl << std::endl;
        std::cout << "                  The image have to be placed in ./data/image directory." << std::endl << std::endl;
        return 0;
    }

    String ROOT_PATH = "../..";
    String SOURCE = "";
    String OUTPUT = "";
    String REAL_IMAGE = "";

#ifdef TEST_PERFORMANCE_TEST_ENABLE
    int C = 512;
    int H = 1080;
    int W = 1920;
#else
    int C = 32;
    int H = 96;
    int W = 128;
#endif
    double MINIMAL_TEST_EXECUTION_TIME = 0.1;
    int LITTER_CPU_CACHE = 0;

    void CheckCpp();
}

int main(int argc, char* argv[])
{
    //Test::CheckCpp();

    Test::Options options(argc, argv);

    if (options.help)
        return Test::PrintHelp();

    if (!options.text.empty())
        Test::Log::s_log.SetLogFile(options.text);

    Test::Groups groups;
    for (const Test::Group & group : Test::g_groups)
        if (options.Required(group))
            groups.push_back(group);
    if (groups.empty())
    {
        std::stringstream ss;
        ss << "There are not any suitable tests for current filters! " << std::endl;
        ss << "  Include filters: " << std::endl;
        for (size_t i = 0; i < options.include.size(); ++i)
            ss << "'" << options.include[i] << "' ";
        ss << std::endl;
        ss << "  Exclude filters: " << std::endl;
        for (size_t i = 0; i < options.exclude.size(); ++i)
            ss << "'" << options.exclude[i] << "' ";
        ss << std::endl;
        TEST_LOG_SS(Error, ss.str());
        return 1;
    }

    ::SimdSetThreadNumber(options.workThreads);

    switch (options.mode)
    {
    case Test::Options::Auto:
        return Test::MakeAutoTests(groups, options);
    case Test::Options::Create:
    case Test::Options::Verify:
        return Test::MakeDataTests(groups, options);
    case Test::Options::Special:
        return Test::MakeSpecialTests(groups, options);
    default:
        return 0;
    }
}

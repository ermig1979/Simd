/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar,
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
#include "Test/TestCompare.h"
#include "Test/TestLog.h"
#include "Test/TestString.h"

#if defined(_MSC_VER)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace Test
{
    typedef bool(*AutoTestPtr)();
    typedef bool(*SpecialTestPtr)();

    struct Group
    {
        String name;
        AutoTestPtr autoTest;
        SpecialTestPtr specialTest;
        double time;
        Group(const String & n, const AutoTestPtr & a, const SpecialTestPtr & s)
            : name(n)
            , autoTest(a)
            , specialTest(s)
            , time(0.0)
        {
        }
    };
    typedef std::vector<Group> Groups;
    Groups g_groups;

#define TEST_ADD_GROUP_0S(name) \
    bool name##SpecialTest(); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, NULL, name##SpecialTest)); return true; } \
    bool name##AtList = name##AddToList();

#define TEST_ADD_GROUP_A0(name) \
    bool name##AutoTest(); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, name##AutoTest, NULL)); return true; } \
    bool name##AtList = name##AddToList();

#define TEST_ADD_GROUP_AS(name) \
    bool name##AutoTest(); \
    bool name##SpecialTest(); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, name##AutoTest, name##SpecialTest)); return true; } \
    bool name##AtList = name##AddToList();

    TEST_ADD_GROUP_A0(AbsDifference);

    TEST_ADD_GROUP_A0(AbsDifferenceSum);
    TEST_ADD_GROUP_A0(AbsDifferenceSumMasked);
    TEST_ADD_GROUP_A0(AbsDifferenceSums3x3);
    TEST_ADD_GROUP_A0(AbsDifferenceSums3x3Masked);
    TEST_ADD_GROUP_A0(SquaredDifferenceSum);
    TEST_ADD_GROUP_A0(SquaredDifferenceSumMasked);
    TEST_ADD_GROUP_A0(SquaredDifferenceSum32f);
    TEST_ADD_GROUP_A0(SquaredDifferenceKahanSum32f);
    TEST_ADD_GROUP_A0(CosineDistance32f);

    TEST_ADD_GROUP_A0(AddFeatureDifference);

    TEST_ADD_GROUP_A0(BgraToBgr);
    TEST_ADD_GROUP_A0(BgraToGray);
    TEST_ADD_GROUP_A0(BgraToRgb);
    TEST_ADD_GROUP_A0(BgraToRgba);
    TEST_ADD_GROUP_A0(BgrToGray);
    TEST_ADD_GROUP_A0(BgrToHsl);
    TEST_ADD_GROUP_A0(BgrToHsv);
    TEST_ADD_GROUP_A0(BgrToRgb);
    TEST_ADD_GROUP_A0(GrayToBgr);
    TEST_ADD_GROUP_A0(Int16ToGray);
    TEST_ADD_GROUP_A0(RgbToGray);
    TEST_ADD_GROUP_A0(RgbaToGray);
    TEST_ADD_GROUP_0S(ConvertImage);

    TEST_ADD_GROUP_A0(BgraToBayer);
    TEST_ADD_GROUP_A0(BgrToBayer);

    TEST_ADD_GROUP_A0(BgrToBgra);
    TEST_ADD_GROUP_A0(GrayToBgra);
    TEST_ADD_GROUP_A0(RgbToBgra);

    TEST_ADD_GROUP_A0(BgraToYuv420p);
    TEST_ADD_GROUP_A0(BgraToYuv420pV2);
    TEST_ADD_GROUP_A0(BgraToYuv422p);
    TEST_ADD_GROUP_A0(BgraToYuv422pV2);
    TEST_ADD_GROUP_A0(BgraToYuv444p);
    TEST_ADD_GROUP_A0(BgraToYuv444pV2);
    TEST_ADD_GROUP_A0(BgrToYuv420p);
    TEST_ADD_GROUP_A0(BgrToYuv422p);
    TEST_ADD_GROUP_A0(BgrToYuv444p);
    TEST_ADD_GROUP_A0(Uyvy422ToYuv420p);
    TEST_ADD_GROUP_A0(BgraToYuva420p);
    TEST_ADD_GROUP_A0(BgraToYuva420pV2);

    TEST_ADD_GROUP_A0(BackgroundGrowRangeSlow);
    TEST_ADD_GROUP_A0(BackgroundGrowRangeFast);
    TEST_ADD_GROUP_A0(BackgroundIncrementCount);
    TEST_ADD_GROUP_A0(BackgroundAdjustRange);
    TEST_ADD_GROUP_A0(BackgroundAdjustRangeMasked);
    TEST_ADD_GROUP_A0(BackgroundShiftRange);
    TEST_ADD_GROUP_A0(BackgroundShiftRangeMasked);
    TEST_ADD_GROUP_A0(BackgroundInitMask);

    TEST_ADD_GROUP_A0(Base64Decode);
    TEST_ADD_GROUP_A0(Base64Encode);

    TEST_ADD_GROUP_A0(BayerToBgr);

    TEST_ADD_GROUP_A0(BayerToBgra);

    TEST_ADD_GROUP_A0(Float32ToBFloat16);
    TEST_ADD_GROUP_A0(BFloat16ToFloat32);

    TEST_ADD_GROUP_A0(Bgr48pToBgra32);

    TEST_ADD_GROUP_A0(Binarization);
    TEST_ADD_GROUP_A0(AveragingBinarization);
    TEST_ADD_GROUP_A0(AveragingBinarizationV2);

    TEST_ADD_GROUP_A0(ConditionalCount8u);
    TEST_ADD_GROUP_A0(ConditionalCount16i);
    TEST_ADD_GROUP_A0(ConditionalSum);
    TEST_ADD_GROUP_A0(ConditionalSquareSum);
    TEST_ADD_GROUP_A0(ConditionalSquareGradientSum);
    TEST_ADD_GROUP_A0(ConditionalFill);

    TEST_ADD_GROUP_A0(ContourMetricsMasked);
    TEST_ADD_GROUP_A0(ContourAnchors);
    TEST_ADD_GROUP_0S(ContourDetector);

    TEST_ADD_GROUP_A0(Copy);
    TEST_ADD_GROUP_A0(CopyFrame);

    TEST_ADD_GROUP_A0(Crc32);
    TEST_ADD_GROUP_A0(Crc32c);

    TEST_ADD_GROUP_A0(DescrIntEncode32f);
    TEST_ADD_GROUP_A0(DescrIntEncode16f);
    TEST_ADD_GROUP_A0(DescrIntDecode32f);
    TEST_ADD_GROUP_A0(DescrIntDecode16f);
    TEST_ADD_GROUP_A0(DescrIntCosineDistance);
    TEST_ADD_GROUP_A0(DescrIntCosineDistancesMxNa);
    TEST_ADD_GROUP_A0(DescrIntCosineDistancesMxNp);

    TEST_ADD_GROUP_A0(DeinterleaveUv);
    TEST_ADD_GROUP_A0(DeinterleaveBgr);
    TEST_ADD_GROUP_A0(DeinterleaveBgra);

    TEST_ADD_GROUP_A0(DetectionHaarDetect32fp);
    TEST_ADD_GROUP_A0(DetectionHaarDetect32fi);
    TEST_ADD_GROUP_A0(DetectionLbpDetect32fp);
    TEST_ADD_GROUP_A0(DetectionLbpDetect32fi);
    TEST_ADD_GROUP_A0(DetectionLbpDetect16ip);
    TEST_ADD_GROUP_A0(DetectionLbpDetect16ii);
    TEST_ADD_GROUP_0S(Detection);

    TEST_ADD_GROUP_A0(AlphaBlending);
    TEST_ADD_GROUP_A0(AlphaBlending2x);
    TEST_ADD_GROUP_A0(AlphaBlendingBgraToYuv420p);
    TEST_ADD_GROUP_A0(AlphaBlendingUniform);
    TEST_ADD_GROUP_A0(AlphaFilling);
    TEST_ADD_GROUP_A0(AlphaPremultiply);
    TEST_ADD_GROUP_A0(AlphaUnpremultiply);
    TEST_ADD_GROUP_0S(DrawLine);
    TEST_ADD_GROUP_0S(DrawRectangle);
    TEST_ADD_GROUP_0S(DrawFilledRectangle);
    TEST_ADD_GROUP_0S(DrawPolygon);
    TEST_ADD_GROUP_0S(DrawFilledPolygon);
    TEST_ADD_GROUP_0S(DrawEllipse);
    TEST_ADD_GROUP_0S(DrawCircle);

    TEST_ADD_GROUP_0S(FontDraw);

    TEST_ADD_GROUP_A0(EdgeBackgroundGrowRangeSlow);
    TEST_ADD_GROUP_A0(EdgeBackgroundGrowRangeFast);
    TEST_ADD_GROUP_A0(EdgeBackgroundIncrementCount);
    TEST_ADD_GROUP_A0(EdgeBackgroundAdjustRange);
    TEST_ADD_GROUP_A0(EdgeBackgroundAdjustRangeMasked);
    TEST_ADD_GROUP_A0(EdgeBackgroundShiftRange);
    TEST_ADD_GROUP_A0(EdgeBackgroundShiftRangeMasked);

    TEST_ADD_GROUP_A0(Fill);
    TEST_ADD_GROUP_A0(FillFrame);
    TEST_ADD_GROUP_A0(FillBgra);
    TEST_ADD_GROUP_A0(FillBgr);
    TEST_ADD_GROUP_A0(FillPixel);
    TEST_ADD_GROUP_A0(Fill32f);

    TEST_ADD_GROUP_A0(Float32ToFloat16);
    TEST_ADD_GROUP_A0(Float16ToFloat32);
    TEST_ADD_GROUP_A0(SquaredDifferenceSum16f);
    TEST_ADD_GROUP_A0(CosineDistance16f);
    TEST_ADD_GROUP_A0(CosineDistancesMxNa16f);
    TEST_ADD_GROUP_AS(CosineDistancesMxNp16f);
    TEST_ADD_GROUP_A0(VectorNormNa16f);
    TEST_ADD_GROUP_A0(VectorNormNp16f);

    TEST_ADD_GROUP_A0(Float32ToUint8);
    TEST_ADD_GROUP_A0(Uint8ToFloat32);

    TEST_ADD_GROUP_A0(Gemm32fNN);
    TEST_ADD_GROUP_A0(Gemm32fNT);

    TEST_ADD_GROUP_A0(ImageSaveToMemory);
    TEST_ADD_GROUP_A0(Nv12SaveAsJpegToMemory);
    TEST_ADD_GROUP_A0(Yuv420pSaveAsJpegToMemory);
    TEST_ADD_GROUP_AS(ImageLoadFromMemory);

    TEST_ADD_GROUP_A0(MeanFilter3x3);
    TEST_ADD_GROUP_A0(MedianFilterRhomb3x3);
    TEST_ADD_GROUP_A0(MedianFilterRhomb5x5);
    TEST_ADD_GROUP_A0(MedianFilterSquare3x3);
    TEST_ADD_GROUP_A0(MedianFilterSquare5x5);
    TEST_ADD_GROUP_A0(GaussianBlur3x3);
    TEST_ADD_GROUP_A0(AbsGradientSaturatedSum);
    TEST_ADD_GROUP_A0(LbpEstimate);
    TEST_ADD_GROUP_A0(NormalizeHistogram);
    TEST_ADD_GROUP_A0(SobelDx);
    TEST_ADD_GROUP_A0(SobelDxAbs);
    TEST_ADD_GROUP_A0(SobelDy);
    TEST_ADD_GROUP_A0(SobelDyAbs);
    TEST_ADD_GROUP_A0(ContourMetrics);
    TEST_ADD_GROUP_A0(Laplace);
    TEST_ADD_GROUP_A0(LaplaceAbs);
    TEST_ADD_GROUP_AS(GaussianBlur);
    TEST_ADD_GROUP_A0(RecursiveBilateralFilter);

    TEST_ADD_GROUP_A0(Histogram);
    TEST_ADD_GROUP_A0(HistogramMasked);
    TEST_ADD_GROUP_A0(HistogramConditional);
    TEST_ADD_GROUP_A0(AbsSecondDerivativeHistogram);
    TEST_ADD_GROUP_A0(ChangeColors);

    TEST_ADD_GROUP_A0(HogDirectionHistograms);
    TEST_ADD_GROUP_A0(HogExtractFeatures);
    TEST_ADD_GROUP_A0(HogDeinterleave);
    TEST_ADD_GROUP_A0(HogFilterSeparable);

    TEST_ADD_GROUP_A0(HogLiteExtractFeatures);
    TEST_ADD_GROUP_A0(HogLiteFilterFeatures);
    TEST_ADD_GROUP_A0(HogLiteResizeFeatures);
    TEST_ADD_GROUP_A0(HogLiteCompressFeatures);
    TEST_ADD_GROUP_A0(HogLiteFilterSeparable);
    TEST_ADD_GROUP_A0(HogLiteFindMax7x7);
    TEST_ADD_GROUP_A0(HogLiteCreateMask);

    TEST_ADD_GROUP_0S(ImageMatcher);

    TEST_ADD_GROUP_A0(Integral);

    TEST_ADD_GROUP_A0(InterferenceIncrement);
    TEST_ADD_GROUP_A0(InterferenceIncrementMasked);
    TEST_ADD_GROUP_A0(InterferenceDecrement);
    TEST_ADD_GROUP_A0(InterferenceDecrementMasked);

    TEST_ADD_GROUP_A0(InterleaveUv);
    TEST_ADD_GROUP_A0(InterleaveBgr);
    TEST_ADD_GROUP_A0(InterleaveBgra);

    TEST_ADD_GROUP_0S(Motion);

    TEST_ADD_GROUP_A0(NeuralConvert);
    TEST_ADD_GROUP_A0(NeuralProductSum);
    TEST_ADD_GROUP_A0(NeuralAddVectorMultipliedByValue);
    TEST_ADD_GROUP_A0(NeuralAddVector);
    TEST_ADD_GROUP_A0(NeuralAddValue);
    TEST_ADD_GROUP_A0(NeuralRoughSigmoid);
    TEST_ADD_GROUP_A0(NeuralRoughSigmoid2);
    TEST_ADD_GROUP_A0(NeuralDerivativeSigmoid);
    TEST_ADD_GROUP_A0(NeuralRoughTanh);
    TEST_ADD_GROUP_A0(NeuralDerivativeTanh);
    TEST_ADD_GROUP_A0(NeuralDerivativeRelu);
    TEST_ADD_GROUP_A0(NeuralPow);
    TEST_ADD_GROUP_A0(NeuralUpdateWeights);
    TEST_ADD_GROUP_A0(NeuralAdaptiveGradientUpdate);
    TEST_ADD_GROUP_A0(NeuralPooling1x1Max3x3);
    TEST_ADD_GROUP_A0(NeuralPooling2x2Max2x2);
    TEST_ADD_GROUP_A0(NeuralPooling2x2Max3x3);
    TEST_ADD_GROUP_0S(NeuralPredict);
    TEST_ADD_GROUP_0S(NeuralTrain);

    TEST_ADD_GROUP_A0(NeuralAddConvolution2x2Forward);
    TEST_ADD_GROUP_A0(NeuralAddConvolution3x3Forward);
    TEST_ADD_GROUP_A0(NeuralAddConvolution4x4Forward);
    TEST_ADD_GROUP_A0(NeuralAddConvolution5x5Forward);
    TEST_ADD_GROUP_A0(NeuralAddConvolution2x2Backward);
    TEST_ADD_GROUP_A0(NeuralAddConvolution3x3Backward);
    TEST_ADD_GROUP_A0(NeuralAddConvolution4x4Backward);
    TEST_ADD_GROUP_A0(NeuralAddConvolution5x5Backward);
    TEST_ADD_GROUP_A0(NeuralAddConvolution2x2Sum);
    TEST_ADD_GROUP_A0(NeuralAddConvolution3x3Sum);
    TEST_ADD_GROUP_A0(NeuralAddConvolution4x4Sum);
    TEST_ADD_GROUP_A0(NeuralAddConvolution5x5Sum);
    TEST_ADD_GROUP_A0(NeuralConvolutionForward);

    TEST_ADD_GROUP_A0(OperationBinary8u);
    TEST_ADD_GROUP_A0(OperationBinary16i);
    TEST_ADD_GROUP_A0(VectorProduct);

    TEST_ADD_GROUP_A0(ReduceColor2x2);
    TEST_ADD_GROUP_A0(ReduceGray2x2);
    TEST_ADD_GROUP_A0(ReduceGray3x3);
    TEST_ADD_GROUP_A0(ReduceGray4x4);
    TEST_ADD_GROUP_A0(ReduceGray5x5);

    TEST_ADD_GROUP_A0(Reorder16bit);
    TEST_ADD_GROUP_A0(Reorder32bit);
    TEST_ADD_GROUP_A0(Reorder64bit);

    TEST_ADD_GROUP_AS(ResizeBilinear);
    TEST_ADD_GROUP_A0(Resizer);
    TEST_ADD_GROUP_0S(ResizeYuv420p);

    TEST_ADD_GROUP_A0(SegmentationShrinkRegion);
    TEST_ADD_GROUP_A0(SegmentationFillSingleHoles);
    TEST_ADD_GROUP_A0(SegmentationChangeIndex);
    TEST_ADD_GROUP_A0(SegmentationPropagate2x2);

    TEST_ADD_GROUP_A0(ShiftBilinear);
    TEST_ADD_GROUP_0S(ShiftDetectorRand);
    TEST_ADD_GROUP_0S(ShiftDetectorFile);

    TEST_ADD_GROUP_A0(GetStatistic);
    TEST_ADD_GROUP_A0(GetMoments);
    TEST_ADD_GROUP_A0(GetObjectMoments);
    TEST_ADD_GROUP_A0(GetRowSums);
    TEST_ADD_GROUP_A0(GetColSums);
    TEST_ADD_GROUP_A0(GetAbsDyRowSums);
    TEST_ADD_GROUP_A0(GetAbsDxColSums);
    TEST_ADD_GROUP_A0(ValueSum);
    TEST_ADD_GROUP_A0(SquareSum);
    TEST_ADD_GROUP_A0(SobelDxAbsSum);
    TEST_ADD_GROUP_A0(SobelDyAbsSum);
    TEST_ADD_GROUP_A0(LaplaceAbsSum);
    TEST_ADD_GROUP_A0(ValueSquareSum);
    TEST_ADD_GROUP_A0(ValueSquareSums);
    TEST_ADD_GROUP_A0(CorrelationSum);

    TEST_ADD_GROUP_A0(StretchGray2x2);

    TEST_ADD_GROUP_A0(SvmSumLinear);

#if defined(SIMD_SYNET_ENABLE)
    TEST_ADD_GROUP_A0(SynetAddBias);
    TEST_ADD_GROUP_A0(SynetAdd8i);
    TEST_ADD_GROUP_A0(SynetEltwiseLayerForward);
    TEST_ADD_GROUP_A0(SynetLrnLayerCrossChannels);
    TEST_ADD_GROUP_A0(SynetShuffleLayerForward);

    TEST_ADD_GROUP_A0(SynetElu32f);
    TEST_ADD_GROUP_A0(SynetGelu32f);
    TEST_ADD_GROUP_A0(SynetHardSigmoid32f);
    TEST_ADD_GROUP_A0(SynetHswish32f);
    TEST_ADD_GROUP_A0(SynetMish32f);
    TEST_ADD_GROUP_A0(SynetPreluLayerForward);
    TEST_ADD_GROUP_A0(SynetRelu32f);
    TEST_ADD_GROUP_A0(SynetRestrictRange32f);
    TEST_ADD_GROUP_A0(SynetSigmoid32f);
    TEST_ADD_GROUP_A0(SynetSoftplus32f);
    TEST_ADD_GROUP_A0(SynetSwish32f);
    TEST_ADD_GROUP_A0(SynetTanh32f);

    TEST_ADD_GROUP_A0(SynetConvert32fTo8u);
    TEST_ADD_GROUP_A0(SynetConvert8uTo32f);
    TEST_ADD_GROUP_A0(SynetSetInput);

    TEST_ADD_GROUP_A0(SynetConvolution8iForward);

    TEST_ADD_GROUP_A0(SynetConvolution32fForward);

    TEST_ADD_GROUP_A0(SynetDeconvolution32fForward);

    TEST_ADD_GROUP_A0(SynetFusedLayerForward0);
    TEST_ADD_GROUP_A0(SynetFusedLayerForward1);
    TEST_ADD_GROUP_A0(SynetFusedLayerForward2);
    TEST_ADD_GROUP_A0(SynetFusedLayerForward3);
    TEST_ADD_GROUP_A0(SynetFusedLayerForward4);
    TEST_ADD_GROUP_A0(SynetFusedLayerForward8);
    TEST_ADD_GROUP_A0(SynetFusedLayerForward9);

    TEST_ADD_GROUP_A0(SynetInnerProduct32fForward);
    TEST_ADD_GROUP_A0(SynetInnerProductLayerForward);
    TEST_ADD_GROUP_A0(SynetInnerProduct8i);

    TEST_ADD_GROUP_A0(SynetMergedConvolution8iForward);

    TEST_ADD_GROUP_A0(SynetMergedConvolution32fForward);

    TEST_ADD_GROUP_A0(SynetNormalizeLayerForward);
    TEST_ADD_GROUP_A0(SynetNormalizeLayerForwardV2);
    TEST_ADD_GROUP_A0(SynetNormalizeLayerForwardV3);

    TEST_ADD_GROUP_A0(SynetPermute);

    TEST_ADD_GROUP_A0(SynetPoolingAverage);
    TEST_ADD_GROUP_A0(SynetPoolingMax32f);
    TEST_ADD_GROUP_A0(SynetPoolingMax8u);

    TEST_ADD_GROUP_A0(SynetScaleLayerForward);
    TEST_ADD_GROUP_A0(SynetScale8iForward);

    TEST_ADD_GROUP_A0(SynetSoftmaxLayerForward);

    TEST_ADD_GROUP_A0(SynetUnaryOperation32f);
#endif

    TEST_ADD_GROUP_A0(TextureBoostedSaturatedGradient);
    TEST_ADD_GROUP_A0(TextureBoostedUv);
    TEST_ADD_GROUP_A0(TextureGetDifferenceSum);
    TEST_ADD_GROUP_A0(TexturePerformCompensation);

    TEST_ADD_GROUP_A0(TransformImage);

    TEST_ADD_GROUP_A0(Uyvy422ToBgr);

    TEST_ADD_GROUP_A0(WarpAffine);
#ifdef SIMD_OPENCV_ENABLE
    TEST_ADD_GROUP_0S(WarpAffineOpenCv);
#endif

#if defined(SIMD_SYNET_ENABLE)
    TEST_ADD_GROUP_A0(WinogradKernel1x3Block1x4SetFilter);
    TEST_ADD_GROUP_A0(WinogradKernel1x3Block1x4SetInput);
    TEST_ADD_GROUP_A0(WinogradKernel1x3Block1x4SetOutput);
    TEST_ADD_GROUP_A0(WinogradKernel1x5Block1x4SetFilter);
    TEST_ADD_GROUP_A0(WinogradKernel1x5Block1x4SetInput);
    TEST_ADD_GROUP_A0(WinogradKernel1x5Block1x4SetOutput);
    TEST_ADD_GROUP_A0(WinogradKernel2x2Block2x2SetFilter);
    TEST_ADD_GROUP_A0(WinogradKernel2x2Block2x2SetInput);
    TEST_ADD_GROUP_A0(WinogradKernel2x2Block2x2SetOutput);
    TEST_ADD_GROUP_A0(WinogradKernel2x2Block4x4SetFilter);
    TEST_ADD_GROUP_A0(WinogradKernel2x2Block4x4SetInput);
    TEST_ADD_GROUP_A0(WinogradKernel2x2Block4x4SetOutput);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block2x2SetFilter);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block2x2SetInput);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block2x2SetOutput);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block3x3SetFilter);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block3x3SetInput);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block3x3SetOutput);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block4x4SetFilter);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block4x4SetInput);
    TEST_ADD_GROUP_A0(WinogradKernel3x3Block4x4SetOutput);

    TEST_ADD_GROUP_0S(WinogradKernel1x3Block1x4);
    TEST_ADD_GROUP_0S(WinogradKernel1x5Block1x4);
    TEST_ADD_GROUP_0S(WinogradKernel2x2Block2x2);
    TEST_ADD_GROUP_0S(WinogradKernel2x2Block4x4);
    TEST_ADD_GROUP_0S(WinogradKernel3x3Block2x2);
    TEST_ADD_GROUP_0S(WinogradKernel3x3Block3x3);
    TEST_ADD_GROUP_0S(WinogradKernel3x3Block4x4);
#endif

    TEST_ADD_GROUP_A0(Yuv444pToBgr);
    TEST_ADD_GROUP_A0(Yuv422pToBgr);
    TEST_ADD_GROUP_AS(Yuv420pToBgr);
    TEST_ADD_GROUP_A0(Yuv444pToHsl);
    TEST_ADD_GROUP_A0(Yuv444pToHsv);
    TEST_ADD_GROUP_A0(Yuv444pToHue);
    TEST_ADD_GROUP_A0(Yuv420pToHue);
    TEST_ADD_GROUP_A0(Yuv444pToRgb);
    TEST_ADD_GROUP_A0(Yuv422pToRgb);
    TEST_ADD_GROUP_A0(Yuv420pToRgb);
    TEST_ADD_GROUP_A0(Yuv420pToUyvy422);

    TEST_ADD_GROUP_A0(Yuva420pToBgra);
    TEST_ADD_GROUP_A0(Yuva444pToBgraV2);
    TEST_ADD_GROUP_A0(Yuv444pToBgra);
    TEST_ADD_GROUP_A0(Yuv444pToBgraV2);
    TEST_ADD_GROUP_A0(Yuv422pToBgra);
    TEST_ADD_GROUP_A0(Yuv420pToBgra);
    TEST_ADD_GROUP_A0(Yuv420pToBgraV2);

    class Task
    {
        Group * _groups;
        size_t _size;
        std::thread _thread;
        volatile double _progress;
    public:
        static volatile bool s_stopped;

        Task(Group * groups, size_t size, bool start)
            : _groups(groups)
            , _size(size)
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
            for (size_t i = 0; i < _size && !s_stopped; ++i)
            {
                _progress = double(i) / double(_size);
                Group & group = _groups[i];
                TEST_LOG_SS(Info, group.name << "AutoTest is started :");
                group.time = GetTime();
                bool result = RunGroup(group);
                group.time = GetTime() - group.time;
                if (result)
                {
                    TEST_LOG_SS(Info, group.name << "AutoTest is finished successfully in " << ToString(group.time, 1, false) << " s." << std::endl);
                }
                else
                {
                    s_stopped = true;
                    TEST_LOG_SS(Error, group.name << "AutoTest has errors. TEST EXECUTION IS TERMINATED!" << std::endl);
                    return;
                }
            }
            _progress = 1.0;
        }

    private:
        static bool RunGroup(const Group & group)
        {
#if defined(_MSC_VER)
            __try
            {
                return group.autoTest();
            }
            __except (EXCEPTION_EXECUTE_HANDLER)
            {
                PrintErrorMessage(GetExceptionCode());
                return false;
            }
#else
            return group.autoTest();
#endif
        }

#if defined(_MSC_VER)
        static void PrintErrorMessage(int code)
        {
            String desc;
            switch (code)
            {
            case EXCEPTION_ACCESS_VIOLATION: desc = "Access violation"; break;
            case EXCEPTION_FLT_DIVIDE_BY_ZERO: desc = "Float divide by zero"; break;
            case EXCEPTION_INT_DIVIDE_BY_ZERO: desc = "Integer divide by zero"; break;
            case EXCEPTION_ILLEGAL_INSTRUCTION: desc = "Illegal instruction"; break;
            case EXCEPTION_STACK_OVERFLOW: desc = "Stack overflow"; break;
            default:
                desc = "Unknown error(" + std::to_string(code) + ")";
            }
            TEST_LOG_SS(Error, "There is unhandled exception: " << desc << " !");
        }
#endif
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
            Special,
        } mode;

        bool help;

        Strings include, exclude;

        String text, html;

        size_t testThreads, workThreads, testRepeats, testStatistics;

        bool printAlign, printInternal, checkCpp;

        Options(int argc, char* argv[])
            : mode(Auto)
            , help(false)
            , testThreads(0)
            , testRepeats(1)
            , workThreads(1)
            , testStatistics(0)
            , printAlign(false)
            , printInternal(true)
            , checkCpp(false)
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
                    case 's': mode = Special; break;
                    default:
                        TEST_LOG_SS(Error, "Unknown command line options: '" << arg << "'!" << std::endl);
                        exit(1);
                    }
                }
                else if (arg.find("-tt=") == 0)
                {
                    testThreads = Simd::Min(FromString<size_t>(arg.substr(4, arg.size() - 4)), (size_t)std::thread::hardware_concurrency());
                }
                else if (arg.find("-tr=") == 0)
                {
                    testRepeats = FromString<size_t>(arg.substr(4, arg.size() - 4));
                }
                else if (arg.find("-ts=") == 0)
                {
                    testStatistics = FromString<int>(arg.substr(4, arg.size() - 4));
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
                else if (arg.find("-pi=") == 0)
                {
                    printInternal = FromString<bool>(arg.substr(4, arg.size() - 4));
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
                else if (arg.find("-cc=") == 0)
                {
                    checkCpp = FromString<bool>(arg.substr(4, arg.size() - 4));
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

    int MakeAutoTests(Groups & groups, const Options & options)
    {
        if (options.testThreads > 0)
        {
            Test::Log::s_log.SetLevel(Test::Log::Error);

            size_t testThreads = Simd::Min(options.testThreads, groups.size());
            size_t total = groups.size();
            size_t block = Simd::DivHi(total, testThreads);
            testThreads = Simd::Min(testThreads, Simd::DivHi(total, block));

            TEST_LOG_SS(Info, "Test threads count = " << testThreads);
            Test::TaskPtrs tasks;
            for (size_t i = 0; i < testThreads; ++i)
            {
                size_t beg = i * block;
                size_t end = std::min(total, beg + block);
                tasks.push_back(Test::TaskPtr(new Test::Task(groups.data() + beg, end - beg, true)));
            }

            std::cout << std::endl;
            double progress, previous = -1.0;
            do
            {
                progress = 0;
                for (size_t i = 0; i < tasks.size(); ++i)
                    progress += tasks[i]->Progress();
                progress /= double(tasks.size());
                if (progress > previous)
                {
                    std::cout << "\rTest progress: " << std::fixed << std::setprecision(1) << progress * 100.0 << "%" << std::flush;
                    previous = progress;
                }
                Test::Sleep(40);
            } while (progress < 1.0 && !Test::Task::s_stopped);
            std::cout << std::endl << std::endl;

            if (!Test::Task::s_stopped)
                Test::Log::s_log.SetLevel(Test::Log::Info);
        }
        else
        {
            Test::Task task(groups.data(), groups.size(), false);
            task.Run();
        }

        if (Test::Task::s_stopped)
            return 1;

        TEST_LOG_SS(Info, "ALL TESTS ARE FINISHED SUCCESSFULLY!" << std::endl);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, Test::PerformanceMeasurerStorage::s_storage.ConsoleReport(options.printAlign, false) <<
            (options.printInternal ? SimdPerformanceStatistic() : ""));
        if (!options.text.empty())
            Test::PerformanceMeasurerStorage::s_storage.TextReport(options.text, options.printAlign);
        if (!options.html.empty())
            Test::PerformanceMeasurerStorage::s_storage.HtmlReport(options.html, options.printAlign);
#endif

        if (options.testStatistics)
        {
            std::sort(groups.begin(), groups.end(), [](const Group& a, const Group& b) { return a.time > b.time; });
            for (size_t i = 0; i < groups.size(); ++i)
            {
                if(groups[i].time >= options.testStatistics)
                    TEST_LOG_SS(Info, "Test " << groups[i].name << " elapsed " << ToString(groups[i].time, 1, false) << " s.");
            }
        }

        return 0;
    }

    int MakeSpecialTests(Groups & groups, const Options & options)
    {
        for (Test::Group & group : groups)
        {
            TEST_LOG_SS(Info, group.name << "SpecialTest is started :");
            group.time = GetTime();
            bool result = group.specialTest();
            group.time = GetTime() - group.time;
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
        std::cout << "               with using of different SIMD instructions such as SSE4.1, " << std::endl;
        std::cout << "               AVX2, and other). Also it can be: " << std::endl;
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
        std::cout << "    -pi=1         to print internal statistics (Cmake parameter SIMD_PERF must be ON)." << std::endl << std::endl;
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
        std::cout << "    -tr=2         a number of test execution repeats." << std::endl;
        std::cout << "    -ts=1         to print statistics of time of tests execution." << std::endl;
        std::cout << "    -cc=1         to check c++ API." << std::endl;
        return 0;
    }

#if defined(_MSC_VER)
    String ROOT_PATH = "../..";
#else
    String ROOT_PATH = "..";
#endif
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
    Test::Options options(argc, argv);

    if (options.help)
        return Test::PrintHelp();

    if(options.checkCpp)
        Test::CheckCpp();

    Test::Groups groups;
    for (const Test::Group& group : Test::g_groups)
    {
        if (options.Required(group))
        {
            for(size_t r = 0; r < options.testRepeats; ++r)
                groups.push_back(group);
        }
    }
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
    case Test::Options::Special:
        return Test::MakeSpecialTests(groups, options);
    default:
        return 0;
    }
}

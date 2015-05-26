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
#include "Test/TestLog.h"

namespace Test
{
    struct Options
    {
        enum Mode
        {
            Auto,
            Create, 
            Verify,
        } mode;

        std::string filter;

        std::string output;

        size_t threads;

        Options(int argc, char* argv[])
            : mode(Auto)
            , threads(0)
        {
            for(int i = 1; i < argc; ++i)
            {
                std::string arg = argv[i];
                if(arg.find("-m=") == 0)
                {
                    switch(arg[3])
                    {
                    case 'a': mode = Auto; break;
                    case 'c': mode = Create; break;
                    case 'v': mode = Verify; break;
                    default:
                        TEST_LOG_SS(Error, "Unknown command line options: '" << arg << "'!" << std::endl); 
                        exit(1);
                    }
                }
                else if(arg.find("-t=") == 0)
                {
#ifdef NDEBUG
                    std::stringstream ss(arg.substr(3, arg.size() - 3));
                    ss >> threads; 
#endif
                }
                else if(arg.find("-f=") == 0)
                {
                    filter = arg.substr(3, arg.size() - 3);
                }
                else if(arg.find("-o=") == 0)
                {
                    output = arg.substr(3, arg.size() - 3);
                }
                else
                {
                    TEST_LOG_SS(Error, "Unknown command line options: '" << arg << "'!" << std::endl); 
                    exit(1);
                }
            }
        }

        bool NeedToPerform(const std::string & name) const
        {
            return filter.empty() || name.find(filter) != std::string::npos;
        }
    };

    typedef bool (*AutoTestPtr)(); 
    typedef bool (*DataTestPtr)(bool create); 

    struct Group
    {
        std::string name;
        AutoTestPtr autoTest;
        DataTestPtr dataTest;
        Group(const std::string & n, const AutoTestPtr & a, const DataTestPtr & d)
            : name(n)
            , autoTest(a)
            , dataTest(d)
        {
        }
    };
    typedef std::vector<Group> Groups;
    Groups g_groups;

#define TEST_ADD_GROUP(name) \
    bool name##AutoTest(); \
    bool name##DataTest(bool create); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, name##AutoTest, name##DataTest)); return true; } \
    bool name##AtList = name##AddToList();

    TEST_ADD_GROUP(AbsDifferenceSum);
    TEST_ADD_GROUP(AbsDifferenceSumMasked);
    TEST_ADD_GROUP(AbsDifferenceSums3x3);
    TEST_ADD_GROUP(AbsDifferenceSums3x3Masked);
    TEST_ADD_GROUP(SquaredDifferenceSum);
    TEST_ADD_GROUP(SquaredDifferenceSumMasked);
    TEST_ADD_GROUP(SquaredDifferenceSum32f);

    TEST_ADD_GROUP(AddFeatureDifference);

    TEST_ADD_GROUP(BgraToBgr);
    TEST_ADD_GROUP(BgraToGray);
    TEST_ADD_GROUP(BgrToGray);
    TEST_ADD_GROUP(BgrToHsl);
    TEST_ADD_GROUP(BgrToHsv);
    TEST_ADD_GROUP(GrayToBgr);

    TEST_ADD_GROUP(BgraToBayer);
    TEST_ADD_GROUP(BgrToBayer);

    TEST_ADD_GROUP(BgrToBgra);
    TEST_ADD_GROUP(GrayToBgra);

    TEST_ADD_GROUP(BgraToYuv420p);
    TEST_ADD_GROUP(BgraToYuv422p);
    TEST_ADD_GROUP(BgraToYuv444p);
    TEST_ADD_GROUP(BgrToYuv420p);
    TEST_ADD_GROUP(BgrToYuv422p);
    TEST_ADD_GROUP(BgrToYuv444p);

    TEST_ADD_GROUP(BackgroundGrowRangeSlow);
    TEST_ADD_GROUP(BackgroundGrowRangeFast);
    TEST_ADD_GROUP(BackgroundIncrementCount);
    TEST_ADD_GROUP(BackgroundAdjustRange);
    TEST_ADD_GROUP(BackgroundAdjustRangeMasked);
    TEST_ADD_GROUP(BackgroundShiftRange);
    TEST_ADD_GROUP(BackgroundShiftRangeMasked);
    TEST_ADD_GROUP(BackgroundInitMask);

    TEST_ADD_GROUP(BayerToBgr);

    TEST_ADD_GROUP(BayerToBgra);

    TEST_ADD_GROUP(Bgr48pToBgra32);

    TEST_ADD_GROUP(Binarization);
    TEST_ADD_GROUP(AveragingBinarization);

    TEST_ADD_GROUP(ConditionalCount8u);
    TEST_ADD_GROUP(ConditionalCount16i);
    TEST_ADD_GROUP(ConditionalSum);
    TEST_ADD_GROUP(ConditionalSquareSum);
    TEST_ADD_GROUP(ConditionalSquareGradientSum);

    TEST_ADD_GROUP(Copy);
    TEST_ADD_GROUP(CopyFrame);

    TEST_ADD_GROUP(Crc32c);

    TEST_ADD_GROUP(DeinterleaveUv);

    TEST_ADD_GROUP(AlphaBlending);

    TEST_ADD_GROUP(EdgeBackgroundGrowRangeSlow);
    TEST_ADD_GROUP(EdgeBackgroundGrowRangeFast);
    TEST_ADD_GROUP(EdgeBackgroundIncrementCount);
    TEST_ADD_GROUP(EdgeBackgroundAdjustRange);
    TEST_ADD_GROUP(EdgeBackgroundAdjustRangeMasked);
    TEST_ADD_GROUP(EdgeBackgroundShiftRange);
    TEST_ADD_GROUP(EdgeBackgroundShiftRangeMasked);

    TEST_ADD_GROUP(Fill);
    TEST_ADD_GROUP(FillFrame);
    TEST_ADD_GROUP(FillBgra);
    TEST_ADD_GROUP(FillBgr);

    TEST_ADD_GROUP(Histogram);
    TEST_ADD_GROUP(HistogramMasked);
    TEST_ADD_GROUP(AbsSecondDerivativeHistogram);

    TEST_ADD_GROUP(HogDirectionHistograms);

    TEST_ADD_GROUP(Integral);

    TEST_ADD_GROUP(InterferenceIncrement);
    TEST_ADD_GROUP(InterferenceIncrementMasked);
    TEST_ADD_GROUP(InterferenceDecrement);
    TEST_ADD_GROUP(InterferenceDecrementMasked);

    TEST_ADD_GROUP(MedianFilterRhomb3x3);
    TEST_ADD_GROUP(MedianFilterRhomb5x5);
    TEST_ADD_GROUP(MedianFilterSquare3x3);
    TEST_ADD_GROUP(MedianFilterSquare5x5);
    TEST_ADD_GROUP(GaussianBlur3x3);
    TEST_ADD_GROUP(AbsGradientSaturatedSum);
    TEST_ADD_GROUP(LbpEstimate);

    TEST_ADD_GROUP(OperationBinary8u);
    TEST_ADD_GROUP(OperationBinary16i);
    TEST_ADD_GROUP(VectorProduct);

    TEST_ADD_GROUP(ReduceGray2x2);
    TEST_ADD_GROUP(ReduceGray3x3);
    TEST_ADD_GROUP(ReduceGray4x4);
    TEST_ADD_GROUP(ReduceGray5x5);

    TEST_ADD_GROUP(Reorder16bit);
    TEST_ADD_GROUP(Reorder32bit);
    TEST_ADD_GROUP(Reorder64bit);

    TEST_ADD_GROUP(ResizeBilinear);

    TEST_ADD_GROUP(SegmentationShrinkRegion);
    TEST_ADD_GROUP(SegmentationFillSingleHoles);
    TEST_ADD_GROUP(SegmentationChangeIndex);
    TEST_ADD_GROUP(SegmentationPropagate2x2);

    TEST_ADD_GROUP(ShiftBilinear);

    TEST_ADD_GROUP(SobelDx);
    TEST_ADD_GROUP(SobelDxAbs);
    TEST_ADD_GROUP(SobelDy);
    TEST_ADD_GROUP(SobelDyAbs);
    TEST_ADD_GROUP(ContourMetrics);
    TEST_ADD_GROUP(ContourMetricsMasked);
    TEST_ADD_GROUP(ContourAnchors);

    TEST_ADD_GROUP(GetStatistic);
    TEST_ADD_GROUP(GetMoments);
    TEST_ADD_GROUP(GetRowSums);
    TEST_ADD_GROUP(GetColSums);
    TEST_ADD_GROUP(GetAbsDyRowSums);
    TEST_ADD_GROUP(GetAbsDxColSums);
    TEST_ADD_GROUP(ValueSum);
    TEST_ADD_GROUP(SquareSum);
    TEST_ADD_GROUP(SobelDxAbsSum);
    TEST_ADD_GROUP(SobelDyAbsSum);
    TEST_ADD_GROUP(CorrelationSum);

    TEST_ADD_GROUP(StretchGray2x2);

    TEST_ADD_GROUP(SvmSumLinear);

    TEST_ADD_GROUP(TextureBoostedSaturatedGradient);
    TEST_ADD_GROUP(TextureBoostedUv);
    TEST_ADD_GROUP(TextureGetDifferenceSum);
    TEST_ADD_GROUP(TexturePerformCompensation);

    TEST_ADD_GROUP(Yuv444pToBgr);
    TEST_ADD_GROUP(Yuv422pToBgr);
    TEST_ADD_GROUP(Yuv420pToBgr);
    TEST_ADD_GROUP(Yuv444pToHsl);
    TEST_ADD_GROUP(Yuv444pToHsv);
    TEST_ADD_GROUP(Yuv444pToHue);
    TEST_ADD_GROUP(Yuv420pToHue);

    TEST_ADD_GROUP(Yuv444pToBgra);
    TEST_ADD_GROUP(Yuv422pToBgra);
    TEST_ADD_GROUP(Yuv420pToBgra);

    class Task
    {
        const Groups & _groups;
        std::thread _thread;
        volatile double _progress;
    public:
        static volatile bool s_stopped;

        Task(const Groups & groups, bool start)
            : _groups(groups)
            , _progress(0)
        {
            if(start)
                _thread = std::thread(&Task::Run, this);
        }

        ~Task()
        {
            if(_thread.joinable())
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
            for(size_t i = 0; i < _groups.size() && !s_stopped; ++i)
            {
                _progress = double(i)/double(_groups.size());
                const Group & group = _groups[i];
                TEST_LOG_SS(Info, group.name << "AutoTest is started :"); 
                bool result = group.autoTest(); 
                TEST_LOG_SS(Info, group.name << "AutoTest - is finished " << (result ? "successfully." : "with errors!") << std::endl);
                if(!result) 
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
}

int main(int argc, char* argv[])
{
    Test::Options options(argc, argv);
    if(!options.output.empty())
        Test::Log::s_log.SetLogFile(options.output);

    Test::Groups groups;
    for(const Test::Group & group : Test::g_groups)
        if(options.NeedToPerform(group.name))
            groups.push_back(group);
    if(groups.empty())
    {
        TEST_LOG_SS(Error, "There are not any suitable tests for current filter '" << options.filter <<"'!" << std::endl); 
        return 1;
    }

    if(options.mode == Test::Options::Auto)
    {
        if(options.threads > 0)
        {
            TEST_LOG_SS(Info, "Test threads count = " << options.threads);

            Test::Log::s_log.SetLevel(Test::Log::Error);

            Test::TaskPtrs tasks;
            for(size_t i = 0; i < options.threads; ++i)
                tasks.push_back(Test::TaskPtr(new Test::Task(groups, true)));

            std::cout << std::endl;
            double progress;
            do 
            {
                progress = 0;
                for(size_t i = 0; i < tasks.size(); ++i)
                    progress += tasks[i]->Progress();
                progress /= double(tasks.size());
                std::cout << "\rTest progress = " << int(progress*100.0) << "%.";
                Test::Sleep(40);
            } while (progress < 1.0 && !Test::Task::s_stopped);
            std::cout << std::endl << std::endl;

            Test::Log::s_log.SetLevel(Test::Log::Info); 
        }
        else
        {
            Test::Task task(groups, false);
            task.Run();
        }

        if(Test::Task::s_stopped)
            return 1;
        
        TEST_LOG_SS(Info, "ALL TESTS ARE FINISHED SUCCESSFULLY!" << std::endl);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
        TEST_LOG_SS(Info, Test::PerformanceMeasurerStorage::s_storage.Report(true, true, false));
#endif
    }
    else
    {
        for(const Test::Group & group : groups)
        {
            bool create = options.mode == Test::Options::Create;
            TEST_LOG_SS(Info, group.name << "DataTest - data " << (create ? "creation" : "verification") << " is started :"); 
            bool result = group.dataTest(create); 
            TEST_LOG_SS(Info, group.name << "DataTest - data " << (create ? "creation" : "verification") << " is finished " << (result ? "successfully." : "with errors!") << std::endl);
            if(!result) 
            { 
                TEST_LOG_SS(Error, "ERROR! TEST EXECUTION IS TERMINATED !" << std::endl); 
                return 1;
            }            
        }
        TEST_LOG_SS(Info, "ALL TESTS ARE FINISHED SUCCESSFULLY!" << std::endl); 
    }
}

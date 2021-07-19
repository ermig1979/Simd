/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2021 Yermalayeu Ihar.
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
#ifndef __TestPerformance_h__
#define __TestPerformance_h__

#include "Test/TestConfig.h"

namespace Test
{
    double GetTime();

    //-------------------------------------------------------------------------

    class PerformanceMeasurer
    {
        String	_description;
        double _start;
        int _count;
        double _total;
        double _min;
        double _max;
        bool _entered;

        long long _size;

    public:
        PerformanceMeasurer(const String & description = "Unnamed");
        PerformanceMeasurer(const PerformanceMeasurer & pm);

        void Enter();
        void Leave(size_t size = 1);

        double Average() const;
        String Statistic() const;

        String Description() const { return _description; }

        void Combine(const PerformanceMeasurer & other);
    };

    //-------------------------------------------------------------------------

    class ScopedPerformanceMeasurer
    {
        PerformanceMeasurer * _pm;
        size_t _size;
    public:

        ScopedPerformanceMeasurer(PerformanceMeasurer & pm) : _pm(&pm), _size(1)
        {
            if (_pm)
                _pm->Enter();
        }

        ScopedPerformanceMeasurer(PerformanceMeasurer * pm) : _pm(pm), _size(1)
        {
            if (_pm)
                _pm->Enter();
        }

        ~ScopedPerformanceMeasurer()
        {
            if (_pm)
                _pm->Leave(_size);
        }

        void SetSize(size_t size) { _size = size; }
    };

    //-------------------------------------------------------------------------

    class PerformanceMeasurerStorage
    {
        typedef PerformanceMeasurer Pm;
        typedef std::shared_ptr<Pm> PmPtr;
        typedef std::map<String, PmPtr> FunctionMap;
        struct Thread
        {
            FunctionMap map;
            bool align;
        };
        typedef std::map<std::thread::id, Thread> ThreadMap;

        ThreadMap _map;
        mutable std::recursive_mutex _mutex;

        Thread & ThisThread();

        typedef std::shared_ptr<class Table> TablePtr;
        void Combine(FunctionMap& map) const;
        TablePtr GenerateTable(bool align) const;

    public:
        static PerformanceMeasurerStorage s_storage;

        PerformanceMeasurerStorage();
        ~PerformanceMeasurerStorage();

        PerformanceMeasurer* Get(String name);

        size_t Align(size_t size);

        String ConsoleReport(bool align = false, bool raw = false) const;

        bool TextReport(const String& path, bool align = false) const;

        bool HtmlReport(const String & path, bool align = false) const;

        void Clear();
    };
}

#define TEST_PERFORMANCE_TEST_(decription) Test::ScopedPerformanceMeasurer ___spm(*(Test::PerformanceMeasurerStorage::s_storage.Get(decription)));
#define TEST_FUNCTION_PERFORMANCE_TEST_ TEST_PERFORMANCE_TEST_(__FUNCTION__)
#define TEST_PERFORMANCE_TEST_SET_SIZE_(size) ___spm.SetSize(size);

#ifdef TEST_PERFORMANCE_TEST_ENABLE
#define TEST_PERFORMANCE_TEST(decription) TEST_PERFORMANCE_TEST_(decription)
#define TEST_FUNCTION_PERFORMANCE_TEST TEST_FUNCTION_PERFORMANCE_TEST_
#define TEST_PERFORMANCE_TEST_SET_SIZE(size) TEST_PERFORMANCE_TEST_SET_SIZE_(size)
#else//TEST_PERFORMANCE_TEST_ENABLE
#define TEST_PERFORMANCE_TEST(decription)
#define TEST_FUNCTION_PERFORMANCE_TEST
#define TEST_PERFORMANCE_TEST_SET_SIZE(size)
#endif//TEST_PERFORMANCE_TEST_ENABLE

#ifdef NDEBUG
#define TEST_EXECUTE_AT_LEAST_MIN_TIME(test) \
{ \
	double startTime = Test::GetTime(); \
	do \
	{ \
        if(Test::LITTER_CPU_CACHE) \
            Simd::LitterCpuCache(Test::LITTER_CPU_CACHE); \
		test; \
	} \
	while(Test::GetTime() - startTime < Test::MINIMAL_TEST_EXECUTION_TIME); \
}
#else
#define TEST_EXECUTE_AT_LEAST_MIN_TIME(test) \
    test; 
#endif

#define TEST_ALIGN(size) \
    Test::PerformanceMeasurerStorage::s_storage.Align(size)

#endif//__TestPerformance_h__

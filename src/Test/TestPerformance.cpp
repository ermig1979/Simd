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
#include <limits>
#include <iomanip>

#include "Test/TestPerformance.h"

#if defined(_MSC_VER)
#define NOMINMAX
#include <windows.h>
#elif defined(__GNUC__)
#include <sys/time.h>
#else
#error Platform is not supported!
#endif

namespace Test
{
#if defined(_MSC_VER)
	double GetFrequency()
	{
		LARGE_INTEGER frequency;
		QueryPerformanceFrequency(&frequency);
		return double(frequency.QuadPart);
	}

	double g_frequency = GetFrequency();

	double GetTime()
	{
		LARGE_INTEGER counter;
		QueryPerformanceCounter(&counter);
		return double(counter.QuadPart)/g_frequency;
	}
#elif defined(__GNUC__)
	double GetTime()
	{
		timeval t1;
		gettimeofday(&t1, NULL);
		return t1.tv_sec * 1000.0 + t1.tv_usec / 1000.0;
	}
#else
#error Platform is not supported!
#endif

	//-------------------------------------------------------------------------

	PerformanceMeasurer::PerformanceMeasurer(const std::string & decription)
        :_decription(decription),
        _count(0),
        _total(0),
        _entered(false),
        _min(std::numeric_limits<double>::max()),
        _max(std::numeric_limits<double>::min()),
        _size(0)
    {
    }

    void PerformanceMeasurer::Enter()
    {
        if (!_entered)
        {
            _entered = true;
            _start = GetTime();
        }
    }

    void PerformanceMeasurer::Leave(size_t size)
    {
        if (_entered)
        {
            _entered = false;
            double difference = double(GetTime() - _start);
            _total += difference;
            _min = std::min(_min, difference);
            _max = std::max(_max, difference);
           ++_count;
           _size += std::max<size_t>(1, size);
        }
    }

    double PerformanceMeasurer::Average() const
    {
        return _count ? (_total / _count) : 0;
    }

	std::string PerformanceMeasurer::Statistic() const
    {
		std::stringstream ss;
        ss << _decription << ": ";
        ss << std::setprecision(0) << std::fixed << _total*1000 << " ms";
        ss << " / " << _count << " = ";
        ss << std::setprecision(2) << std::fixed << Average()*1000.0 << " ms";
        ss << std::setprecision(2) << " {min=" << _min*1000.0 << "; max=" << _max*1000.0 << "}";
        if(_size > (long long)_count)
        {
            double size = double(_size);
            ss << std::setprecision(3) << " [<s>=" << size/_count*0.001 << " kb; <t>=" << _total/size*1000000000 << " ns]";
        }
        return ss.str();
    }

    void PerformanceMeasurer::Combine(const PerformanceMeasurer & other)
    {
        _count += other._count;
        _total += other._total;
        _min = std::min(_min, other._min);
        _max = std::max(_max, other._max);
        _size += other._size;
    }

    //-------------------------------------------------------------------------
    PerformanceMeasurerStorage::~PerformanceMeasurerStorage()
    {
        for(Map::iterator it = _map.begin(); it != _map.end(); ++it)
            delete it->second;
    }

    PerformanceMeasurer* PerformanceMeasurerStorage::Get(const std::string & name)
    {
        PerformanceMeasurer *pm = NULL;
        Map::iterator it = _map.find(name);
        if(it == _map.end())
        {
            pm = new PerformanceMeasurer(name);
            _map[name] = pm;
       }
        else
            pm = it->second;
        return pm;
    }

    std::string PerformanceMeasurerStorage::Statistic() const
    {
        std::vector<std::string> statistics;
        for(Map::const_iterator it = _map.begin(); it != _map.end(); ++it)
            statistics.push_back(it->second->Statistic());

        std::sort(statistics.begin(), statistics.end());

		std::stringstream statistic;
        for(size_t i = 0; i < statistics.size(); ++i)
            statistic << statistics[i] << std::endl;
        return statistic.str();
    }

    PerformanceMeasurerStorage PerformanceMeasurerStorage::s_storage = PerformanceMeasurerStorage();
}

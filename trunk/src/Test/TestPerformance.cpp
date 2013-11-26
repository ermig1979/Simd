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
#include "Test/TestUtils.h"

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
		return t1.tv_sec + t1.tv_usec / 1000000.0;
	}
#else
#error Platform is not supported!
#endif

	//-------------------------------------------------------------------------

	PerformanceMeasurer::PerformanceMeasurer(const std::string & decription)
        : _decription(decription)
        , _count(0)
        , _total(0)
        , _entered(false)
        , _min(std::numeric_limits<double>::max())
        , _max(std::numeric_limits<double>::min())
        , _size(0)
    {
    }

    PerformanceMeasurer::PerformanceMeasurer(const PerformanceMeasurer & pm)
        : _decription(pm._decription)
        , _count(pm._count)
        , _total(pm._total)
        , _entered(pm._entered)
        , _min(pm._min)
        , _max(pm._max)
        , _size(pm._size)
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

    PerformanceMeasurerStorage PerformanceMeasurerStorage::s_storage = PerformanceMeasurerStorage();

    PerformanceMeasurerStorage::~PerformanceMeasurerStorage()
    {
    }

    PerformanceMeasurer* PerformanceMeasurerStorage::Get(std::string name)
    {
        name = name + (_align ? "{a}" : "{u}");

        PerformanceMeasurer *pm = NULL;
        Map::iterator it = _map.find(name);
        if(it == _map.end())
        {
            pm = new PerformanceMeasurer(name);
            _map[name].reset(pm);
       }
        else
            pm = it->second.get();
        return pm;
    }

    size_t PerformanceMeasurerStorage::Align(size_t size)
    {
        s_storage._align = size%Simd::DEFAULT_MEMORY_ALIGN == 0;
        return s_storage._align ? Simd::DEFAULT_MEMORY_ALIGN : 1;
    }    

    std::string PerformanceMeasurerStorage::Report() const
    {
        Map filtered;
        size_t sizeMax = 0;
        for(Map::const_iterator it = _map.begin(); it != _map.end(); ++it)
        {
            const std::string & desc = it->second->Description();
            if(desc[5] == ':' && desc[11] == ':' && desc.find("Crc32") == std::string::npos)
            {
                filtered[desc].reset(new Pm(*it->second));
                sizeMax = std::max(desc.size(), sizeMax);
            }
        }

        struct Statistic
        {
            PmPtr base;
            PmPtr sse2A;
            PmPtr sse2U;
            PmPtr avx2A;
            PmPtr avx2U;
        };
        typedef std::map<std::string, Statistic> StatisticMap;

        StatisticMap statistic;
        double timeMax = 0;
        for(Map::const_iterator it = filtered.begin(); it != filtered.end(); ++it)
        {
            const std::string & desc = it->second->Description();
            std::string name = desc.substr(12, desc.size() - 15);
            Statistic & s = statistic[name];
            if(desc[6] == 'B')
                s.base.reset(new Pm(*it->second));
            if(desc[6] == 'S' && desc[desc.size() - 2] == 'a')
                s.sse2A.reset(new Pm(*it->second));
            if(desc[6] == 'S' && desc[desc.size() - 2] == 'u')
                s.sse2U.reset(new Pm(*it->second));
            if(desc[6] == 'A' && desc[desc.size() - 2] == 'a')
                s.avx2A.reset(new Pm(*it->second));
            if(desc[6] == 'A' && desc[desc.size() - 2] == 'u')
                s.avx2U.reset(new Pm(*it->second));
            timeMax = std::max(timeMax, it->second->Average());
        }

        const size_t ic = std::max<size_t>(1, (size_t)::log10(timeMax) + 3);
        const size_t fc = 3;

        std::vector<std::string> statistics;
        for(StatisticMap::const_iterator it = statistic.begin(); it != statistic.end(); ++it)
        {
            const Statistic & s = it->second;
            std::stringstream ss;
            ss << ExpandToRight(it->first, sizeMax - 15) << " | ";
            ss << ToString(s.base->Average()*1000.0, ic, fc) << " ";
            ss << ToString(s.sse2A->Average()*1000.0, ic, fc) << " ";
            ss << ToString(s.avx2A->Average()*1000.0, ic, fc) << " | ";
            ss << ToString(s.base->Average()/s.sse2A->Average(), ic, fc) << " ";
            ss << ToString(s.base->Average()/s.avx2A->Average(), ic, fc) << " ";
            ss << ToString(s.sse2A->Average()/s.avx2A->Average(), ic, fc) << " | ";
            ss << ToString(s.sse2U->Average()/s.sse2A->Average(), ic, fc) << " ";
            ss << ToString(s.avx2U->Average()/s.avx2A->Average(), ic, fc) << " ";
            statistics.push_back(ss.str());
        }

        std::sort(statistics.begin(), statistics.end());

        std::stringstream report;
        report << std::endl;
        report << ExpandToRight("Function", sizeMax - 15) << " | ";
        report << ExpandToLeft("Base", ic + fc + 1) << " ";
        report << ExpandToLeft("Sse2", ic + fc + 1) << " ";
        report << ExpandToLeft("Avx2", ic + fc + 1) << " | ";
        report << ExpandToLeft("B/S2", ic + fc + 1) << " ";
        report << ExpandToLeft("B/A2", ic + fc + 1) << " ";
        report << ExpandToLeft("S2/A2", ic + fc + 1) << " | ";
        report << ExpandToLeft("S2:U/A", ic + fc + 1) << " ";
        report << ExpandToLeft("A2:U/A", ic + fc + 1) << " ";
        report << std::endl;
        for(ptrdiff_t i = sizeMax - 15 + 3*2 + 8*(ic + fc + 1 + 1); i >= 0; --i)
            report << "-";
        report << std::endl;

        for(size_t i = 0; i < statistics.size(); ++i)
            report << statistics[i] << std::endl;
        return report.str();
    }
}

/*
* Tests for Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2015 Yermalayeu Ihar.
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

	PerformanceMeasurer::PerformanceMeasurer(const std::string & description)
        : _description(description)
        , _count(0)
        , _total(0)
        , _entered(false)
        , _min(std::numeric_limits<double>::max())
        , _max(std::numeric_limits<double>::min())
        , _size(0)
    {
    }

    PerformanceMeasurer::PerformanceMeasurer(const PerformanceMeasurer & pm)
        : _description(pm._description)
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
        ss << _description << ": ";
        ss << std::setprecision(0) << std::fixed << _total*1000 << " ms";
        ss << " / " << _count << " = ";
        ss << std::setprecision(3) << std::fixed << Average()*1000.0 << " ms";
        ss << std::setprecision(3) << " {min=" << _min*1000.0 << "; max=" << _max*1000.0 << "}";
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
        s_storage._align = size%SIMD_ALIGN == 0;
        return s_storage._align ? SIMD_ALIGN : sizeof(void*);
    }

    static std::string FunctionShortName(const std::string & description)
    {
        bool isApi = description.find("Simd::") == std::string::npos;
        if(isApi)
        {
            return description.substr(4, description.size() - 7);
        }
        else
        {
            size_t pos = description.rfind("::");
            return description.substr(pos + 2, description.size() - pos - 5);
        }
    }

    static bool Aligned(const std::string & description)
    {
        return description[description.size() - 2] == 'a';
    }

    static double Relation(const PerformanceMeasurer & a, const PerformanceMeasurer & b)
    {
        return b.Average() > 0 ? a.Average()/b.Average() : 0;
    }

    std::string PerformanceMeasurerStorage::Report(bool sse42_, bool align, bool raw) const
    {
        struct Statistic
        {
            std::pair<Pm, Pm> simd;
            std::pair<Pm, Pm> base;
            Pm sse42;
            std::pair<Pm, Pm> sse2;
            std::pair<Pm, Pm> ssse3;
            std::pair<Pm, Pm> sse41;
            std::pair<Pm, Pm> avx2;
            std::pair<Pm, Pm> vmx;
            std::pair<Pm, Pm> vsx;
        };
        typedef std::map<std::string, Statistic> StatisticMap;

        StatisticMap statistic;
        double timeMax = 0;
        size_t sizeMax = 8;
        bool sse2 = false, ssse3 = false, sse41 = false, sse42 = false, avx2 = false, vmx = false, vsx = false;
        for(Map::const_iterator it = _map.begin(); it != _map.end(); ++it)
        {
            const std::string & desc = it->second->Description();
            std::string name = FunctionShortName(desc);
            Statistic & s = statistic[name];
            if(desc.find("Simd::") == std::string::npos && desc.find("Simd") == 0)
            {
                if(Aligned(desc))
                    s.simd.first = *it->second;
                else
                    s.simd.second = *it->second;
            }
            if(desc.find("Simd::Base::") != std::string::npos)
            {
                if(Aligned(desc))
                    s.base.first = *it->second;
                else
                    s.base.second = *it->second;
            }
            if(desc.find("Simd::Sse2::") != std::string::npos || desc.find("Simd::Sse::") != std::string::npos)
            {
                if(Aligned(desc))
                    s.sse2.first = *it->second;
                else
                    s.sse2.second = *it->second;
                sse2 = true;
            }
            if(desc.find("Simd::Ssse3::") != std::string::npos)
            {
                if(Aligned(desc))
                    s.ssse3.first = *it->second;
                else
                    s.ssse3.second = *it->second;
                ssse3 = true;
            }
            if(desc.find("Simd::Sse41::") != std::string::npos)
            {
                if(Aligned(desc))
                    s.sse41.first = *it->second;
                else
                    s.sse41.second = *it->second;
                sse41 = true;
            }
            if(desc.find("Simd::Sse42::") != std::string::npos)
            {
                s.sse42 = *it->second;
                sse42 = true && sse42_;
            }
            if(desc.find("Simd::Avx2::") != std::string::npos || desc.find("Simd::Avx::") != std::string::npos)
            {
                if(Aligned(desc))
                    s.avx2.first = *it->second;
                else
                    s.avx2.second = *it->second;
                avx2 = true;
            }
            if(desc.find("Simd::Vmx::") != std::string::npos)
            {
                if(Aligned(desc))
                    s.vmx.first = *it->second;
                else
                    s.vmx.second = *it->second;
                vmx = true;
            }
            if(desc.find("Simd::Vsx::") != std::string::npos)
            {
                if(Aligned(desc))
                    s.vsx.first = *it->second;
                else
                    s.vsx.second = *it->second;
                vsx = true;
            }

            timeMax = std::max(timeMax, it->second->Average());
            sizeMax = std::max(name.size(), sizeMax);
        }

        const size_t ic = 1 + (size_t)::log10(std::max(timeMax*1000, 1.0));
        const size_t ir = 3;
        const size_t fc = 3;

        std::vector<std::string> statistics;
        for(StatisticMap::const_iterator it = statistic.begin(); it != statistic.end(); ++it)
        {
            const Statistic & s = it->second;
            std::stringstream ss;
            ss << ExpandToRight(it->first, sizeMax) << " | ";

            ss << ToString(s.simd.first.Average()*1000.0, ic, fc) << " ";
            ss << ToString(s.base.first.Average()*1000.0, ic, fc) << " ";
            if(sse2) ss << ToString(s.sse2.first.Average()*1000.0, ic, fc) << " ";
            if(ssse3) ss << ToString(s.ssse3.first.Average()*1000.0, ic, fc) << " ";
            if(sse41) ss << ToString(s.sse41.first.Average()*1000.0, ic, fc) << " ";
            if(sse42) ss << ToString(s.sse42.Average()*1000.0, ic, fc) << " ";
            if(avx2) ss << ToString(s.avx2.first.Average()*1000.0, ic, fc) << " ";
            if(vmx) ss << ToString(s.vmx.first.Average()*1000.0, ic, fc) << " ";
            if(vsx) ss << ToString(s.vsx.first.Average()*1000.0, ic, fc) << " ";
            ss << "| ";

            if(sse2 || ssse3 || sse41 || sse42 || avx2 || vmx || vsx)
            {
                if(sse2) ss << ToString(Relation(s.base.first, s.sse2.first), ir, fc) << " ";
                if(ssse3) ss << ToString(Relation(s.base.first, s.ssse3.first), ir, fc) << " ";
                if(sse41) ss << ToString(Relation(s.base.first, s.sse41.first), ir, fc) << " ";
                if(sse42) ss << ToString(Relation(s.base.first, s.sse42), ir, fc) << " ";
                if(avx2) ss << ToString(Relation(s.base.first, s.avx2.first), ir, fc) << " ";
                if(vmx) ss << ToString(Relation(s.base.first, s.vmx.first), ir, fc) << " ";
                if(vsx) ss << ToString(Relation(s.base.first, s.vsx.first), ir, fc) << " ";
                ss << "| ";
            }

            if(sse2 && (ssse3 || sse41))
            {
                if(ssse3) ss << ToString(Relation(s.sse2.first, s.ssse3.first), ir, fc) << " ";
                if(sse41) ss << ToString(Relation(s.sse2.first, s.sse41.first), ir, fc) << " ";
                ss << "| ";
            }

            if((sse2 || ssse3 || sse41) && avx2)
            {
                if(sse2) ss << ToString(Relation(s.sse2.first, s.avx2.first), ir, fc) << " ";
                if(ssse3) ss << ToString(Relation(s.ssse3.first, s.avx2.first), ir, fc) << " ";
                if(sse41) ss << ToString(Relation(s.sse41.first, s.avx2.first), ir, fc) << " ";
                ss << "| ";
            }

            if(align)
            {
                ss << ToString(Relation(s.base.second, s.base.first), ir, fc) << " ";
                if(sse2) ss << ToString(Relation(s.sse2.second, s.sse2.first), ir, fc) << " ";
                if(ssse3) ss << ToString(Relation(s.ssse3.second, s.ssse3.first), ir, fc) << " ";
                if(sse41) ss << ToString(Relation(s.sse41.second, s.sse41.first), ir, fc) << " ";
                if(avx2) ss << ToString(Relation(s.avx2.second, s.avx2.first), ir, fc) << " ";
                if(vmx) ss << ToString(Relation(s.vmx.second, s.vmx.first), ir, fc) << " ";
                if(vsx) ss << ToString(Relation(s.vsx.second, s.vsx.first), ir, fc) << " ";
                ss << "| ";
            }
            statistics.push_back(ss.str());
        }

        std::sort(statistics.begin(), statistics.end());

        std::stringstream header;
        header << ExpandToRight("Function", sizeMax) << " | ";

        header << ExpandToLeft("Simd", ic + fc + 1) << " ";
        header << ExpandToLeft("Base", ic + fc + 1) << " ";
        if(sse2) header << ExpandToLeft("Sse2", ic + fc + 1) << " ";
        if(ssse3) header << ExpandToLeft("Ssse3", ic + fc + 1) << " ";
        if(sse41) header << ExpandToLeft("Sse41", ic + fc + 1) << " ";
        if(sse42) header << ExpandToLeft("Sse42", ic + fc + 1) << " ";
        if(avx2) header << ExpandToLeft("Avx2", ic + fc + 1) << " ";
        if(vmx) header << ExpandToLeft("Vmx", ic + fc + 1) << " ";
        if(vsx) header << ExpandToLeft("Vsx", ic + fc + 1) << " ";
        header << "| ";

        if(sse2 || ssse3 || sse41 || sse42 || avx2 || vmx || vsx)
        {
            if(sse2) header << ExpandToLeft("B/S2", ir + fc + 1) << " ";
            if(ssse3) header << ExpandToLeft("B/S3", ir + fc + 1) << " ";
            if(sse41) header << ExpandToLeft("B/S41", ir + fc + 1) << " ";
            if(sse42) header << ExpandToLeft("B/S42", ir + fc + 1) << " ";
            if(avx2) header << ExpandToLeft("B/A2", ir + fc + 1) << " ";
            if(vmx) header << ExpandToLeft("B/Vm", ir + fc + 1) << " ";
            if(vsx) header << ExpandToLeft("B/Vs", ir + fc + 1) << " ";
            header << "| ";
        }

        if(sse2 && (ssse3 || sse41))
        {
            if(ssse3) header << ExpandToLeft("S2/S3", ir + fc + 1) << " ";
            if(sse41) header << ExpandToLeft("S2/S41", ir + fc + 1) << " ";
            header << "| ";
        }

        if((sse2 || ssse3 || sse41) && avx2)
        {
            if(sse2) header << ExpandToLeft("S2/A2", ir + fc + 1) << " ";
            if(ssse3) header << ExpandToLeft("S3/A2", ir + fc + 1) << " ";
            if(sse41) header << ExpandToLeft("S41/A2", ir + fc + 1) << " ";
            header << "| ";
        }

        if(align)
        {
            header << ExpandToLeft("B:U/A", ir + fc + 1) << " ";
            if(sse2) header << ExpandToLeft("S2:U/A", ir + fc + 1) << " ";
            if(ssse3) header << ExpandToLeft("S3:U/A", ir + fc + 1) << " ";
            if(sse41) header << ExpandToLeft("S41:U/A", ir + fc + 1) << " ";
            if(avx2) header << ExpandToLeft("A2:U/A", ir + fc + 1) << " ";
            if(vmx) header << ExpandToLeft("Vm:U/A", ir + fc + 1) << " ";
            if(vsx) header << ExpandToLeft("Vs:U/A", ir + fc + 1) << " ";
            header << "| ";
        }

        std::stringstream separator;
        for(size_t i = 0; i < header.str().size(); ++i)
            separator << "-";

        std::stringstream report;

        if(raw)
        {
            report << std::endl << "Raw performance report:" << std::endl << std::endl;
            for(Map::const_iterator it = _map.begin(); it != _map.end(); ++it)
                report << it->second->Statistic() << std::endl;
        }

        report << std::endl << "Performance report:" << std::endl << std::endl;
        report << separator.str() << std::endl;
        report << header.str() << std::endl;
        report << separator.str() << std::endl;
        for(size_t i = 0; i < statistics.size(); ++i)
            report << statistics[i] << std::endl;
        report << separator.str() << std::endl;

        return report.str();
    }
}

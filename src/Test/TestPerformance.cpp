/*
* Tests for Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2022 Yermalayeu Ihar.
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
#include "Test/TestFile.h"
#include "Test/TestTable.h"
#include "Test/TestString.h"
#include "Test/TestHtml.h"

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
        return double(counter.QuadPart) / g_frequency;
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

    PerformanceMeasurer::PerformanceMeasurer(const String & description)
        : _description(description)
        , _count(0)
        , _total(0)
        , _min(std::numeric_limits<double>::max())
        , _max(std::numeric_limits<double>::min())
        , _entered(false)
        , _size(0)
    {
    }

    PerformanceMeasurer::PerformanceMeasurer(const PerformanceMeasurer & pm)
        : _description(pm._description)
        , _count(pm._count)
        , _total(pm._total)
        , _min(pm._min)
        , _max(pm._max)
        , _entered(pm._entered)
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

    String PerformanceMeasurer::Statistic() const
    {
        std::stringstream ss;
        ss << _description << ": ";
        ss << std::setprecision(0) << std::fixed << _total * 1000 << " ms";
        ss << " / " << _count << " = ";
        ss << std::setprecision(3) << std::fixed << Average()*1000.0 << " ms";
        ss << std::setprecision(3) << " {min=" << _min*1000.0 << "; max=" << _max*1000.0 << "}";
        if (_size > (long long)_count)
        {
            double size = double(_size);
            ss << std::setprecision(3) << " [<s>=" << size / _count*0.001 << " kb; <t>=" << _total / size * 1000000000 << " ns]";
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

    PerformanceMeasurerStorage PerformanceMeasurerStorage::s_storage;

    PerformanceMeasurerStorage::PerformanceMeasurerStorage()
    {
    }

    PerformanceMeasurerStorage::~PerformanceMeasurerStorage()
    {
    }

    PerformanceMeasurerStorage::Thread & PerformanceMeasurerStorage::ThisThread()
    {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        return _map[std::this_thread::get_id()];
    }

    PerformanceMeasurer* PerformanceMeasurerStorage::Get(String name)
    {
        Thread & thread = ThisThread();
        name = name + (thread.align ? "{a}" : "{u}");
        PerformanceMeasurer * pm = NULL;
        FunctionMap::iterator it = thread.map.find(name);
        if (it == thread.map.end())
        {
            pm = new PerformanceMeasurer(name);
            thread.map[name].reset(pm);
        }
        else
            pm = it->second.get();
        return pm;
    }

    size_t PerformanceMeasurerStorage::Align(size_t size)
    {
        Thread & thread = ThisThread();
        thread.align = size%SIMD_ALIGN == 0;
        return thread.align ? SIMD_ALIGN : sizeof(void*);
    }

    static String FunctionShortName(const String & description)
    {
        bool isApi = description.find("Simd::") == std::string::npos;
        if (isApi)
        {
            return description.substr(4, description.size() - 7);
        }
        else
        {
            size_t pos = description.rfind("::");
            return description.substr(pos + 2, description.size() - pos - 5);
        }
    }

    template <class Measurer> double Relation(const Measurer & a, const Measurer & b)
    {
        return b.Average() > 0 ? a.Average() / b.Average() : 0;
    }

    class CommonPerformance
    {
        int _count;
        double _value;

    public:
        CommonPerformance()
            : _count(0)
            , _value(0)
        {
        }

        void Add(const PerformanceMeasurer & pm)
        {
            if (pm.Average() > 0)
            {
                _count++;
                _value += ::log(pm.Average());
            }
        }

        double Average() const
        {
            return _count > 0 ? ::exp(_value / _count) : 0;
        }
    };

    struct Name
    {
        const char * full;
        const char * brief;
    };

    template <class T> struct Statistic
    {
        T simd;
        T base;
        T sse41;
        T avx;
        T avx2;
        T avx512bw;
        T avx512vnni;
        T avx512bf16;
        T amx;
        T vmx;
        T vsx;
        T neon;

        size_t Size() const { return sizeof(Statistic) / sizeof(T); };
        T & operator [] (size_t i) { return (&simd)[i]; }
        const T & operator [] (size_t i) const { return (&simd)[i]; }
    };
    typedef std::pair<PerformanceMeasurer, PerformanceMeasurer> Function;
    typedef Statistic<Function> FunctionStatistic;
    typedef std::map<std::string, FunctionStatistic> FunctionStatisticMap;
    typedef std::pair<CommonPerformance, CommonPerformance> Common;
    typedef Statistic<Common> CommonStatistic;
    typedef Statistic<bool> StatisticEnable;
    typedef Statistic<Name> StatisticNames;

    template <class T> const T & Previous(const T & f)
    {
        return (&f)[-1].first.Average() > 0 ? (&f)[-1] : Previous((&f)[-1]);
    }

    static inline void AddToFunction(const PerformanceMeasurer & src, Function & dst, bool & enable)
    {
        const String & desc = src.Description();
        bool align = desc[desc.size() - 2] == 'a';
        (align ? dst.first : dst.second) = src;
        enable = true;
    }

    static inline void AddToFunction(const PerformanceMeasurer & src, FunctionStatistic & dst, StatisticEnable & enable)
    {
        const String & desc = src.Description();
        if (desc.find("Simd::") == std::string::npos && desc.find("Simd") == 0)
            AddToFunction(src, dst.simd, enable.simd);
        if (desc.find("Simd::Base::") != std::string::npos)
            AddToFunction(src, dst.base, enable.base);
        if (desc.find("Simd::Sse41::") != std::string::npos)
            AddToFunction(src, dst.sse41, enable.sse41);
        if (desc.find("Simd::Avx::") != std::string::npos)
            AddToFunction(src, dst.avx, enable.avx);
        if (desc.find("Simd::Avx2::") != std::string::npos)
            AddToFunction(src, dst.avx2, enable.avx2);
        if (desc.find("Simd::Avx512bw::") != std::string::npos)
            AddToFunction(src, dst.avx512bw, enable.avx512bw);
        if (desc.find("Simd::Avx512vnni::") != std::string::npos)
            AddToFunction(src, dst.avx512vnni, enable.avx512vnni);
        if (desc.find("Simd::Avx512bf16::") != std::string::npos)
            AddToFunction(src, dst.avx512bf16, enable.avx512bf16);
        if (desc.find("Simd::Amx::") != std::string::npos)
            AddToFunction(src, dst.amx, enable.amx);
        if (desc.find("Simd::Vmx::") != std::string::npos)
            AddToFunction(src, dst.vmx, enable.vmx);
        if (desc.find("Simd::Vsx::") != std::string::npos)
            AddToFunction(src, dst.vsx, enable.vsx);
        if (desc.find("Simd::Neon::") != std::string::npos)
            AddToFunction(src, dst.neon, enable.neon);
    }

    static inline const Function & Cond(const Function & a, const Function & b)
    {
        return a.first.Average() > 0 ? a : b;
    }

    static inline void Add(const Function & src, Common & dst)
    {
        dst.first.Add(src.first);
        dst.second.Add(src.second);
    }

    static void AddToCommon(const FunctionStatistic & s, const StatisticEnable & enable, CommonStatistic & d)
    {
        Add(s.simd, d.simd);
        if (enable.base) Add(Cond(s.base, s.simd), d.base);
        if (enable.sse41) Add(Cond(s.sse41, s.base), d.sse41);
        if (enable.avx) Add(Cond(s.avx, Cond(s.sse41, s.base)), d.avx);
        if (enable.avx2) Add(Cond(s.avx2, Cond(s.avx, Cond(s.sse41, s.base))), d.avx2);
        if (enable.avx512bw) Add(Cond(s.avx512bw, Cond(s.avx2, Cond(s.avx, Cond(s.sse41, s.base)))), d.avx512bw);
        if (enable.avx512vnni) Add(Cond(s.avx512vnni, Cond(s.avx512bw, Cond(s.avx2, Cond(s.avx, Cond(s.sse41, s.base))))), d.avx512vnni);
        if (enable.avx512bf16) Add(Cond(s.avx512bf16, Cond(s.avx512vnni, Cond(s.avx512bw, Cond(s.avx2, Cond(s.avx, Cond(s.sse41, s.base)))))), d.avx512bf16);
        if (enable.amx) Add(Cond(s.amx, Cond(s.avx512bf16, Cond(s.avx512vnni, Cond(s.avx512bw, Cond(s.avx2, Cond(s.avx, Cond(s.sse41, s.base))))))), d.amx);
        if (enable.vmx) Add(Cond(s.vmx, s.base), d.vmx);
        if (enable.vsx) Add(Cond(s.vsx, Cond(s.vmx, s.base)), d.vsx);
        if (enable.neon) Add(Cond(s.neon, s.base), d.neon);
    }

	static void AddHeader(Table & table, const StatisticNames & names, const StatisticEnable & enable, bool align)
	{
		size_t col = 0, last = 0;
        for (size_t i = 0; i < enable.Size(); ++i)
            if (enable[i])
                last = i;
		table.SetHeader(col++, "Function", true);
		for (size_t i = 0; i < enable.Size(); ++i)
			if (enable[i])
				table.SetHeader(col++, names[i].full, i == last, Table::Right);
        if (enable[1])
        {
            for (size_t i = 2; i < enable.Size(); ++i)
                if (enable[i])
                    table.SetHeader(col++, String(names[1].brief) + "/" + names[i].brief, i == last, Table::Right);
            for (size_t i = 2, p = 1; i < enable.Size(); ++i)
                if (enable[i])
                    table.SetHeader(col++, names[p].brief + String("/") + names[i].brief, i == last, Table::Right), p = i;
        }
		if (align)
		{
			for (size_t i = 0; i < enable.Size(); ++i)
				if (enable[i])
					table.SetHeader(col++,String(names[i].brief) + ":U/A", i == last, Table::Right);
		}
	}

    template <class Value> static void AddRow(Table & table, size_t row, const String & name, const Statistic<Value> & statistic, const StatisticEnable & enable, bool align, double timeMax)
    {
        const int V = (timeMax > 0.001 ? 3 : (timeMax > 0.0001 ? 1 : 2)), R = 2;
        size_t col = 0;
        table.SetCell(col++, row, name);
        for (size_t i = 0; i < statistic.Size(); ++i)
            if (enable[i])
                table.SetCell(col++, row, ToString(statistic[i].first.Average()*(timeMax < 0.001 ? 1000000.0 : 1000.0), V, false));
        if (enable[1])
        {
            for (size_t i = 2; i < statistic.Size(); ++i)
                if (enable[i])
                    table.SetCell(col++, row, ToString(Test::Relation(statistic[1].first, statistic[i].first), R, false));
            for (size_t i = 2; i < statistic.Size(); ++i)
                if (enable[i])
                    table.SetCell(col++, row, ToString(Test::Relation(Previous(statistic[i]).first, statistic[i].first), R, false));
        }
        if (align)
        {
            for (size_t i = 0; i < statistic.Size(); ++i)
                if (enable[i])
                    table.SetCell(col++, row, ToString(Test::Relation(statistic[i].second, statistic[i].first), R, false));
        }
    }

    void PerformanceMeasurerStorage::Combine(FunctionMap& map) const
    {
        std::lock_guard<std::recursive_mutex> lock(_mutex);
        for (ThreadMap::const_iterator thread = _map.begin(); thread != _map.end(); ++thread)
        {
            for (FunctionMap::const_iterator function = thread->second.map.begin(); function != thread->second.map.end(); ++function)
            {
                if (map.find(function->first) == map.end())
                    map[function->first].reset(new PerformanceMeasurer(function->first));
                map[function->first]->Combine(*function->second);
            }
        }
    }

    PerformanceMeasurerStorage::TablePtr PerformanceMeasurerStorage::GenerateTable(bool align) const
    {
        FunctionMap map;
        Combine(map);

        FunctionStatisticMap functions;
        CommonStatistic common;
        StatisticEnable enable = { false, false, false, false, false, false, false, false, false, false, false, false};
        StatisticNames names = { { "API", "A" },{ "Base", "Bs" },{ "Sse41", "S4" },{ "Avx", "A1" },{ "Avx2", "A2" },{ "Avx5b", "A5" },{ "Vnni", "Vn" },{ "Bf16", "Bf" },{ "Amx", "Am" },{ "Vmx", "Vm" },{ "Vsx", "Vs" },{ "Neon", "Ne" } };
        double timeMax = 0;
        for (FunctionMap::const_iterator it = map.begin(); it != map.end(); ++it)
        {
            const PerformanceMeasurer & pm = *it->second;
            String name = FunctionShortName(pm.Description());
            AddToFunction(pm, functions[name], enable);
            timeMax = std::max(timeMax, pm.Average());
        }

        for (FunctionStatisticMap::const_iterator it = functions.begin(); it != functions.end(); ++it)
            AddToCommon(it->second, enable, common);

        size_t size = 0;
        for (size_t i = 0; i < enable.Size(); ++i)
            if (enable[i])
                size++;
        TablePtr table(new Table(size*(align ? 4 : 3) - 3, 1 + functions.size()));
        AddHeader(*table, names, enable, align);
        size_t row = 0;
        table->SetRowProp(row, true, true);
        AddRow(*table, row++, String("Common, ") + (timeMax < 0.001 ? "us" : "ms"), common, enable, align, timeMax);
        for (FunctionStatisticMap::const_iterator it = functions.begin(); it != functions.end(); ++it)
            AddRow(*table, row++, it->first, it->second, enable, align, timeMax);
        return table;
    }

    static String TestTitle()
    {
        return "Simd Library Performance Report:";
    }

    static String TestInfo()
    {
        std::stringstream info;
        info << "Test execution time: " + GetCurrentDateTimeString();
        info << ". Simd Library version: " << SimdVersion() << ".";
        return info.str();
    }

    String PerformanceMeasurerStorage::ConsoleReport(bool align, bool raw) const
    {
        std::stringstream report;

        report << std::endl << std::endl << TestTitle() << std::endl << std::endl;

        report << TestInfo() << std::endl << std::endl;

        if (raw)
        {
            FunctionMap map;
            Combine(map);
            for (FunctionMap::const_iterator it = map.begin(); it != map.end(); ++it)
                report << it->second->Statistic() << std::endl;
        }
        else
        {
            report << GenerateTable(align)->GenerateText();
        }

        return report.str();
    }  

    bool PerformanceMeasurerStorage::TextReport(const String& path, bool align) const
    {
        CreatePathIfNotExist(path, true);
        std::ofstream file(path);
        if (!file.is_open())
            return false;

        file << TestTitle() << std::endl << std::endl;

        file << TestInfo() << std::endl << std::endl;

        file << GenerateTable(align)->GenerateText();

        file.close();

        return true;
    }
    
    bool PerformanceMeasurerStorage::HtmlReport(const String & path, bool align) const
    {
        CreatePathIfNotExist(path, true);
        std::ofstream file(path);
        if (!file.is_open())
            return false;

        Html html(file);

        html.WriteBegin("html", Html::Attr(), true, true);
        html.WriteValue("title", Html::Attr(), TestTitle(), true);
        html.WriteBegin("body", Html::Attr(), true, true);

        html.WriteValue("h1", Html::Attr("id", "home"), TestTitle(), true);

        html.WriteValue("h4", Html::Attr(), TestInfo(), true);

        html.WriteText(GenerateTable(align)->GenerateHtml(html.Indent()), false, false);

        html.WriteEnd("body", true, true);
        html.WriteEnd("html", true, true);

        file.close();

        return true;
    }

    void PerformanceMeasurerStorage::Clear()
    {
        _map.clear();
    }
}

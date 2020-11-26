/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2020 Yermalayeu Ihar.
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
#include "Simd/SimdPerformance.h"

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))
namespace Simd
{
    namespace Base
    {
        SIMD_INLINE double Miliseconds(int64_t count)
        {
            return double(count) / double(TimeFrequency()) * 1000.0;
        }

        PerformanceMeasurer::PerformanceMeasurer(const String& name, int64_t flop)
            : _name(name)
            , _flop(flop)
            , _count(0)
            , _current(0)
            , _total(0)
            , _min(std::numeric_limits<int64_t>::max())
            , _max(std::numeric_limits<int64_t>::min())
            , _entered(false)
            , _paused(false)
        {
        }

        PerformanceMeasurer::PerformanceMeasurer(const PerformanceMeasurer & pm)
            : _name(pm._name)
            , _flop(pm._flop)
            , _count(pm._count)
            , _start(pm._start)
            , _current(pm._current)
            , _total(pm._total)
            , _min(pm._min)
            , _max(pm._max)
            , _entered(pm._entered)
            , _paused(pm._paused)
        {
        }

        void PerformanceMeasurer::Enter()
        {
            if (!_entered)
            {
                _entered = true;
                _paused = false;
                _start = TimeCounter();
            }
        }

        void PerformanceMeasurer::Leave(bool pause)
        {
            if (_entered || _paused)
            {
                if (_entered)
                {
                    _entered = false;
                    _current += TimeCounter() - _start;
                }
                if (!pause)
                {
                    _total += _current;
                    _min = std::min(_min, _current);
                    _max = std::max(_max, _current);
                    ++_count;
                    _current = 0;
                }
                _paused = pause;
            }
        }

        String PerformanceMeasurer::Statistic() const
        {
            std::stringstream ss;
            ss << _name << ": ";
            ss << std::setprecision(0) << std::fixed << Miliseconds(_total) << " ms";
            ss << " / " << _count << " = ";
            ss << std::setprecision(3) << std::fixed << Average() << " ms";
            ss << std::setprecision(3) << " {min=" << Miliseconds(_min) << "; max=" << Miliseconds(_max) << "}";
            if (_flop)
                ss << " " << std::setprecision(1) << GFlops() << " GFlops";
            return ss.str();
        }

        void PerformanceMeasurer::Combine(const PerformanceMeasurer& other)
        {
            _count += other._count;
            _total += other._total;
            _min = std::min(_min, other._min);
            _max = std::max(_max, other._max);
        }

        double PerformanceMeasurer::Average() const
        {
            return _count ? (Miliseconds(_total) / _count) : 0;
        }

        double PerformanceMeasurer::GFlops() const
        {
            return _count && _flop && _total > 0 ? (double(_flop) * _count / Miliseconds(_total) / 1000000.0) : 0;
        }

        //---------------------------------------------------------------------

        PerformanceMeasurerStorage PerformanceMeasurerStorage::s_storage;

        const char * PerformanceMeasurerStorage::PerformanceStatistic()
        {
            if (_map.empty())
                return "";
            FunctionMap combined;
            std::lock_guard<std::mutex> lock(_mutex);
            for (ThreadMap::const_iterator thread = _map.begin(); thread != _map.end(); ++thread)
            {
                for (FunctionMap::const_iterator function = thread->second.begin(); function != thread->second.end(); ++function)
                {
                    if (combined.find(function->first) == combined.end())
                        combined[function->first].reset(new PerformanceMeasurer(*function->second));
                    else
                        combined[function->first]->Combine(*function->second);
                }
            }
            std::stringstream report;
            report << std::endl << "Simd Library Internal Performance Statistics:" << std::endl;
            for (FunctionMap::const_iterator it = combined.begin(); it != combined.end(); ++it)
                report << it->second->Statistic() << std::endl;
            _report = report.str();
            return _report.c_str();
        }
    }
}
#endif//SIMD_PERFORMANCE_STATISTIC

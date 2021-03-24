/*
* Simd Library (http://ermig1979.github.io/Simd).
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
#ifndef __SimdPerformance_h__
#define __SimdPerformance_h__

#include "Simd/SimdDefs.h"

#include <string>
#include <sstream>

namespace Simd
{
    typedef std::string String;

    template <class T> SIMD_INLINE String ToStr(const T & value)
    {
        std::stringstream ss;
        ss << value;
        return ss.str();
    }
}

#if defined(SIMD_PERFORMANCE_STATISTIC) && (defined(NDEBUG) || defined(SIMD_PERF_STAT_IN_DEBUG))

#include "Simd/SimdTime.h"

#include <limits>
#include <iostream>
#include <iomanip>
#include <memory>
#include <map>
#include <thread>
#include <mutex>
#include <algorithm>

namespace Simd
{
    namespace Base
    {
        class PerformanceMeasurer
        {
            String	_name;
            int64_t _start, _current, _total, _min, _max;
            int64_t _count, _flop;
            bool _entered, _paused;

        public:
            PerformanceMeasurer(const String& name = "Unknown", int64_t flop = 0);

            PerformanceMeasurer(const PerformanceMeasurer& pm);

            void Enter();

            void Leave(bool pause = false);

            String Statistic() const;

            void Combine(const PerformanceMeasurer& other);

        private:
            double Average() const;
            double GFlops() const;
        };

        class PerformanceMeasurerHolder
        {
            PerformanceMeasurer * _pm;

        public:
            SIMD_INLINE PerformanceMeasurerHolder(PerformanceMeasurer * pm, bool enter = true)
                : _pm(pm)
            {
                if (_pm && enter)
                    _pm->Enter();
            }

            SIMD_INLINE void Enter()
            {
                if (_pm)
                    _pm->Enter();
            }

            SIMD_INLINE void Leave(bool pause)
            {
                if (_pm)
                    _pm->Leave(pause);
            }

            SIMD_INLINE ~PerformanceMeasurerHolder()
            {
                if (_pm)
                    _pm->Leave();
            }
        };

        class PerformanceMeasurerStorage
        {
            typedef PerformanceMeasurer Pm;
            typedef std::shared_ptr<Pm> PmPtr;
            typedef std::map<String, PmPtr> FunctionMap;
            typedef std::map<std::thread::id, FunctionMap> ThreadMap;

            ThreadMap _map;
            mutable std::mutex _mutex;
            String _report;

            SIMD_INLINE FunctionMap & ThisThread()
            {
                static thread_local FunctionMap * thread = NULL;
                if (thread == NULL)
                {
                    std::lock_guard<std::mutex> lock(_mutex);
                    thread = &_map[std::this_thread::get_id()];
                }
                return *thread;
            }

        public:
            static PerformanceMeasurerStorage s_storage;

            PerformanceMeasurerStorage()
            {
            }

            SIMD_INLINE PerformanceMeasurer * Get(const String & name, int64_t flop = 0)
            {
                FunctionMap & thread = ThisThread();
                PerformanceMeasurer * pm = NULL;
                FunctionMap::iterator it = thread.find(name);
                if (it == thread.end())
                {
                    pm = new PerformanceMeasurer(name, flop);
                    thread[name].reset(pm);
                }
                else
                    pm = it->second.get();
                return pm;
            }

            SIMD_INLINE PerformanceMeasurer * Get(const String func, const String & desc, int64_t flop = 0)
            {
                return Get(func + "{ " + desc + " }", flop);
            }

            const char* PerformanceStatistic();
        };
    }
}
#define SIMD_PERF_FUNCF(flop) Simd::Base::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)(Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, (int64_t)(flop)))
#define SIMD_PERF_FUNC() SIMD_PERF_FUNCF(0)
#define SIMD_PERF_BEGF(desc, flop) Simd::Base::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)(Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, desc, (int64_t)(flop)))
#define SIMD_PERF_BEG(desc) SIMD_PERF_BEGF(desc, 0)
#define SIMD_PERF_IFF(cond, desc, flop) Simd::Base::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)((cond) ? Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, desc, (int64_t)(flop)) : NULL)
#define SIMD_PERF_IF(cond, desc) SIMD_PERF_IFF(cond, desc, 0)
#define SIMD_PERF_END(desc) Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, desc)->Leave();
#define SIMD_PERF_INITF(name, desc, flop) Simd::Base::PerformanceMeasurerHolder name(Simd::Base::PerformanceMeasurerStorage::s_storage.Get(SIMD_FUNCTION, desc, (int64_t)(flop)), false);
#define SIMD_PERF_INIT(name, desc)  SIMD_PERF_INITF(name, desc, 0);
#define SIMD_PERF_START(name) name.Enter(); 
#define SIMD_PERF_PAUSE(name) name.Leave(true);
#define SIMD_PERF_EXT(ext) Simd::Base::PerformanceMeasurerHolder SIMD_CAT(__pmh, __LINE__)((ext)->Perf(SIMD_FUNCTION)) 
#else//SIMD_PERFORMANCE_STATISTIC
#define SIMD_PERF_FUNCF(flop)
#define SIMD_PERF_FUNC()
#define SIMD_PERF_BEGF(desc, flop)
#define SIMD_PERF_BEG(desc)
#define SIMD_PERF_IFF(cond, desc, flop)
#define SIMD_PERF_IF(cond, desc)
#define SIMD_PERF_END(desc)
#define SIMD_PERF_INITF(name, desc, flop)
#define SIMD_PERF_INIT(name, desc)
#define SIMD_PERF_START(name)
#define SIMD_PERF_PAUSE(name)
#define SIMD_PERF_EXT(ext)
#endif//SIMD_PERFORMANCE_STATISTIC 

#endif//__SimdPerformance_h__

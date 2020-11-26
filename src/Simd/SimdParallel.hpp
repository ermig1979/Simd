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
#ifndef __SimdParallel_hpp__
#define __SimdParallel_hpp__

#include <vector>
#include <thread>
#ifndef SIMD_FUTURE_DISABLE
#include <future>
#endif

namespace Simd
{
    template<class Function> inline void Parallel(size_t begin, size_t end, const Function & function, size_t threadNumber, size_t blockAlign = 1)
    {
#ifdef SIMD_FUTURE_DISABLE
        function(0, begin, end);
#else
        threadNumber = std::min<size_t>(threadNumber, std::thread::hardware_concurrency());
        if (threadNumber <= 1 || size_t(blockAlign*1.5) >= (end - begin))
            function(0, begin, end);
        else
        {
            std::vector<std::future<void>> futures;

            size_t blockSize = (end - begin + threadNumber - 1) / threadNumber;
            blockSize = (blockSize + blockAlign - 1) / blockAlign * blockAlign;
            size_t blockBegin = begin;
            size_t blockEnd = blockBegin + blockSize;

            for (size_t thread = 0; thread < threadNumber && blockBegin < end; ++thread)
            {
                futures.push_back(std::move(std::async(std::launch::async, [blockBegin, blockEnd, thread, &function] { function(thread, blockBegin, blockEnd); })));
                blockBegin += blockSize;
                blockEnd = std::min(blockBegin + blockSize, end);
            }

            for (size_t i = 0; i < futures.size(); ++i)
                futures[i].wait();
        }
#endif
    }
}

#endif//__SimdParallel_hpp__

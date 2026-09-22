/*
* Simd Library (http://ermig1979.github.io/Simd).
*
* Copyright (c) 2011-2023 Yermalayeu Ihar.
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
    /*! @ingroup thread
    
        \fn void Parallel(size_t begin, size_t end, const Function & function, size_t threadNumber, size_t blockAlign = 1);

        \short Splits a half-open index range across threads and waits until every part finishes.

        Divides [begin, end) into contiguous blocks and calls
        function(thread, blockBegin, blockEnd) for each block. thread is the
        zero-based index of that block. The blocks do not overlap and together
        cover [begin, end). blockEnd is exclusive. The call returns after every
        callback has returned.

        The callback is usually a lambda with signature
        void(size_t thread, size_t blockBegin, size_t blockEnd).
        thread indexes a per-thread scratch buffer that the caller allocates
        for threadNumber slots. If the range is short, or if alignment yields
        fewer blocks than threadNumber, only the leading slots are used
        (indices 0, 1, ...). The other slots stay untouched.

        threadNumber is limited by std::thread::hardware_concurrency().
        If that value is 0, the limit is 0 and the call stays on the calling thread.
        The whole range is processed on the calling thread, as one call
        function(0, begin, end), when any of the following is true:
        - threadNumber is 0 or 1;
        - end - begin is less than or equal to size_t(blockAlign * 1.5)
          (for the default blockAlign of 1 this threshold is 1);
        - SIMD_FUTURE_DISABLE is defined, so std::future is not used.

        Otherwise every block except the last has a length that is a multiple
        of blockAlign. That length is (end - begin) divided by threadNumber,
        rounded up, and then rounded up to a multiple of blockAlign. The last
        block ends at end and may be shorter. The first block starts at begin,
        so alignment is relative to begin rather than to 0. blockAlign must be
        at least 1; the default is 1. end must be greater than or equal to begin.
        An empty range (begin == end) still performs one callback
        function(0, begin, begin).

        In the multi-thread path the callbacks of one call run at the same time.
        They may read shared inputs, and they must not write the same memory.
        The usual pattern stores private scratch in a buffer indexed by thread.
        A callback must accept a short tail: the last block can be shorter than
        blockAlign.

        Real usage:
        - ResizerNearest and WarpAffine split destination rows [0, dstH) with
          blockAlign 1. The thread count comes from ::SimdGetThreadNumber.
          Nearest resize also caps it by the output size in bytes divided by 4 MB.
          WarpAffine indexes a per-thread buffer as buf + thread * size.
        - GemmNN and GemmNT split the N dimension [0, N) with blockAlign equal
          to the micro-kernel width (_microN; it is 4 in GemmNT). thread selects
          the packed panel of that worker. A product with M * N * K below
          256 * 256 * 256 * 2 sets threadNumber to 1 before the call.
        - Detection scans rows [rect.top, rect.bottom). blockAlign is 2 when the
          cascade steps two rows at a time (through-column kernels), otherwise 1.
          threadNumber is forced to 1 when the ROI area is below 10000 pixels
          for a HAAR cascade, or below 30000 pixels for an LBP cascade.
        - Descriptor comparison splits [0, N) and writes partial results to
          buffer[thread]. blockAlign is the inner step, or a multiple of it
          (1, 256 or 1024), so every block except the tail is a whole number of steps.

        Using example:
        \code
        #include <vector>
        #include "Simd/SimdParallel.hpp"

        int main()
        {
            const size_t height = 480;
            const size_t threadNumber = 4;
            std::vector<int> scratch(threadNumber, 0);

            Simd::Parallel(0, height, [&](size_t thread, size_t begin, size_t end)
            {
                for (size_t y = begin; y < end; ++y)
                    scratch[thread] += (int)y;
            }, threadNumber, 1);

            return 0;
        }
        \endcode

        \param [in] begin - the first index of the range (inclusive).
        \param [in] end - the index one past the last element of the range.
        \param [in] function - a callback function(thread, blockBegin, blockEnd) invoked for one block.
        \param [in] threadNumber - the maximum number of blocks. Limited by std::thread::hardware_concurrency().
        \param [in] blockAlign - block length alignment, relative to begin. It must be at least 1. By default it is equal to 1.
    */
    template<class Function> inline void Parallel(size_t begin, size_t end, const Function & function, size_t threadNumber, size_t blockAlign = 1)
    {
#ifdef SIMD_FUTURE_DISABLE
        function(0, begin, end);
#else
        static const size_t threadNumberMax = std::thread::hardware_concurrency();
        threadNumber = std::min<size_t>(threadNumber, threadNumberMax);
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

/*
* Simd Library (http://simd.sourceforge.net).
*
* Copyright (c) 2011-2017 Yermalayeu Ihar.
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
#ifndef __SimdImageMatcher_hpp__
#define __SimdImageMatcher_hpp__

#include "Simd/SimdLib.hpp"

#include <vector>

namespace Simd
{
    template <class T, template<class> class A>
    struct ImageMatcher
    {
        typedef T Tag;
        typedef Simd::View<A> View; /*!< An image type definition. */
        typedef std::vector<uint8_t, A<uint8_t> > Vector;

        struct Data
        {
            Tag tag;

        private:
            Data(const Tag & t, size_t mainSize, size_t fastSize)
                : tag(t)
            {
                data.resize(mainSize + fastSize, 0);
                main = data.data();
                fast = main + mainSize;
            }

            Vector data;
            uint8_t * main;
            uint8_t * fast;

            friend struct ImageMatcher;
            friend struct Matcher;
            friend struct Matcher0D;
        };
        typedef std::shared_ptr<Data> DataPtr;

        struct Result
        {
            const Data * data;
            double difference;

            Result(const Data * da, double di)
                : data(da)
                , difference(di)
            {
            }
        };
        typedef std::vector<Result> Results;

        bool Empty() const
        {
            return !_matcher;
        }

        size_t Size() const
        {
            return _matcher ? _matcher->Size() : 0;
        }

        bool Init(double threshold, size_t size = 16, size_t number = 0, bool normalized = false)
        {
            if(number >= 10000 && threshold < 0.10)
                _matcher.reset(new Matcher_3D(threshold, size, number, normalized));
            else if(number > 1000 && !normalized)
                _matcher.reset(new Matcher_1D(threshold, size, number));
            else
                _matcher.reset(new Matcher_0D(threshold, size, number));
            return (bool)_matcher;
        }

        DataPtr Create(const View & view, const Tag & tag)
        {
            const size_t main = _matcher->main;
            const size_t fast = _matcher->fast;

            DataPtr data(DataPtr(new Data(tag, Square(main), Square(fast))));

            Simd::ResizeBilinear(view, View(main, main, main, View::Gray8, data->main).Ref());

            size_t step = main / fast;
            size_t area = Simd::Square(step);

            for (int fast_y = 0; fast_y < fast; ++fast_y)
            {
                for (int fast_x = 0; fast_x < fast; ++fast_x)
                {
                    size_t sum = area / 2;
                    for (size_t y = fast_y*step, y_end = y + step; y < y_end; ++y)
                    {
                        const uint8_t * pm = data->main + y*main;
                        for (size_t x = fast_x*step, x_end = x + step; x < x_end; ++x)
                            sum += pm[x];
                    }
                    data->fast[fast_y*fast + fast_x] = uint8_t(sum / area);
                }
            }

            return data;
        }

        bool Find(const DataPtr & data, Results & results)
        {
            results.clear();
            _matcher->Find(data, results);
            return results.size() != 0;
        }

        void Add(const DataPtr & data)
        {
            _matcher->Add(data);
        }

    private:
        typedef std::vector<DataPtr> Set;
        typedef std::vector<Set> Sets;

        struct Matcher
        {
            const size_t fast;
            const size_t main;

            Matcher(double threshold, size_t size)
                : fast(4)
                , main(size)
                , _fastSize(fast*fast)
                , _mainSize(size*size)
                , _size(0)
            {
                _fastMax = size_t(Square(threshold*UINT8_MAX)*_fastSize);
                _mainMax = size_t(Square(threshold*UINT8_MAX)*_mainSize);
            }

            size_t Size() const { return _size; }

            virtual ~Matcher() {}
            virtual void Add(const DataPtr & data) = 0; 
            virtual void Find(const DataPtr & data, Results & results) = 0;

        protected:
            Sets _sets;
            size_t _mainMax, _fastMax, _fastSize, _mainSize, _size;

            void FindIn(const Set & set, const DataPtr & data, Results & results)
            {
                for (size_t i = 0; i < set.size(); ++i)
                {
                    double difference = 0;
                    if (Compare(set[i], data, difference))
                        results.push_back(Result(set[i].get(), difference));
                }
            }

            bool Compare(const DataPtr & a, const DataPtr &  b, double & difference)
            {
                uint64_t fastSum = 0;
                ::SimdSquaredDifferenceSum(a->fast, _fastSize, b->fast, _fastSize, _fastSize, 1, &fastSum);
                if (fastSum > _fastMax)
                    return false;

                uint64_t mainSum = 0;
                ::SimdSquaredDifferenceSum(a->main, _mainSize, b->main, _mainSize, _mainSize, 1, &mainSum);
                if (mainSum > _mainMax)
                    return false;

                difference = ::sqrt(double(mainSum)/_mainSize/ UINT8_MAX/ UINT8_MAX);

                return true;
            }
        };
        typedef std::unique_ptr<Matcher> MatcherPtr;
        MatcherPtr _matcher;

        struct Matcher_0D : public Matcher
        {
            Matcher_0D(double threshold, size_t size, size_t number)
                : Matcher(threshold, size)
            {
                _sets.resize(1);
                _sets[0].reserve(number);
            }

            virtual void Add(const DataPtr & data)
            {
                _sets[0].push_back(data);
                _size++;
            }

            virtual void Find(const DataPtr & data, Results & results)
            {
                FindIn(_sets[0], data, results);
            }
        };

        struct Matcher_1D : public Matcher
        {
            Matcher_1D(double threshold, size_t size, size_t number)
                : Matcher(threshold, size)
                , _range(256)
            {
                _sets.resize(_range);
                _half = (int)ceil(0.5 + double(_range)*threshold);
            }

            virtual void Add(const DataPtr & data)            
            {
                _sets[Get(data)].push_back(data);
                _size++;
            }

            virtual void Find(const DataPtr & data, Results & results)
            {
                size_t index = Get(data);
                for (size_t i = std::max(index, _half) - _half, end = std::min(index + _half, _range); i < end; ++i)
                    FindIn(_sets[i], data, results);
            }

        private:
            size_t _range, _half;

            size_t Get(const DataPtr & data)
            {
                size_t sum = _fastSize/2;
                for (size_t i = 0; i < _fastSize; ++i)
                    sum += data->fast[i];
                return sum >> 4;
            }
        };

        struct Matcher_3D : public Matcher
        {
            Matcher_3D(double threshold, size_t size, size_t number, bool normalized)
                : Matcher(threshold, size)
                , _normalized(normalized)
            {
                const int MAX_RANGES[] = { 48, 48, 48, 48, 48, 48, 40, 32, 28, 24, 24 };
                _maxRange = MAX_RANGES[int(threshold / 0.01)];

                _shift.x = _maxRange >> 2;
                _shift.y = _maxRange >> 2;
                _shift.z = _normalized ? (_maxRange >> 2) : 0;

                _range.x = _maxRange >> 1;
                _range.y = _maxRange >> 1;
                _range.z = _normalized ? (_maxRange >> 1) : _maxRange;

                _stride.x = 1;
                _stride.y = _range.x;
                _stride.z = _range.x*_range.y;

                _sets.resize(_range.z*_range.x*_range.y);
                _half = (int)ceil(0.5 + double(_maxRange)*threshold);
            }

            virtual void Add(const DataPtr & data)
            {
                Index i;
                Get(data, i);
                _sets[i.x*_stride.x + i.y*_stride.y + i.z*_stride.z].push_back(data);
                _size++;
            }

            virtual void Find(const DataPtr & data, Results & results)
            {
                Index i, lo, hi;
                Get(data, i);

                lo.x = std::max(0, i.x - _half)*_stride.x;
                lo.y = std::max(0, i.y - _half)*_stride.y;
                lo.z = std::max(0, i.z - _half)*_stride.z;

                hi.x = std::min(_range.x, i.x + _half)*_stride.x;
                hi.y = std::min(_range.y, i.y + _half)*_stride.y;
                hi.z = std::min(_range.z, i.z + _half)*_stride.z;

                for (int z = lo.z; z < hi.z; z += _stride.z)
                    for (int y = lo.y; y < hi.y; y += _stride.y)
                        for (int x = lo.x; x < hi.x; x += _stride.x)
                            FindIn(_sets[x + y + z], data, results);
            }

        private:
            int _maxRange, _half;
            bool _normalized;

            struct Index
            {
                int x; 
                int y;
                int z;
            };
            Index _shift, _range, _stride;

            void Get(const DataPtr & data, Index & index)
            {
                const uint8_t * p = data->fast;
                int s[2][2];
                s[0][0] = p[0x0] + p[0x1] + p[0x4] + p[0x5];
                s[0][1] = p[0x2] + p[0x3] + p[0x6] + p[0x7];
                s[1][0] = p[0x8] + p[0x9] + p[0xC] + p[0xD];
                s[1][1] = p[0xA] + p[0xB] + p[0xE] + p[0xF];

                index.x = (s[0][0] - s[0][1] + s[1][0] - s[1][1] + 0x7FF)*_maxRange >> 12;
                index.y = (s[0][0] + s[0][1] - s[1][0] - s[1][1] + 0x7FF)*_maxRange >> 12;
                index.z = (s[0][0] + s[1][1] + (_normalized ? (0x7FF - s[1][0] - s[1][1]) : (s[1][0] + s[1][1])))*_maxRange >> 12;

                index.x = std::max(0, std::min(_range.x - 1, index.x - _shift.x));
                index.y = std::max(0, std::min(_range.y - 1, index.y - _shift.y));
                index.z = std::max(0, std::min(_range.z - 1, index.z - _shift.z));
            }
        };
    };
}

#endif//__SimdImageMatcher_hpp__
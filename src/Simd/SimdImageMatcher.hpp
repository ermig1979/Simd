/*
* Simd Library (http://ermig1979.github.io/Simd).
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
    /*! @ingroup cpp_image_matcher

        \short The ImageMatcher structure provides fast algorithm of searching of similar images.

        Using example (the filter removes duplicates from the list):
        \verbatim
        #include "Simd/SimdImageMatcher.hpp"

        typedef Simd::ImageMatcher<size_t, Simd::Allocator> ImageMatcher;
        typedef std::shared_ptr<View> ViewPtr;
        typedef std::vector<ViewPtr> ViewPtrs;

        void FilterDuplicates(const ViewPtrs & src, double threshold, ViewPtrs & dst)
        {
            ImageMatcher matcher;
            matcher.Init(threshold, ImageMatcher::Hash16x16, src.size());
            for (size_t i = 0; i < src.size(); ++i)
            {
                ImageMatcher::HashPtr hash = matcher.Create(*src[i], i);
                ImageMatcher::Results results;
                if (!matcher.Find(hash, results))
                {
                    matcher.Add(hash);
                    dst.push_back(src[i]);
                }
            }
        }
        \endverbatim
    */
    template <class Tag, template<class> class Allocator>
    struct ImageMatcher
    {
        typedef Simd::View<Allocator> View; /*!< An image type definition. */

        /*!
            \short The Hash structure is used for fast image matching.

            To create the structure use method Simd::ImageMatcher::Create().
        */
        struct Hash
        {
            Tag tag; /*!< An arbitrary tag linked with the image. */

        private:
            Hash(const Tag & t, size_t mainSize, size_t fastSize)
                : tag(t)
                , skip(false)
            {
                hash.resize(mainSize + fastSize, 0);
                main = hash.data();
                fast = main + mainSize;
            }

            std::vector<uint8_t, Allocator<uint8_t> > hash;
            uint8_t * main;
            uint8_t * fast;
            mutable bool skip;

            friend struct ImageMatcher;
        };
        typedef std::shared_ptr<Hash> HashPtr; /*!< A shared pointer to Hash structure. */

        /*!
            \short The Result structure is a result of matching current image and images added before to ImageMatcher.
        */
        struct Result
        {
            const Hash * hash; /*!< A hash to found similar image. */
            const double difference; /*!< A mean squared difference between current and found similar image. */

            /*!
                Creates a new Result structure.

                \param [in] h - a pointer to hash of found similar image.
                \param [in] d - A mean squared difference.
            */
            Result(const Hash * h, double d)
                : hash(h)
                , difference(d)
            {
            }
        };
        typedef std::vector<Result> Results; /*!< A vector with results. */

        /*!
            \enum HashType

            Describes size of reduced image used in image Hash.
        */
        enum HashType
        {
            Hash16x16, /*!< 16x16 reduced image size. */
            Hash32x32, /*!< 32x32 reduced image size. */
            Hash64x64, /*!< 32x32 reduced image size. */
        };

        /*!
            Signalizes true if ImageMatcher is initialized.

            \return true if ImageMatcher is initialized.
        */
        bool Empty() const
        {
            return !_matcher;
        }

        /*!
            Gets total number of images added to ImageMatcher.

            \return total number of images added to ImageMatcher.
        */
        size_t Size() const
        {
            return _matcher ? _matcher->Size() : 0;
        }

        /*!
            Initializes ImageMatcher for search.

            \param [in] threshold - a maximal mean squared difference for similar images. By default it is equal to 0.05.
            \param [in] type - a type of Hash used for matching. By default it is equal to ImageMatcher::Hash16x16.
            \param [in] number - an estimated total number of images used for matching. By default it is equal to 0.
            \param [in] normalized - a flag signalized that images have normalized histogram. By default it is false.
            \return the result of the operation.
        */
        bool Init(double threshold = 0.05, HashType type = Hash16x16, size_t number = 0, bool normalized = false)
        {
            static const size_t sizes[] = { 16, 32, 64 };
            size_t size = sizes[type];

            if (number >= 10000 && threshold < 0.10)
                _matcher.reset(new Matcher_3D(threshold, size, number, normalized));
            else if (number > 1000 && !normalized)
                _matcher.reset(new Matcher_1D(threshold, size, number));
            else
                _matcher.reset(new Matcher_0D(threshold, size, number));
            return (bool)_matcher;
        }

        /*!
            Creates hash for given image.

            \param [in] view - an input image.
            \param [in] tag - a tag of arbitrary type.
            \return the smart pointer to Hash for image matching.
        */
        HashPtr Create(const View & view, const Tag & tag)
        {
            const size_t main = _matcher->main;
            const size_t fast = _matcher->fast;

            HashPtr hash(HashPtr(new Hash(tag, Square(main), Square(fast))));

            View gray;
            if (view.format == View::Gray8)
                gray = view;
            else
            {
                gray.Recreate(view.Size(), View::Gray8);
                Simd::Convert(view, gray);
            }

            Simd::ResizeBilinear(gray, View(main, main, main, View::Gray8, hash->main).Ref());

            size_t step = main / fast;
            size_t area = Simd::Square(step);

            for (size_t fast_y = 0; fast_y < fast; ++fast_y)
            {
                for (size_t fast_x = 0; fast_x < fast; ++fast_x)
                {
                    size_t sum = area / 2;
                    for (size_t y = fast_y*step, y_end = y + step; y < y_end; ++y)
                    {
                        const uint8_t * pm = hash->main + y*main;
                        for (size_t x = fast_x*step, x_end = x + step; x < x_end; ++x)
                            sum += pm[x];
                    }
                    hash->fast[fast_y*fast + fast_x] = uint8_t(sum / area);
                }
            }

            return hash;
        }

        /*!
            Finds all similar images earlier added to ImageMatcher for given image.

            \param [in] hash - a smart pointer to hash of the image.
            \param [out] results - a list of found similar images.
            \return true if similar images were found.
        */
        bool Find(const HashPtr & hash, Results & results)
        {
            results.clear();
            _matcher->Find(hash, results);
            return results.size() != 0;
        }

        /*!
            Adds given image to ImageMatcher.

            \param [in] hash - a smart pointer to hash of the image.
        */
        void Add(const HashPtr & hash)
        {
            _matcher->Add(hash);
        }

        /*!
            Skips searching of the image in ImageMatcher.

            \param [in] hash - a smart pointer to hash of the image.
        */
        void Skip(const HashPtr & hash)
        {
            hash->skip = true;
        }

    private:
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
                , _threshold(threshold)
            {
                _fastMax = uint64_t(Square(threshold*UINT8_MAX)*_fastSize);
                _mainMax = uint64_t(Square(threshold*UINT8_MAX)*_mainSize);
            }

            size_t Size() const { return _size; }

            virtual ~Matcher() {}
            virtual void Add(const HashPtr & hash) = 0;
            virtual void Find(const HashPtr & hash, Results & results) = 0;

        protected:
            typedef std::vector<HashPtr> Set;
            typedef std::vector<Set> Sets;
            Sets _sets;
            size_t _fastSize, _mainSize, _size;
            uint64_t _mainMax, _fastMax;
            double _threshold;

            void AddIn(size_t index, const HashPtr & hash)
            {
                _sets[index].push_back(hash);
                _size++;
            }

            void FindIn(size_t index, const HashPtr & hash, Results & results)
            {
                const Set & set = _sets[index];
                for (size_t i = 0; i < set.size(); ++i)
                {
                    double difference = 0;
                    if (Compare(set[i], hash, difference))
                        results.push_back(Result(set[i].get(), difference));
                }
            }

            bool Compare(const HashPtr & a, const HashPtr &  b, double & difference)
            {
                if (a->skip || b->skip)
                    return false;

                uint64_t fastSum = 0;
                ::SimdSquaredDifferenceSum(a->fast, _fastSize, b->fast, _fastSize, _fastSize, 1, &fastSum);
                if (fastSum > _fastMax)
                    return false;

                uint64_t mainSum = 0;
                ::SimdSquaredDifferenceSum(a->main, _mainSize, b->main, _mainSize, _mainSize, 1, &mainSum);
                if (mainSum > _mainMax)
                    return false;

                difference = ::sqrt(double(mainSum) / _mainSize / UINT8_MAX / UINT8_MAX);

                return difference <= _threshold;
            }
        };
        typedef std::unique_ptr<Matcher> MatcherPtr;
        MatcherPtr _matcher;

        struct Matcher_0D : public Matcher
        {
            Matcher_0D(double threshold, size_t size, size_t number)
                : Matcher(threshold, size)
            {
                this->_sets.resize(1);
                this->_sets[0].reserve(number);
            }

            virtual void Add(const HashPtr & hash)
            {
                this->AddIn(0, hash);
            }

            virtual void Find(const HashPtr & hash, Results & results)
            {
                this->FindIn(0, hash, results);
            }
        };

        struct Matcher_1D : public Matcher
        {
            Matcher_1D(double threshold, size_t size, size_t number)
                : Matcher(threshold, size)
                , _range(256)
            {
                this->_sets.resize(_range);
                _half = (int)ceil(double(_range)*threshold);
            }

            virtual void Add(const HashPtr & hash)
            {
                this->AddIn(Get(hash), hash);
            }

            virtual void Find(const HashPtr & hash, Results & results)
            {
                size_t index = Get(hash);
                for (size_t i = std::max(index, _half) - _half, end = std::min(index + _half + 1, _range); i < end; ++i)
                    this->FindIn(i, hash, results);
            }

        private:
            size_t _range, _half;

            size_t Get(const HashPtr & hash)
            {
                size_t sum = 0;
                for (size_t i = 0; i < this->_fastSize; ++i)
                    sum += hash->fast[i];
                return sum >> 4;
            }
        };

        struct Matcher_3D : public Matcher
        {
            Matcher_3D(double threshold, size_t size, size_t number, bool normalized)
                : Matcher(threshold, size)
                , _normalized(normalized)
            {
                const int MAX_RANGES[] = { 96, 96, 96, 96, 96, 96, 80, 64, 56, 48, 48 };
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

                this->_sets.resize(_range.z*_range.x*_range.y);
                _half = (int)ceil(double(_maxRange)*threshold);
            }

            virtual void Add(const HashPtr & hash)
            {
                Index i;
                Get(hash, i);
                this->AddIn(i.x*_stride.x + i.y*_stride.y + i.z*_stride.z, hash);
            }

            virtual void Find(const HashPtr & hash, Results & results)
            {
                Index i, lo, hi;
                Get(hash, i);

                lo.x = std::max(0, i.x - _half)*_stride.x;
                lo.y = std::max(0, i.y - _half)*_stride.y;
                lo.z = std::max(0, i.z - _half)*_stride.z;

                hi.x = std::min(_range.x, i.x + _half + 1)*_stride.x;
                hi.y = std::min(_range.y, i.y + _half + 1)*_stride.y;
                hi.z = std::min(_range.z, i.z + _half + 1)*_stride.z;

                for (int z = lo.z; z < hi.z; z += _stride.z)
                    for (int y = lo.y; y < hi.y; y += _stride.y)
                        for (int x = lo.x; x < hi.x; x += _stride.x)
                            this->FindIn(x + y + z, hash, results);
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

            void Get(const HashPtr & hash, Index & index)
            {
                const uint8_t * p = hash->fast;
                int s[2][2];
                s[0][0] = p[0x0] + p[0x1] + p[0x4] + p[0x5];
                s[0][1] = p[0x2] + p[0x3] + p[0x6] + p[0x7];
                s[1][0] = p[0x8] + p[0x9] + p[0xC] + p[0xD];
                s[1][1] = p[0xA] + p[0xB] + p[0xE] + p[0xF];

                index.x = (s[0][0] - s[0][1] + s[1][0] - s[1][1] + 0x7FF)*_maxRange >> 12;
                index.y = (s[0][0] + s[0][1] - s[1][0] - s[1][1] + 0x7FF)*_maxRange >> 12;
                index.z = (s[0][0] + s[1][1] + (_normalized ? (0x7FF - s[1][0] - s[0][1]) : (s[1][0] + s[0][1])))*_maxRange >> 12;

                index.x = std::max(0, std::min(_range.x - 1, index.x - _shift.x));
                index.y = std::max(0, std::min(_range.y - 1, index.y - _shift.y));
                index.z = std::max(0, std::min(_range.z - 1, index.z - _shift.z));
            }
        };
    };
}

#endif//__SimdImageMatcher_hpp__

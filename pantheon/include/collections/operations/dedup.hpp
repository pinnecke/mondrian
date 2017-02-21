#ifndef PANTHEON_DEDUP_HPP
#define PANTHEON_DEDUP_HPP

namespace pantheon
{
    namespace collections
    {
        namespace operations
        {
            namespace dedup
            {
                using namespace pantheon::functional;

                /// Runs a deduplication on an array starting at \p begin and ending at \p end using
                /// a sorting-based approach. If required, a custom comparator \p comp can be used to compare
                /// elements. The parameter \p dest must be a non-value pointer that is capable to contain
                /// at least (\p end - \p begin) elements (if no duplication was contained in the array).
                /// The function returns a pointer to the element after the last element in the desintation array.
                ///
                /// \tparam ValueType type of array for both source and destination array
                /// \param begin begin of array that should be deduped
                /// \param end pointer to element after last element array that should be deduped
                /// \param dest pointer to an array that should contain the deduplicated content of the
                ///                          source array. Must be capable to contain at least as many elements as the
                ///                          source array.
                /// \return A pointer to the element after the last element of the destination array
                template<class ValueType, class Compare = comparators::function_t<ValueType>>
                ValueType *sort_based(const ValueType *begin, const ValueType *end, ValueType *dest,
                                      Compare comp = comparators::less<ValueType>()) {
                    assert (begin != nullptr);
                    assert (end != nullptr);
                    assert (dest != nullptr);

                    size_t n = 0;
                    for (auto lhs = begin; lhs != end; ++lhs) {
                        if (not binary_search(dest, dest + n, *lhs, comp)) {
                            *(dest + (n++)) = *lhs;
                            sort(dest, dest + n);
                        }
                    }
                    return dest + n;
                }
            }
        }
    }
}

#endif //PANTHEON_DEDUP_HPP

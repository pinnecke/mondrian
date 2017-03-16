// Vector-Pipes - a framework for the push-based iterator model with support of vectorized execution
// Copyright (C) 2017  Marcus Pinnecke (marcus.pinnecke@ovgu.de)
//
// This program is free software; you can redistribute it and/or modify it under the terms of the
// GNU General Public License as published by the Free Software Foundation; either version 2 of the
// License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
// even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License al ong with this program; if
// not, write to the Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
// Boston, MA  02110-1301, USA.

#pragma once

#include <vpipes.hpp>

namespace mondrian
{
    namespace vpipes
    {
        template<class Input, class InputTupletIdType = size_t>
        class consumer
        {
        public:
            using input_t = Input;
            using input_tupletid_t = InputTupletIdType;
            using input_chunk_t = chunk<input_t, input_tupletid_t>;
            using materializer_t = typename functional::batched_materializes<input_t, input_tupletid_t>::func_t;

            template<class IL, class IR, class ILTID, class IRTID>
            friend class bi_pipe_tail;

        private:
            materializer_t materialize_func;

        public:
            consumer(materializer_t&& materialize_func): materialize_func(materialize_func) { }

        protected:
            virtual void on_consume(input_tupletid_t *begin, input_tupletid_t *end) { };

            virtual void on_cleanup() { };

            inline virtual void lookup(input_t *out_values_begin, input_t *out_values_end,
                                       const input_tupletid_t *tid_begin,
                                       const input_tupletid_t *tid_end) final __attribute__((always_inline))
            {
                __builtin_prefetch(out_values_begin, PREFETCH_RW_FOR_WRITE, PREFETCH_LOCALITY_KEEP_IN_CACHES_HIGH);
                __builtin_prefetch(tid_begin, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_REMOVE_FROM_CACHE);
                materialize_func(out_values_begin, out_values_end, tid_begin, tid_end);
            }

        public:
            virtual void close() {
                on_cleanup();
            }

            inline virtual void consume(input_chunk_t *data) final __attribute__((always_inline))
            {
                auto iterator = data->get_iterator();
                if (__builtin_expect(!iterator.is_empty(), true)) {
                    on_consume(iterator.begin, iterator.end);
                }
            }

            inline virtual void consume(input_tupletid_t *begin, input_tupletid_t *end) final __attribute__((always_inline))
            {
                assert (begin != nullptr);
                assert (end != nullptr);
                assert (begin <= end);
                on_consume(begin, end);
            }
        };
    }
}
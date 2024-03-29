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
        template<class Input>
        class consumer
        {
            statistics::operator_run statistics;

        public:
            using input_t = Input;
            using input_batch_t = batch<input_t>;

            template<class IL, class IR, class ILTID, class IRTID>
            friend class bi_pipe_tail;

        protected:
            virtual void on_consume(__in__ const input_batch_t *data) { };

            virtual void on_cleanup() { };

        public:
            virtual void close() {
                on_cleanup();
            }

            inline virtual void consume(__in__ const input_batch_t *data) final __attribute__((always_inline))
            {
                statistics.num_batches++;
                if (__builtin_expect(!data->is_empty(), true)) {
                    statistics.num_tuplets += data->get_size();
                    on_consume(data);
                } else statistics.num_empty_batches++;
            }

            const statistics::operator_run *get_input_statistics() const
            {
                return &statistics;
            }

            virtual const char *get_class_name() const = 0;
        };
    }
}
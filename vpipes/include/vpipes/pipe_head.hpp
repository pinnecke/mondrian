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
        template<class Output>
        class pipe_head : public pipe<Output, Output>
        {
            using super = pipe<Output, Output>;

        public:
            using typename super::input_t;
            using typename super::consumer_t;

        private:
            tuplet_id_t *begin, *end;

        protected:
            virtual void on_start() = 0;

            inline virtual const tuplet_id_t *get_begin() final __attribute__((always_inline))
            {
                __builtin_prefetch(begin, PREFETCH_RW_FOR_READ, PREFETCH_LOCALITY_KEEP_IN_CACHES_NORMAL);
                return begin;
            }

            inline virtual const tuplet_id_t *get_end() final __attribute__((always_inline))
            {
                return end;
            }

        public:
            pipe_head(consumer_t *destination, tuplet_id_t *begin, tuplet_id_t *end,
                            unsigned batch_size) :
                            super(destination, batch_size), begin(begin), end(end) {};

            inline virtual void on_consume(tuplet_id_t *begin, tuplet_id_t *end) override final {};

            inline virtual void start() final { on_start(); }
        };
    }
}


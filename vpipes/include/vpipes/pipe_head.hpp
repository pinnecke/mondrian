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

#include "pipe.hpp"

namespace mondrian
{
    namespace vpipes
    {
        template<class Output, class OutputTupletIdType = size_t>
        class pipe_head : public pipe<Output, Output, OutputTupletIdType, OutputTupletIdType>
        {
            using super = pipe<Output, Output, OutputTupletIdType, OutputTupletIdType>;

        public:
            using typename super::input_t;
            using typename super::input_tupletid_t;
            using typename super::consumer_t;

        private:
            input_tupletid_t *begin, *end;

        protected:
            virtual void on_start() = 0;

            virtual const input_tupletid_t *get_begin() final { return begin; }

            virtual const input_tupletid_t *get_end() final { return end; }

        public:
            pipe_head(consumer_t *consumer, input_tupletid_t *begin, input_tupletid_t *end,
                            unsigned chunk_size) :
                            super(consumer, chunk_size), begin(begin), end(end) {};

            virtual void on_consume(input_tupletid_t *begin, input_tupletid_t *end) override final {};

            virtual void start() final { on_start(); }
        };
    }
}


#pragma once

#include <query_engine/operators/vector.hpp>

namespace mondrian {
    namespace query_engine {
        namespace operators {

            template<class ValueType>
            class push_operator {
            private:
                using vector_t = vector<ValueType>;
                push_operator<ValueType> *consumer;
                vector <ValueType> *result = nullptr;
                size_t size;

                void reset() { result = new vector<ValueType>(size); }

                void cleanup() {
                    if (result != nullptr) {
                        result->release();
                        delete (result);
                        result = nullptr;
                    }
                }

                void send() {
                    consumer->consume(result);
                    cleanup();
                }

            protected:
                virtual void
                on_consume(const ValueType **begin, const ValueType **end) { /* might be overridden */ };

                virtual void on_produce() { /* might be overridden */ };

                virtual void yield(const ValueType **value) final {
                    if (result->add(*value) == vector_t::state::full) {
                        send();
                        reset();
                    }
                }

                virtual void close() final {
                    if (consumer != nullptr) {
                        send();
                        consumer->close();
                    }
                    cleanup();
                }

                virtual ValueType lookup(const ValueType **ptr) final {
                    return **ptr;
                }

            public:
                push_operator(push_operator<ValueType> *consumer, unsigned vector_size) : consumer(consumer),
                                                                                          size(vector_size) {
                    reset();
                }

                virtual void produce() final { on_produce(); }

                virtual void consume(vector <ValueType> *data) final {
                    auto iterator = data->get_iterator();
                    on_consume(iterator.begin, iterator.end);
                }
            };

        }
    }
}

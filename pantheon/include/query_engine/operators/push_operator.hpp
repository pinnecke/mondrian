#pragma once

#include <query_engine/operators/vector.hpp>

namespace mondrian
{
    namespace query_engine
    {
        namespace operators
        {
            template<class InputType, class InputPointerType = InputType*>
            class push_operator
            {
            public:
                using input_t = InputType;
                using input_pointer_t = InputPointerType;

            private:
                using vector_t = vector<input_t, input_pointer_t>;
                push_operator<input_t, input_pointer_t> *consumer;
                vector <input_t, input_pointer_t> *result = nullptr;
                size_t size;

                void reset() { result = new vector<input_t, input_pointer_t>(size); }

                void cleanup()
                {
                    if (result != nullptr) {
                        result->release();
                        delete (result);
                        result = nullptr;
                    }
                }

                void send()
                {
                    consumer->consume(result);
                    cleanup();
                }

            protected:
                virtual void on_consume(const input_pointer_t *begin, const input_pointer_t *end) { };

                virtual void on_produce() { };

                virtual void on_close() { };

                virtual void on_cleanup() { };

                virtual void forward(const input_pointer_t *value) final
                {
                    if (result->add(*value) == vector_t::state::full) {
                        send();
                        reset();
                    }
                }

                virtual void close() final
                {
                    if (consumer != nullptr)
                    {
                        send();
                        reset();
                        on_close();
                        send();
                        consumer->close();
                        on_cleanup();
                    }
                    cleanup();
                }

                virtual input_t lookup(const input_pointer_t *ptr) final
                {
                    return **ptr;
                }

                virtual const input_pointer_t as_reference(const input_pointer_t *ptr) final
                {
                    return *ptr;
                }

            public:
                push_operator(push_operator<input_t, input_pointer_t> *consumer, unsigned vector_size) :
                        consumer(consumer), size(vector_size)
                {
                    reset();
                }

                virtual void produce() final { on_produce(); }

                virtual void consume(vector <input_t, input_pointer_t> *data) final
                {
                    auto iterator = data->get_iterator();
                    if (!iterator.is_empty())
                        on_consume(iterator.begin, iterator.end);
                }
            };

        }
    }
}

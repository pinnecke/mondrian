#ifndef PANTHEON_COLUMN_STORE_HPP
#define PANTHEON_COLUMN_STORE_HPP

#include <cstddef>
#include <vector>
#include <mutex>
#include <assert.h>
#include <cmath>

#include <utils/strings.hpp>
#include <logger.hpp>

using namespace std;

namespace pantheon
{
    namespace storage
    {
        namespace host
        {
            namespace column_store
            {
                template <class ValueType, class TupletIdType = size_t>
                class base_column
                {
                public:
                    using tuplet_id_t = TupletIdType;
                    using value_t = ValueType;
                    using to_string_function_t = pantheon::utils::strings::to_string::function_t <value_t>;

                    constexpr size_t value_t_min_value()
                    {
                        return std::numeric_limits<value_t>::min();
                    }

                    constexpr size_t value_t_max_value()
                    {
                        return std::numeric_limits<value_t>::max();
                    }

                    constexpr size_t get_maximum_tuplet_id()
                    {
                        return std::numeric_limits<tuplet_id_t>::max();
                    }

                    struct mem_info
                    {
                        size_t column_store_type_size;
                        size_t number_of_data_pages;
                        size_t total_flags_size;
                        size_t total_mutex_size;
                        size_t total_approx_free_mask_size;
                        size_t total_approx_null_mask_size;
                        size_t total_link_size;
                        size_t total_payload_size;
                    };

                protected:

                    struct record
                    {
                        TupletIdType tid;
                        ValueType payload;
                    };

                    struct data_page
                    {
                        struct push_result
                        {
                            const value_t *tail;
                            tuplet_id_t start, end;
                            value_t **value_ptr;
                        };


                        struct
                        {
                            vector<bool> null_mask;
                            vector<bool> in_use_mask;


                            struct
                            {
                                char is_readable : 1;
                                char is_writeable : 1;
                                char is_locked : 1;
                                char reserved : 6;
                            } flags;
                        } header;

                        mutex page_mutex;
                        base_column<value_t, tuplet_id_t> *parent;
                        record *content;
                        data_page *prev, *next;
                        size_t slots_in_use;

                        void lock()
                        {
                            this->page_mutex.lock();
                            this->header.flags.is_locked = true;
                        }

                        void unlock()
                        {
                            this->page_mutex.unlock();
                            this->header.flags.is_locked = false;
                        }

                        void push_payload(push_result * result, const value_t *begin, const value_t *end)
                        {
                            assert (result != nullptr);
                            assert (begin != nullptr);
                            assert (end != nullptr);
                            assert (end >= begin);

                           // lock();

                            size_t position = 0;
                            size_t max_portion = min(parent->per_page_capacity, size_t(end - begin));

                            result->value_ptr = (value_t **) malloc(max_portion * sizeof(value_t *));
                            assert (result->value_ptr != nullptr);
                            result->start = parent->next_tuplet_id;

                            while (position < max_portion) {
                                if (!this->header.in_use_mask[position]) {
                                    this->content[position].payload = begin[position];
                                    this->content[position].tid = parent->next_tuplet_id++;
                                    result->value_ptr[position] = &this->content[position].payload;

                                    this->header.in_use_mask[position++] = true;
                                    slots_in_use++;
                                }
                            }
                            result->end = parent->next_tuplet_id;
                            result->tail = begin + position;

                            //    unlock();
                        }

                        bool is_full()
                        {
                            return (slots_in_use == parent->per_page_capacity);
                        }

                    };

                    struct data_page_ref
                    {
                        data_page *page;
                        data_page_ref *prev, *next;
                    };

                    data_page *active_pages_head = nullptr, *active_pages_tail = nullptr;
                    data_page_ref *free_list_1st_tier = nullptr;
                    data_page *free_list_2nd_tier = nullptr;

                    struct invertex_page_index
                    {
                        value_t **tuplet_ids;
                        size_t size = 0, capacity = 0;

                        enum IndexError { Okay, MallocFailed, LimitReached, TupletIdAlreadyRegistered,
                                          TupletNotRegistered};

                        invertex_page_index(size_t capacity)
                        {
                            assert (capacity > 0);
                            tuplet_ids = (value_t **) malloc (capacity * sizeof(value_t *));
                            assert (tuplet_ids != nullptr);
                            this->capacity = capacity;
                            for (size_t idx = 0; idx < capacity; ++idx) {
                                tuplet_ids[idx] = nullptr;
                            }
                        }

                        IndexError link(tuplet_id_t tuplet_id, value_t *value_in_page)
                        {
                            assert (value_in_page != nullptr);

                            if (tuplet_id == numeric_limits<tuplet_id_t>::max()) {
                                LOG_WARNING("Link tuplet id %zu to value %p failed: tuplet id limit reached",
                                            size_t(tuplet_id), value_in_page);
                                return IndexError::LimitReached;
                            }

                            if (tuplet_id >= capacity)
                            {
                                size_t new_capacity = this->capacity * 1.4f;
                                if ((tuplet_ids = (value_t **) realloc(tuplet_ids, new_capacity * sizeof(value_t *))) == nullptr) {
                                    LOG_WARNING("Link tuplet id %zu to value %p failed: inverted index memory reallocation failed",
                                                size_t(tuplet_id), value_in_page);
                                    return IndexError::MallocFailed;
                                }
                                else {
                                    for (size_t idx = capacity; idx < new_capacity; ++idx)
                                    {
                                        tuplet_ids[idx] = nullptr;
                                    }
                                    this->capacity = new_capacity;
                                    LOG_INFO("Resized inverted page index %p", this);
                                }
                            }

                            if (tuplet_ids[tuplet_id] != nullptr)
                            {
                                LOG_WARNING("Link tuplet id %zu to value %p failed: tuplet: collision",
                                            size_t(tuplet_id), value_in_page);
                                return IndexError::TupletIdAlreadyRegistered;
                            }
                            else {
                                LOG_INFO("Created inverted page index for tuplet id %zu to value %p",
                                         size_t(tuplet_id), value_in_page);
                                tuplet_ids[tuplet_id] = value_in_page;
                                return IndexError::Okay;
                            }
                         }

                        IndexError update(tuplet_id_t tuplet_id, value_t *value_in_page)
                        {
                            IndexError result;
                            if ((result = unlink(&tuplet_id, &tuplet_id + 1)) != IndexError::Okay) {
                                LOG_WARNING("Update inverted page index for tuplet id %zu to value %p failed:"
                                            "unlink failed", size_t(tuplet_id), value_in_page);
                                return result;
                            }
                            else {
                                LOG_INFO("Invoke update for inverted page index for tuplet id %zu from value %p to value %p",
                                         size_t(tuplet_id), tuplet_ids[tuplet_id], value_in_page);
                                return link(tuplet_id, value_in_page);
                            }
                        }

                        IndexError unlink(const tuplet_id_t *begin, const tuplet_id_t *end)
                        {
                            for (const tuplet_id_t *it = begin; it != end; ++it) {
                                tuplet_id_t tuplet_id = *it;
                                assert (tuplet_id < capacity);
                                if (tuplet_ids[tuplet_id] == nullptr) { LOG_WARNING(
                                            "Unlink tuplet id %zu in inverted page index failed: "
                                                    "not registered", size_t(tuplet_id));
                                    return IndexError::TupletNotRegistered;
                                } else { LOG_INFO( "Removed tuplet id %zu from inverted page index",
                                            size_t(tuplet_id));

                                    tuplet_ids[tuplet_id] = nullptr;
                                    return IndexError::Okay;
                                }
                            }
                        }

                        value_t *get_value_by_tuplet_id(tuplet_id_t tuplet_id)
                        {
                            assert (tuplet_id < capacity);
                            assert (tuplet_ids[tuplet_id] != nullptr);
                            return tuplet_ids[tuplet_id];
                        }
                    } inverted_index;

                    mutex mutex;
                    tuplet_id_t next_tuplet_id;

                    struct
                    {
                        char is_locked : 1;

                    } flags;

                    char *name = nullptr;
                    size_t per_page_capacity, capacity;
                    to_string_function_t to_string_function;
                    size_t number_of_data_pages;

                public:
                    base_column(const char *column_name, size_t per_page_capacity, size_t capacity,
                                to_string_function_t to_string_function): to_string_function(to_string_function),
                                                                          inverted_index(capacity)
                    {
                        using namespace pantheon::utils::strings;

                        this->next_tuplet_id = 0;
                        this->capacity = capacity;
                        this->name = trim_inplace(strdup(column_name));
                        assert (strlen(this->name) > 0);
                        this->flags.is_locked = false;
                        create_data_pages_in_2nd_tier(per_page_capacity, capacity);
                    }

                    base_column(const base_column& other) = delete;
                    base_column(const base_column&& other) = delete;

                    void append(const value_t *values_begin, const value_t *values_end)
                    {
                        assert (values_begin != nullptr);
                        assert (values_end != nullptr);
                        lock();

                        next_step:

                        if (this->free_list_1st_tier != nullptr)
                        { // use free slots in pages that are already in active pages space

                        } else if (this->free_list_2nd_tier != nullptr)
                        { // re-link existing but unused pages that are not in active pages space
                            data_page *page = this->free_list_2nd_tier;
                            typename data_page::push_result result;
                            page->push_payload(&result, values_begin, values_end);

                            size_t value_idx = 0;
                            for (tuplet_id_t id = result.start; id != result.end; ++id, ++value_idx)
                            {
                                auto link_result = inverted_index.link(id, result.value_ptr[value_idx]);
                                assert (link_result == invertex_page_index::IndexError::Okay);
                            }
                            std::free(result.value_ptr);

                            move_page_from_2nd_tier(page);
                            if (result.tail != values_end) {
                                values_begin = result.tail;
                                goto next_step;
                            }
                        } else
                        { // create new data page in 2nd tier free list
                            this->capacity *= 1.4f;
                            create_data_pages_in_2nd_tier(this->per_page_capacity, this->capacity);
                            goto next_step;
                        }

                        unlock();
                    }

                    void remove(const tuplet_id_t *begin, const tuplet_id_t *end)
                    {
                        assert (begin != nullptr);
                        assert (end != nullptr);
                        assert (begin <= end);
                        inverted_index.unlink(begin, end);
                        // TODO:
                    }

                    void raw_print(FILE *file, const tuplet_id_t *begin, const tuplet_id_t *end)
                    {
                        assert (file != nullptr);
                        assert (begin != nullptr);
                        assert (end != nullptr);

                        fprintf(file, "tuplet id | value\n");
                        for (auto it = begin; it != end; ++it) {
                            auto value = inverted_index.get_value_by_tuplet_id(*it);
                            char *str = nullptr;
                            fprintf(file, "%zu | %s\n", *it, (value != nullptr ? (str = to_string_function(value)) : "(NULL)"));
                            if (str != nullptr)
                                std::free(str);
                        }
                    }

                    void get_memory_info(mem_info *info)
                    {
                        assert (info != nullptr);
                        lock();
                        info->column_store_type_size = sizeof(base_column<value_t , tuplet_id_t >);
                        info->number_of_data_pages = number_of_data_pages;
                        info->total_flags_size = info->total_mutex_size = info->total_approx_free_mask_size =
                            info->total_approx_null_mask_size = info->total_link_size = info->total_payload_size = 0;

                        for (auto it = active_pages_head; it != nullptr; it = it->next)
                        {
                            info->total_flags_size += sizeof(it->header.flags);
                            info->total_mutex_size += sizeof(it->page_mutex);
                            info->total_approx_free_mask_size += log2(float(sizeof(it->header.in_use_mask) * it->header.in_use_mask.capacity()));
                            info->total_approx_null_mask_size += log2(float(sizeof(it->header.null_mask) * it->header.null_mask.capacity()));
                            info->total_link_size += sizeof(it->prev) + sizeof(it->next);
                            info->total_payload_size += sizeof(it->content) * per_page_capacity;
                        }
                        unlock();
                    }

                    void to_string(FILE *file, bool line_breaks = true)
                    {
                        lock();
                        fprintf(file, "column_store(name='%s', locked=%s, pages=[%s",
                                this->name, BOOL_TO_STRING(this->flags.is_locked == 1), line_breaks ? "\n" : "");
                        for(data_page *cursor = active_pages_head; cursor != nullptr; cursor = cursor->next) {
                            page_to_string(file, cursor, line_breaks);
                            fprintf(file, cursor->next != nullptr? (line_breaks ? ", \n" : ", ") : "");
                        }
                        fprintf(file, "]");
                        unlock();
                    }

                protected:
                    void lock()
                    {
                        this->mutex.lock();
                        this->flags.is_locked = true;
                    }

                    void unlock()
                    {
                        this->mutex.unlock();
                        this->flags.is_locked = false;
                    }

                    void page_to_string(FILE *file, data_page *page, bool line_breaks)
                    {
                        assert(file != nullptr);
                        assert(page != nullptr);
                        fprintf(file, "\tdata_page(adr=0x%016lx, prev=0x%016lx, next=0x%016lx, "
                                        "locked=%s, readable=%s, writeable=%s, content=[%s",
                                (uintptr_t) page, (uintptr_t) page->prev, (uintptr_t) page->next,
                                BOOL_TO_STRING(page->header.flags.is_locked == 1),
                                BOOL_TO_STRING(page->header.flags.is_readable == 1),
                                BOOL_TO_STRING(page->header.flags.is_writeable == 1),
                                line_breaks ? "\n" : "");
                        for (size_t i = 0; i < this->per_page_capacity; ++i) {
                            bool null = page->header.null_mask[i];
                            bool in_use = page->header.in_use_mask[i];
                            record *rec = page->content + i;
                            char *payload_str = to_string_function(&rec->payload);
                            assert (payload_str != nullptr);
                            fprintf(file, "%s[slot=%zu, null=%s, in_use=%s, data='%s' (tuplet_id=%zu)]%s",
                                    line_breaks ? "\t\t" : "",
                                    i, BOOL_TO_STRING(null == 1), BOOL_TO_STRING(in_use == 1),
                                    (in_use? payload_str : (null? "(NULL)" : "(FREE)" )), (in_use? rec->tid : 0),
                                    line_breaks? "\n" : "");
                           // std::free (payload_str);
                        }
                        fprintf(file, "])");
                    }

                    data_page *create_page(data_page *prev, data_page *next) {
                        data_page *page = (data_page *) calloc(1, sizeof(data_page));
                        assert (page != nullptr);
                        page->parent = this;
                        page->content = (record *) calloc(per_page_capacity, sizeof(record));
                        page->header.flags.is_readable = true;
                        page->header.flags.is_writeable = true;
                        page->header.flags.is_locked = true;
                        page->header.in_use_mask.reserve(per_page_capacity);
                        page->header.null_mask.reserve(per_page_capacity);
                        page->prev = prev;
                        page->next = next;

                        return page;
                    }
                    void create_data_pages_in_2nd_tier(size_t per_page_capacity, size_t capacity)
                    {
                        assert (per_page_capacity > 0);
                        assert (capacity > 0);
                        assert (this->free_list_2nd_tier == nullptr);

                        this->per_page_capacity = per_page_capacity;
                        size_t num_of_pages = ceil(capacity/float(per_page_capacity));

                        this->free_list_2nd_tier = create_page(nullptr, nullptr);
                        data_page *succ = free_list_2nd_tier;

                        for (size_t i = 1; i < num_of_pages; ++i) {
                            data_page *page = create_page(succ, nullptr);
                            succ->next = page;
                            succ = page;
                        }

                        LOG_INFO("Created %d pages in 2nd tier free store (per-page-cap: %d, cap: %d)",
                                 num_of_pages, per_page_capacity, capacity);

                        number_of_data_pages = num_of_pages;
                    }

                    void move_page_from_2nd_tier(data_page *page)
                    {
                        assert (this->active_pages_head == nullptr || this->active_pages_tail != nullptr);

                        this->free_list_2nd_tier = this->free_list_2nd_tier->next;
                        page->next = nullptr;

                        if (this->active_pages_head == nullptr) {
                            page->prev = nullptr;
                            active_pages_head = page;
                            this->active_pages_tail = active_pages_head;
                        } else {
                            page->prev = active_pages_tail;
                            this->active_pages_tail->next = page;
                            active_pages_tail = page;
                        }

                        if (!page->is_full()) {
                            data_page_ref *ref = (data_page_ref *) malloc(sizeof(data_page_ref));
                            ref->page = page;
                            ref->prev = nullptr;
                            ref->next = this->free_list_1st_tier;
                            this->free_list_1st_tier = ref;
                            LOG_INFO("Created link to page %p in 1st tier free store",
                                     page);
                        } else {
                            LOG_INFO("Moved page %p from 2nd tier free store to 1st tier free store",
                                     page);
                        }
                    }

                };
            }
        }
    }
}

#endif //PANTHEON_COLUMN_STORE_HPP

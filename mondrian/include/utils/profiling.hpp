#ifndef PANTHEON_PROFILING_HPP
#define PANTHEON_PROFILING_HPP

#include <chrono>
#include <utility>

namespace mondrian {
    namespace utils {
        namespace profiling {
            /* Taken from http://stackoverflow.com/questions/2808398/easily-measure-elapsed-time */
            template<typename TimeT = std::chrono::nanoseconds>
            struct measure
            {
                template<typename F, typename ...Args>
                static typename TimeT::rep execute(F &&func, Args &&... args)
                {
                    auto start = std::chrono::steady_clock::now();
                    std::forward<decltype(func)>(func)(std::forward<Args>(args)...);
                    auto duration = std::chrono::duration_cast< TimeT>
                            (std::chrono::steady_clock::now() - start);
                    return duration.count();
                }
            };
        }
    }
}



#endif //PANTHEON_PROFILING_HPP

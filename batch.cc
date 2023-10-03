#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <queue>

struct Lock
{

    const int batch_size = 64;

    std::atomic<int> loading{0};
    std::queue<int> index_queue{};
    std::shared_mutex queue_mutex{};

    Lock()
    {
        for (int i = 0; i < batch_size; ++i)
        {
            index_queue.push(i);
        }
    }

    int push(int input) {


        queue_mutex.lock();

        index_queue.


    }
};
#ifndef __NEW_ARMOR_DETECTOR__THREAD_POOL_HPP__
#define __NEW_ARMOR_DETECTOR__THREAD_POOL_HPP__

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <thread>

namespace rm_auto_aim
{
template <typename T>
class ThreadPool
{
   public:
    ThreadPool(size_t _num_threads);
    ThreadPool(size_t _num_threads, std::function<void(T)> _func);

    ~ThreadPool();

    // void enqueueTask(std::function<void()> _task);
    void enqueueObj2BeProcess(T _obj);

   private:
    std::vector<std::thread> workers_;
    std::queue<T> obj_queue_;
    // single functional task
    //  void function(T _obj);

    std::function<void(T)> function_;

    // Obj waiting to be processed by the functional task

    std::mutex queue_mutex_;
    std::condition_variable condition_;
    bool stop_;
};

template <typename T>
ThreadPool<T>::ThreadPool(size_t _num_threads, std::function<void(T)> _func) : stop_(false), function_(_func)
{
    for (size_t i = 0; i < _num_threads; ++i)
    {
        workers_.emplace_back(
            [this]()
            {
                while (true)
                {
                    T obj;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex_);
                        this->condition_.wait(lock, [this]() { return this->stop_ || !this->obj_queue_.empty(); });
                        if (this->stop_ && this->obj_queue_.empty())
                            return;
                        // this->function_(this->obj_queue_.pop());
                        obj = std::move(this->obj_queue_.front());
                        this->obj_queue_.pop();
                    }
                    std::invoke(this->function_, obj);
                }
            });
    }
}

template <typename T>
ThreadPool<T>::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread &worker : workers_)
        worker.join();
}

template <typename T>
void ThreadPool<T>::enqueueObj2BeProcess(T _obj)
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        obj_queue_.push(std::move(_obj));
    }
    condition_.notify_one();
}

}  // namespace rm_auto_aim
#endif
#pragma once

#include <deque>

template<typename TData, typename TContainer = std::deque<TData>>
class concurrent_queue
{
	private:
	size_t m_capacity;
	TContainer m_queue;
	
	mutable boost::mutex *ptr_m_mutex;

	boost::condition_variable *ptr_m_queue_empty;
	boost::condition_variable *ptr_m_queue_full;


	public:
	concurrent_queue(size_t capacity = 10)
	{
		ptr_m_mutex = new boost::mutex;
		ptr_m_queue_empty = new boost::condition_variable;
		ptr_m_queue_full = new boost::condition_variable;

		set_capacity(capacity);
	}
	~concurrent_queue()
	{
		delete ptr_m_mutex;
		delete ptr_m_queue_empty;
		delete ptr_m_queue_full;
	}
	void set_capacity(size_t capacity)
	{
		m_capacity = capacity;
	}

	size_t get_capacity()
	{
		return m_capacity;
	}
	size_t size() const
	{
		boost::mutex::scoped_lock lock(*ptr_m_mutex);
		return m_queue.size();
	}
	void clear()
	{
		boost::mutex::scoped_lock lock(*ptr_m_mutex);
		m_queue.clear();
		if(m_capacity != 0)
		ptr_m_queue_full->notify_one();
	}
	bool empty() const
	{
		boost::mutex::scoped_lock lock(*ptr_m_mutex);
		return m_queue.empty();
	}

	void push(const TData &data)
	{
		TData t_data = data;
		boost::mutex::scoped_lock lock(*ptr_m_mutex);
		if(m_capacity != 0)
		{
			while(m_queue.size() >= m_capacity)
			{
				ptr_m_queue_full->wait(lock);
			}
		}
		m_queue.push_back(t_data);
		lock.unlock();
		ptr_m_queue_empty->notify_one();
	}

	TData pop()
	{
		boost::mutex::scoped_lock lock(*ptr_m_mutex);
		while(m_queue.empty())
		{
			ptr_m_queue_empty->wait(lock);
		}
		TData popped_value = std::move(m_queue.front());
		m_queue.pop_front();

		lock.unlock();
		if(m_capacity != 0)
			ptr_m_queue_full->notify_one();
		return std::move(popped_value);
	}

	bool try_pop(TData& popped_value)
	{
		boost::mutex::scoped_lock lock(*ptr_m_mutex);
		if(m_queue.empty())
			return false;
		popped_value = std::move(m_queue.front());
		m_queue.pop_front();
		return true;
	}
};
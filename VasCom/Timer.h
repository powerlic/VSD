#pragma once
#include <Windows.h>
#include "Thread.h"

class Timer : public Thread
{
	typedef void(CALLBACK *Timerfunc)(void* p);
	typedef Timerfunc TimerHandler;
public:
	Timer()
		:m_handler(0)
		, m_interval(-1)
	{
	}

	void registerHandler(TimerHandler handler, void* p)
	{
		m_handler = handler;
		m_parameter = p;
	}

	void setInterval(int millisecond)
	{
		m_interval = millisecond;
	}

	void Run()
	{
		unsigned long tickNow = ::GetTickCount();
		unsigned long tickLastTime = tickNow;

		while (!IsStop())
		{
			tickNow = ::GetTickCount();
			if (tickNow - tickLastTime > m_interval)
			{
				if (m_handler)
				{
					(*m_handler)(m_parameter);
				}
				tickLastTime = ::GetTickCount();
			}

			::Sleep(1);
		}
	}

	void Cancel()
	{
		Stop();
	}

private:
	TimerHandler m_handler;
	int             m_interval;
	void*         m_parameter;
};



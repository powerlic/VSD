#pragma once
#include"VasCom.h"
#include"VasMsgTrans.h"
#include"VasIO.h"

void MsgProcFun(const string &msg);

void TestSendRegResult();
void TestHeartbeat();
void TestFeedback();
void TestVasResultReturn();
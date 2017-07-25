/*
 * Copyright (c) 2013 Pavel Shramov <shramov@mexmat.net>
 *
 * json2pb is free software; you can redistribute it and/or modify
 * it under the terms of the MIT license. See LICENSE for details.
 */

#ifndef __JSON2PB_H__
#define __JSON2PB_H__

#include <string>

namespace google
{
namespace protobuf
{
	class Message;
}
}

#ifdef _DEBUG
#pragma comment(lib, "jansson_d.lib")
#else
#pragma comment(lib, "jansson.lib")
#endif

void json2pb(google::protobuf::Message &msg, const char *buf, size_t size);
std::string pb2json(const google::protobuf::Message &msg);

#endif//__JSON2PB_H__

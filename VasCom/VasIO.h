#include "VasProto.prototxt.pb.h"
#include "google/protobuf/message.h"

#include <stdio.h> 
#include <stdlib.h>  
#include <string>
#include <fstream>
#include <sys/types.h> 
#include <sys/stat.h>
#include <fcntl.h>
#include <io.h>

namespace vas 
{
	using ::google::protobuf::Message;

	bool ReadProtoFromTextFile(const char *file_name,Message*proto);
	void WriteProtoToTextFile(const Message& proto, const char* filename);
	void TestProto();
}



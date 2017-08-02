#include"stdafx.h"
#include "VasIO.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

namespace vas
{
	using google::protobuf::io::FileInputStream;
	using google::protobuf::io::FileOutputStream;
	using google::protobuf::io::ZeroCopyInputStream;
	using google::protobuf::io::CodedInputStream;
	using google::protobuf::io::ZeroCopyOutputStream;
	using google::protobuf::io::CodedOutputStream;
	using google::protobuf::Message;


	bool ReadProtoFromTextFile(const char *file_name, Message*proto)
	{
		int fd = _open(file_name, O_RDONLY);
		CHECK_NE(fd, -1) << "File not found: " << file_name;
		FileInputStream* input = new FileInputStream(fd);
		bool success = google::protobuf::TextFormat::Parse(input, proto);
		delete input;
		_close(fd);
		return success;
	}
	void WriteProtoToTextFile(const Message& proto, const char* filename)
	{
		int fd = _open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
		FileOutputStream* output = new FileOutputStream(fd);
		CHECK(google::protobuf::TextFormat::Print(proto, output));
		delete output;
		_close(fd);
	}
	

	/*void TestProto()
	{
		VasService vas_services;
		const char *file_name = "..\\proto\\service_list.prototxt";
		ReadProtoFromTextFile(file_name, &vas_services);

		for (size_t i = 0; i < vas_services.service_paramter_size(); i++)
		{
			ServiceParameter service_para = vas_services.service_paramter(i);
			DecodeParameter decode_para = service_para.decode_parameter();
			DetectParameter detect_para = service_para.detect_parameter();
			if (decode_para.has_dst_width())
			{
				std::cout << decode_para.dst_height()<<std::endl;
			}
			if (decode_para.has_decode_method())
			{
				std::cout << decode_para.decode_method() << std::endl;
			}
			BgParameter bg_para = detect_para.bg_parameter();
			BgGuassianParameter bg_guassian_para = bg_para.guassian_parameter();
			std::cout << bg_guassian_para.num_history() << std::endl;
			for (size_t j = 0; j < detect_para.reg_type_size(); j++)
			{
				std::cout << detect_para.reg_type(j) << std::endl;
			}
		}


	}*/
}

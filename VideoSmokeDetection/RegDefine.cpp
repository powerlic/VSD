#include"stdafx.h"
#include"RegDefine.h"


const char* GetVasRegTypeName(RegType reg_type)
{
	for (size_t i = 0; reg_info_list[i].reg_type_name; i++)
	{
		if (reg_type == reg_info_list[i].reg_type)
		{
			return  reg_info_list[i].reg_type_name;
		}
	}
}

const char* GetVasRegModeName(RegMode reg_mode)
{
	for (size_t i = 0; reg_mode_name_list[i].reg_mode_name; i++)
	{
		if (reg_mode == reg_mode_name_list[i].reg_mode)
		{
			return  reg_mode_name_list[i].reg_mode_name;
		}
	}
}
const char* GetVasWeatherName(Weather weather)
{
	for (size_t i = 0; weather_name_list[i].weather_name; i++)
	{
		if (weather == weather_name_list[i].weather_type)
		{
			return  weather_name_list[i].weather_name;
		}
	}

}

const char* GetVasRegSceneName(RegScene reg_scene)
{
	for (size_t i = 0; reg_scene_name_list[i].reg_scene_name; i++)
	{
		if (reg_scene == reg_scene_name_list[i].reg_scene)
		{
			return  reg_scene_name_list[i].reg_scene_name;
		}
	}

}


const char* GetVasNigthtimeName(Nighttime nighttime)
{
	for (size_t i = 0; nighttime_name_list[i].night_time_name; i++)
	{
		if (nighttime == nighttime_name_list[i].nighttime)
		{
			return  nighttime_name_list[i].night_time_name;
		}
	}
}

const RegType &GetVasRegType(const string &reg_para_str)
{
	RegType reg_type = UNKOWNTYPE;
	for (size_t i = 0; reg_info_list[i].reg_type_name; i++)
	{
		if (reg_para_str.compare(reg_info_list[i].reg_type_name) == 0)
		{
			return reg_info_list[i].reg_type;
		}
	}
	return reg_type;
}
const RegType &GetVasRegType(int reg_type_code)
{
	RegType reg_type = UNKOWNTYPE;
	for (size_t i = 0; reg_info_list[i].reg_type_name; i++)
	{
		if (reg_type_code == reg_info_list[i].reg_type_code)
		{
			return reg_info_list[i].reg_type;
		}
	}
	return reg_type;
}

const RegScene &GetVasRegScene(const string &reg_para_str)
{
	for (size_t i = 0; reg_scene_name_list[i].reg_scene_name; i++)
	{
		if (reg_para_str.compare(reg_scene_name_list[i].reg_scene_name) == 0)
		{
			return reg_scene_name_list[i].reg_scene;
		}
	}
	return UNKOWNSCENE;
}

const RegScene &GetVasRegScene(int reg_scene_code)
{
	for (size_t i = 0; reg_scene_name_list[i].reg_scene_name; i++)
	{
		if (reg_scene_code == reg_scene_name_list[i].scene_code)
		{
			return reg_scene_name_list[i].reg_scene;
		}
	}
	return UNKOWNSCENE;
}

const Weather& GetVasRegWeather(const string &reg_para_str)
{
	for (size_t i = 0; weather_name_list[i].weather_name; i++)
	{
		if (reg_para_str.compare(weather_name_list[i].weather_name) == 0)
		{
			return weather_name_list[i].weather_type;
		}
	}
	return UNKOWNWEATHER;
}
const Weather& GetVasRegWeather(int weather_code)
{
	for (size_t i = 0; weather_name_list[i].weather_name; i++)
	{
		if (weather_code == weather_name_list[i].weather_code)
		{
			return weather_name_list[i].weather_type;
		}
	}
	return UNKOWNWEATHER;
}

const Nighttime& GetVasRegNighttime(const string &reg_para_str)
{
	for (size_t i = 0; nighttime_name_list[i].night_time_name; i++)
	{
		if (reg_para_str.compare(nighttime_name_list[i].night_time_name) == 0)
		{
			return nighttime_name_list[i].nighttime;
		}
	}
	return UNKNOWNNIGHTIME;
}

const Nighttime& GetVasRegNighttime(int nighttime_code)
{
	for (size_t i = 0; nighttime_name_list[i].night_time_name; i++)
	{
		if (nighttime_code == nighttime_name_list[i].nighttime_code)
		{
			return nighttime_name_list[i].nighttime;
		}
	}
	return UNKNOWNNIGHTIME;
}

RegPara GetRegParaByModelVersion(const string &model_version)
{
	string reg_type_code_str = model_version.substr(0, 3);
	string reg_scene_code_str = model_version.substr(3, 3);
	string weather_str = model_version.substr(6, 2);
	string nighttime_str = model_version.substr(8, 1);

	int reg_type_code = atoi(reg_type_code_str.c_str());
	int reg_scene_code = atoi(reg_scene_code_str.c_str());
	int weather_code = atoi(weather_str.c_str());
	int nighttime_code = atoi(nighttime_str.c_str());

	RegType reg_type = GetVasRegType(reg_type_code);
	RegScene reg_scene = GetVasRegScene(reg_scene_code);
	Weather weather = GetVasRegWeather(weather_code);
	Nighttime nighttime = GetVasRegNighttime(nighttime_code);


	if (reg_type != UNKOWNTYPE&&reg_scene != UNKOWNSCENE&&weather != UNKOWNWEATHER&&nighttime != UNKNOWNNIGHTIME)
	{
		RegPara reg_para(reg_type, reg_scene, weather, nighttime);
		return reg_para;
	}
	RegPara error_reg_para;
	return error_reg_para;


}
ModelFilePath GetModelFilePathFromEvent(const ModelFileEvent &model_file_event)
{
	//ModelFilePath(const string &deploy = "", const string &label = "", const string &trained = "", const string &mean = "", const string &set_model_version = "")
	ModelFilePath model_file_path(model_file_event.caffe_file_path.deploy_file_path, model_file_event.caffe_file_path.label_file_path, model_file_event.caffe_file_path.trained_file_path, model_file_event.caffe_file_path.mean_file_path, model_file_event.model_version);
	return model_file_path;
}
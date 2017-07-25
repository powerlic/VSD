#pragma once

#include<string>

using namespace std;

enum RegMode
{
	unkown_mode = 0,
	image_mode = 1,
	video_mode = 2,
};

enum RegStauts
{
	reg_on,
	reg_off
};


enum RegType
{
	UNKOWNTYPE=0,
	SMOKE=1,
	FIRE=2,
	HUMAN=3,
	
};

enum RegScene
{
	UNKOWNSCENE=0,
	INDOOR=1,
	FOREST=2,
	VILLAGE=3,
	STREET=4,
	SQUARE=5,
	
};


enum Weather
{
	SUNNY=1,
	CLOUDY=2,
	OVERCAST=3,//“ıÃÏ
	SNOWY=4,
	RAINY=5,
	FOGGY=6,//ŒÌÃÏ
	WINDY=7,
	UNKOWNWEATHER
};

enum Nighttime
{
	DAY=1,
	NIGHT=2,
	UNKNOWNNIGHTIME
};

struct RegInfo
{
	char* reg_type_name;
	RegType reg_type;
	int reg_type_code;
};

const RegInfo reg_info_list[] = 
{
	{ "human", HUMAN,3},
	{"fire",FIRE,2},
	{ "smoke", SMOKE, 1 },
	{ "unkowntype", UNKOWNTYPE, 0 }
};

struct RegModeName
{
	char *reg_mode_name;
	RegMode reg_mode;
	int reg_mode_code;
};


const RegModeName reg_mode_name_list[] =
{
	{ "image", image_mode,1 },
	{ "video", video_mode,2 },
	{ "unknownmode", unkown_mode,0 }
};

struct WeatherName
{
	const char *weather_name;
	Weather weather_type;
	int weather_code;
};


const WeatherName weather_name_list[] = 
{
	{ "sunny", SUNNY,1},
	{ "cloudy", CLOUDY,2 },
	{ "overcast", OVERCAST,3 },
	{ "snowy", SNOWY,4 },
	{ "rainy", RAINY,5 },
	{ "fogy", FOGGY,6},
	{ "windy", WINDY,7},
	{"unkownweather", UNKOWNWEATHER,0}
};

struct RegSceneName
{
	const char *reg_scene_name;
	RegScene reg_scene;
	int scene_code;
};

const RegSceneName reg_scene_name_list[] =
{
	{"indoor", INDOOR,1 },
	{"forest", FOREST,2},
	{"village", VILLAGE,3 },
	{"street", STREET,4},
	{"square",SQUARE,5},
	{ "unknownscene",UNKOWNSCENE,0}
};

struct NighttimeName
{
	const char *night_time_name;
	Nighttime nighttime;
	int nighttime_code;
};


const NighttimeName nighttime_name_list[]=
{
	{ "day", DAY,1},
	{"night",NIGHT,2},
	{"unknownnighttime",UNKNOWNNIGHTIME,0}
};


struct RegPara
{
	RegType reg_type;
	RegScene reg_scene;
	Weather weather;
	Nighttime nightime;

	bool operator==(const RegPara reg_para2)
	{
		return
			(
			reg_para2.reg_type == reg_type
			&&reg_para2.reg_scene == reg_scene
			&&reg_para2.weather == weather
			&&reg_para2.nightime == nightime
			);
	};
	RegPara(RegType set_reg_type=UNKOWNTYPE, RegScene set_reg_scene=UNKOWNSCENE, Weather set_weather=UNKOWNWEATHER, Nighttime set_nightime=UNKNOWNNIGHTIME)
	{
		reg_type = set_reg_type;
		reg_scene = set_reg_scene;
		weather = set_weather;
		nightime = set_nightime;
	}
};



//model event

enum ModelEventType
{
	ModelAdd = 1,
	ModelChanged = 2,
	UNKOWNMODELEVENTTYPE = 3
};


struct CaffeFilePath
{
	string deploy_file_path;
	string trained_file_path;
	string mean_file_path;
	string label_file_path;
	CaffeFilePath(const string &deploy = "", const string &label = "", const string &trained = "", const string &mean = "")
	{
		deploy_file_path = deploy;
		label_file_path = label;
		trained_file_path = trained;
		mean_file_path = mean;
	}
	const CaffeFilePath operator=(const CaffeFilePath& p)
	{
		CaffeFilePath path;
		path.deploy_file_path = p.deploy_file_path;
		path.trained_file_path = p.trained_file_path;
		path.mean_file_path = p.mean_file_path;
		path.label_file_path = p.label_file_path;
		return path;
	}

};

struct ModelFilePath
{
	//RegPara reg_para;
	CaffeFilePath caffe_file_path;
	string model_version;
	ModelFilePath(const string &deploy = "", const string &label = "", const string &trained = "", const string &mean = "", const string &set_model_version = "")
	{
		caffe_file_path.deploy_file_path = deploy;
		caffe_file_path.label_file_path = label;
		caffe_file_path.trained_file_path = trained;
		caffe_file_path.mean_file_path = mean;
		model_version = set_model_version;
	}
};


struct ModelFileEvent
{
	RegPara reg_para;
	CaffeFilePath caffe_file_path;
	ModelEventType event_type;
	string model_version;
	ModelFileEvent()
	{
		event_type = UNKOWNMODELEVENTTYPE;
		model_version = "unkown";
	}
};




const char* GetVasRegTypeName(RegType reg_type);
const char* GetVasRegModeName(RegMode reg_mode);
const char* GetVasWeatherName(Weather weather);
const char* GetVasRegSceneName(RegScene reg_scene);
const char* GetVasNigthtimeName(Nighttime nighttime);


const RegType &GetVasRegType(const string &reg_para_str);
const RegScene &GetVasRegScene(const string &reg_para_str);
const Weather& GetVasRegWeather(const string &reg_para_str);
const Nighttime& GetVasRegNighttime(const string &reg_para_str);

const RegType &GetVasRegType(int reg_type_code);
const RegScene &GetVasRegScene(int reg_scene_code);
const Weather& GetVasRegWeather(int weather_code);
const Nighttime& GetVasRegNighttime(int nighttime_code);


RegPara GetRegParaByModelVersion(const string &model_version);

ModelFilePath GetModelFilePathFromEvent(const ModelFileEvent &model_file_event);






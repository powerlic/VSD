# worker.thrift
# Dean Chen (csfreebird@gmail.com)
#
 /**
 * Thrift files can namespace, package, or prefix their output in various
 * target languages.
 */
namespace cpp com.cetc38.videodetect
namespace java com.cetc38.videodetect


enum SeviceStatusType
 {
  SERVICE_NO_SUCH_ID=0,
  SERVICE_NORMAL = 1,
  SERVICE_STREAM_CONNECTING = 2,
  SERVICE_STREAM_FAULT = 3,
  SERVICE_FILE_FAULT = 4,
  SERVICE_STREAM_DECODE_FAULT = 5,
  SERVICE_DELETE_SUCCESS = 6,
  //feedback
  SERVICE_START_SUCCESS=7,
  SERVICE_START_FALIED=8,
  SERVICE_DELETE_SUCESS=9,
  SERVICE_DELETE_FALIED=10,
}

struct DetectStatus
{
	1: required string serviceId,
	2: required SeviceStatusType streamStatus,
}

struct RegRect
{
	1: required i32 x,
	2: required i32 y,
	3: required i32 width,
	4: required i32 height,
}

struct DetectResult{
	1: required string serviceId,
	2: required i32 hitTime,
	3: required i32 height,
	4: required i32 width,
	5: required list<string> regTypes,
	6: required list<double> regScores,
	7: required list<RegRect> rects
	8: required string previewPicURL,
	9: required string videoURL,
} 

struct DetectType{
	1: required string regType,
	2: required double sensitivity,
}

struct DetectServiceConfiguration{
	1: required string serviceId,
	2: required string streamURL,
	3: required i32 streamType,
	4: required i32 decodeMode,
	5: required i32 frameWidth,
	6: required i32 frameHeight,
	7: required list<DetectType> detectType,
}


/**
 * Defining a removed class named WorkerManager
 */

service DetectService {
 
  /**
   * client calls ping method to make sure service process is active or dead
   */
   void addService(1: DetectServiceConfiguration serviceConfig),
   
   SeviceStatusType deleteService(1: string serviceId),

   
   /**
    * check service state of the specified serviceId
	*/
   SeviceStatusType checkService(1: string serviceId),

 	/**
    * get the max service num 
	*/
	i32 getMaxServiceNum(),

	/**
    * get all service_id on running
	*/
	list<string> getServices(),


 	/**
    * report the service status error
	*/
	void reportServiceStatus(1:DetectStatus detectStatus),

	/**
    * send the detect result
	*/
	void sendSeriveDetectResult(1:DetectResult detectResult)
}


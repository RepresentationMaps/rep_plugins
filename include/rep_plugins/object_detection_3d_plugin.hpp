#ifndef REP_PLUGINS_OBJECT_DETECTION_PLUGIN_H_
#define REP_PLUGINS_OBJECT_DETECTION_PLUGIN_H_

#include <fstream>
#include <algorithm>

#include <rep_plugins/rep_plugin_base.hpp>
#include <openvdb/openvdb.h>

#include <chatgpt_ros/ChattingRequest.h>
#include <rep_plugins/ResetContext.h>

#include <yaml-cpp/yaml.h>

#include <sensor_msgs/CameraInfo.h>
#include <vision_msgs/Detection3DArray.h>

namespace rep_plugins{
	class ObjectDetection3DPlugin: public rep_plugins::PluginBase{
		private:
			static constexpr float epsilon = 1e-4;
			
			ros::Subscriber object_detection_sub_;
			ros::ServiceClient chatgpt_client_;
			ros::ServiceClient chatgpt_reset_client_;
			ros::ServiceServer reset_context_server_;

			float habituation_parameter_;

			std::string classes_path_;
			vision_msgs::Detection3DArray detections_msg_;

			std::vector<std::string> classes_;
			std::vector<float> saliencies_;

			bool use_depth_;
			double cx_, cy_;
			double fx_, fy_;

			bool use_saliency_;

			void objectDetectionCallback(const vision_msgs::Detection3DArrayConstPtr& msg);

			bool resetContextCallback(rep_plugins::ResetContext::Request& req,
						   			  rep_plugins::ResetContext::Response& res);

		public:
			ObjectDetection3DPlugin(){}
			~ObjectDetection3DPlugin(){}
			void updateMap();
			void initialize(){
				chatgpt_client_ = n_.serviceClient<chatgpt_ros::ChattingRequest>("/chatting_request");
				chatgpt_reset_client_ = n_.serviceClient<chatgpt_ros::ChattingRequest>("/chatting_reset");

				use_saliency_ = true;

				n_.param<bool>("/rep_plugins/object_detection/use_depth", use_depth_, false);
				n_.param<std::string>("/rep_plugins/object_detection/classes_path", classes_path_, std::string());
				
				std::ifstream input_file(classes_path_);
			    if (input_file.is_open())
			    {
			    	std::string class_line;
			        while (std::getline(input_file, class_line)){
			            classes_.push_back(class_line);
			            saliencies_.push_back(0.0);
			        }
			        input_file.close();
			    }

			    chatgpt_ros::ChattingRequest chatting_request;
				chatting_request.request.request = std::string("context: \"A human is looking for their cup.\"\nobjects: [person, cup, cell phone, boat, tv, potted plant]");
				if(chatgpt_client_.call(chatting_request)){
					ROS_INFO_STREAM(chatting_request.response.answer);
					YAML::Node saliencies = YAML::Load(chatting_request.response.answer);
					for(auto it = saliencies.begin(); it != saliencies.end(); ++it){
						ROS_INFO_STREAM("Class: "<<it->first.as<std::string>()<<"\tsaliency: "<<it->second.as<double>());
						auto it_od = std::find(classes_.begin(), classes_.end(), it->first.as<std::string>());
						if (it_od != classes_.end()){
							saliencies_[int(it_od-classes_.begin())] = it->second.as<double>();
						}
					}
				}else{
					ROS_ERROR("ChatGPT node not reachable!");
					use_saliency_ = false;
				}

				object_detection_sub_ = n_.subscribe<vision_msgs::Detection3DArray>("/world/objects/detections3d", 1, &ObjectDetection3DPlugin::objectDetectionCallback, this);
				reset_context_server_ = n_.advertiseService("/rep_plugins/object_detection/reset_context", &ObjectDetection3DPlugin::resetContextCallback, this);
			}
	};
};
#endif
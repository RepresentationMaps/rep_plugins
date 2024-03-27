#include <pluginlib/class_list_macros.h>
#include <rep_plugins/rep_plugin_base.hpp>
#include <rep_plugins/object_detection_3d_plugin.hpp>
#include <ros/ros.h>

#include <geometry_msgs/Point.h>

namespace rep_plugins{
	void ObjectDetection3DPlugin::objectDetectionCallback(const vision_msgs::Detection3DArrayConstPtr& msg){
		if(!sensor_frame_ && !msg->header.frame_id.empty())
			sensor_frame_ = msg->header.frame_id;

		detections_msg_ = *msg;
	}

	bool ObjectDetection3DPlugin::resetContextCallback(rep_plugins::ResetContext::Request& req,
						   			  				 rep_plugins::ResetContext::Response& res){
		chatgpt_ros::ChattingRequest chatting_request;
		chatting_request.request.request = req.request;
		if(chatgpt_reset_client_.call(chatting_request)){
			std::fill(saliencies_.begin(), saliencies_.end(), 0.0f);

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
			ROS_ERROR("Ops, something went wrong trying to communicate with the ChatGPT node");
		}
		res.answer = chatting_request.response.answer;
		return true;
	}

	void ObjectDetection3DPlugin::updateMap(){
		local_attention_map_->clear();
		local_grid_results_.clear();

		if (sensor_frame_ && !(*sensor_frame_==ref_frame_)){
			transformAttentionMap();
		}

		auto detected_object_map = openvdb::Int32Grid::create();
		auto saliency_object_map = openvdb::FloatGrid::create();

		detected_object_map->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
		detected_object_map->transformPtr()->postTranslate(offset_);
		saliency_object_map->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
		saliency_object_map->transformPtr()->postTranslate(offset_);

		if(sensor_frame_){
			local_attention_map_->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
			for(auto& detected_object: detections_msg_.detections){
				float object_saliency = saliencies_[detected_object.results[0].id];
				int object_id = detected_object.results[0].id;
				if (object_saliency >= 0.0){
					geometry_msgs::Point object_position = detected_object.bbox.center.position;
					geometry_msgs::Vector3 object_size = detected_object.bbox.size;

					openvdb::Vec3d bb_corner_1(object_position.x-(object_size.x/2.d), object_position.y-(object_size.y/2.d), object_position.z-(object_size.z/2.d));
					openvdb::Vec3d bb_corner_2(object_position.x+(object_size.x/2.d), object_position.y+(object_size.y/2.d), object_position.z+(object_size.z/2.d));
					
					if (use_saliency_){
						fillBox(*saliency_object_map,
								bb_corner_1,
								bb_corner_2,
								object_saliency);
					}
					fillBox(*detected_object_map,
						    bb_corner_1,
						    bb_corner_2,
						    object_id);
				}	
			}
		}

		local_grid_results_.push_back(GridResult(saliency_object_map, {{"hasSaliencyLowerThan"}, {"hasSaliencyHigherThan"}}));
		local_grid_results_.push_back(GridResult(detected_object_map, {{"containsObject"}}));
	}
}

PLUGINLIB_EXPORT_CLASS(rep_plugins::ObjectDetection3DPlugin, rep_plugins::PluginBase)
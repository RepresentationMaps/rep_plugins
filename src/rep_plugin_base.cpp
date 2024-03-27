#include <rep_plugins/rep_plugin_base.hpp>
#include <geometry_msgs/TransformStamped.h>

#include <ros/ros.h>

#include <openvdb/tools/Interpolation.h>
#include <openvdb/util/NullInterrupter.h>

#include <thread>
#include <functional>
#include <type_traits>

namespace rep_plugins{
	
	PluginBase::PluginBase(const bool& threaded):
	threaded_(threaded)
	{
		voxel_size_ = 0.1;
		offset_ = openvdb::math::Vec3d(voxel_size_/2., voxel_size_/2., voxel_size_/2.);
		local_attention_map_ = openvdb::FloatGrid::create(); // To delete
		// const openvdb::math::Vec3d offset(voxel_size_/2., voxel_size_/2., voxel_size_/2.);
		initial_transform_ = openvdb::math::Transform::createLinearTransform(voxel_size_);
		initial_transform_->postTranslate(offset_);
	}
	PluginBase::PluginBase() : PluginBase(false) {}

	PluginBase::~PluginBase(){
		for (auto& agent_maps: maps_){
			for (auto& map: agent_maps.second){
				map.second.reset();
			}
		}
	}

	void PluginBase::addMap(const std::string& agent, const std::vector<std::string>& maps){
		if (maps_.find(agent) == maps_.end()){
			maps_.insert(std::pair<std::string, std::map<std::string, openvdb::GridBase::Ptr>>(agent, std::map<std::string, openvdb::GridBase::Ptr>()));
		}
		for (auto& map: maps){
			if (maps_per_type_.find(map) == maps_per_type_.end()){
				maps_per_type_.insert(std::pair<std::string, std::vector<openvdb::GridBase::Ptr>>(map, std::vector<openvdb::GridBase::Ptr>()));
			}
			openvdb::GridBase::Ptr grid;
			if (map == "saliency"){
				grid = openvdb::FloatGrid::create();
				maps_[agent].insert(std::pair<std::string, openvdb::GridBase::Ptr>("saliency", grid));
			}else if(map == "occupancy"){
				grid = openvdb::BoolGrid::create();
				maps_[agent].insert(std::pair<std::string, openvdb::GridBase::Ptr>("occupancy", grid));
			}else{
				continue;
			}
			maps_per_type_[map].push_back(grid);
			maps_[agent][map]->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
			maps_[agent][map]->transformPtr()->postTranslate(offset_);
		}
	}

	openvdb::Coord PluginBase::localToGlobalTransform(const openvdb::Vec3d& current_coord){
		// This function is meant to transform into grid coordinates
		if (global_transform_){
			openvdb::Vec4d h_current_coord(current_coord[0], current_coord[1], current_coord[2], 1.0);
			h_current_coord = (*global_transform_)*h_current_coord;
			return openvdb::Coord(static_cast<int>(std::round(h_current_coord[0])),
					   		  	  static_cast<int>(std::round(h_current_coord[1])),
					   		  	  static_cast<int>(std::round(h_current_coord[2])));
		}
	}

	void PluginBase::initializeMaps(const std::vector<std::string>& agents,
									const std::vector<std::string>& maps){
		for (auto& agent: agents){
			//addMap(agent, maps);
		}
	}

	void PluginBase::transformAttentionMap(){
		geometry_msgs::TransformStamped transform;
		try{
			transform = tf_buffer_->lookupTransform(*sensor_frame_, ref_frame_, ros::Time(0));
			auto rotation = transform.transform.rotation;
			auto translation = transform.transform.translation;
			openvdb::Quatd quaternion(rotation.x, rotation.y, rotation.z, rotation.w);
			openvdb::Mat3d rot_matrix(quaternion);
			openvdb::Mat4d homogenous_matrix = openvdb::math::Mat4d::identity();
			homogenous_matrix.setMat3(rot_matrix);
			openvdb::Vec3d index_translation = initial_transform_->worldToIndex(openvdb::Vec3d(translation.x, translation.y, translation.z));
			index_translation = -quaternion.inverse().rotateVector(index_translation);
			openvdb::Vec3d h_index_translation(index_translation[0], index_translation[1], index_translation[2]);
			homogenous_matrix.setTranslation(h_index_translation);

			global_transform_ = homogenous_matrix;
		}catch (tf2::TransformException &ex) {
			ROS_WARN("%s",ex.what());
			return;
    	}
	}
	
};
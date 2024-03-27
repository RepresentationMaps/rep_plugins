#include <pluginlib/class_list_macros.h>
#include <rep_plugins/rep_plugin_base.hpp>
#include <rep_plugins/simple_faces_plugin.hpp>

#include <tf2_geometry_msgs/tf2_geometry_msgs.h>

#include <geometry_msgs/PointStamped.h>


namespace rep_plugins{
	void SimpleFacesPlugin::initialize(){
		std::string sensor_frame;
		n_.param<std::string>("/rep_plugins/faces_root_frame", sensor_frame, "camera_color_optical_frame");

		sensor_frame_ = sensor_frame;
		hri_listener_.setReferenceFrame(*sensor_frame_);
	}

	void SimpleFacesPlugin::updateMap(){
		local_attention_map_->clear();
		local_grid_results_.clear();

		if (sensor_frame_)
			transformAttentionMap();

		auto tracked_faces = hri_listener_.getFaces();

		// Now we implement a map-filling iteration;
		// There is not a proper way to do this,
		// but the idea is to:
		// 1. be sure to take into acconut all the maps
		// 2. if something can be optimized, do it
		
		// In this case, we want to fill the robot saliency map
		// as it is the only one we care about currently
		auto saliency_map = openvdb::FloatGrid::create();

		saliency_map->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
		saliency_map->transformPtr()->postTranslate(offset_);

		for(auto& tracked_face: tracked_faces){
			if (auto face_ptr = tracked_face.second.lock()){
				auto face_transform = face_ptr->transform();
				if (face_transform){
					if (!sensor_frame_){
						sensor_frame_ = face_transform->header.frame_id;
					}

					openvdb::Vec3d face_position = openvdb::Vec3d(face_transform->transform.translation.x,
					  		  				 					  face_transform->transform.translation.y,
					  		  				 					  face_transform->transform.translation.z);

					addSphere(*saliency_map,
					  		  face_position,
					  		  face_radius_,
					  		  voxel_size_,
					  		  std::bind(&intensity, face_position, std::placeholders::_1, face_radius_));
				}
			}
		}

		local_grid_results_.push_back(GridResult(saliency_map, {{"hasSaliencyLowerThan"}, {"hasSaliencyHigherThan"}}));
	} 
}

PLUGINLIB_EXPORT_CLASS(rep_plugins::SimpleFacesPlugin, rep_plugins::PluginBase)
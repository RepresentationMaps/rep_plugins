#include <pluginlib/class_list_macros.h>
#include <rep_plugins/rep_plugin_base.hpp>
#include <rep_plugins/agents_gaze_plugin.hpp>

#include <geometry_msgs/TransformStamped.h>

#include <functional>

namespace rep_plugins{
	void AgentsGazePlugin::updateMap(){
		local_attention_map_->clear();
		local_grid_results_.clear();

		if (sensor_frame_ && !(*sensor_frame_==ref_frame_))
			transformAttentionMap();

		auto robot_saliency_map = openvdb::FloatGrid::create();

		robot_saliency_map->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
		robot_saliency_map->transformPtr()->postTranslate(offset_);

		auto tracked_faces = hri_listener_.getFaces();
		try{
			for (auto& face_w_ptr: tracked_faces){
				if(auto face_ptr = face_w_ptr.second.lock()){
					if (face_ptr->gazeTransform()){
						auto agent_gaze_map = openvdb::BoolGrid::create();
						agent_gaze_map->setTransform(openvdb::math::Transform::createLinearTransform(voxel_size_));
						agent_gaze_map->transformPtr()->postTranslate(offset_);

						auto gaze_transform = *(face_ptr->gazeTransform());
						auto face_id = face_ptr->id();

						if(!sensor_frame_ && !gaze_transform.header.frame_id.empty())
							sensor_frame_ = gaze_transform.header.frame_id;

						openvdb::Quatd quaternion_rotation(gaze_transform.transform.rotation.x,
														   gaze_transform.transform.rotation.y,
														   gaze_transform.transform.rotation.z,
														   gaze_transform.transform.rotation.w);

						addCone(*robot_saliency_map,
								0.5,
								3.0,
								quaternion_rotation.rotateVector(openvdb::Vec3d(0.0, 0.0, 1.0)),
								std::bind(&intensity, std::placeholders::_1, 0.2, 0.1, 1e-4),
								openvdb::Vec3d(gaze_transform.transform.translation.x,
											   gaze_transform.transform.translation.y,
											   gaze_transform.transform.translation.z));
						addCone(*agent_gaze_map,
								0.5,
								3.0,
								quaternion_rotation.rotateVector(openvdb::Vec3d(0.0, 0.0, 1.0)),
								std::bind(&intensity, std::placeholders::_1, 0.2, 0.1, 1e-4),
								openvdb::Vec3d(gaze_transform.transform.translation.x,
											   gaze_transform.transform.translation.y,
											   gaze_transform.transform.translation.z));

						local_grid_results_.push_back(GridResult(agent_gaze_map, {{"isInFieldOfView", face_id}}));
					}
				}
			}

			local_grid_results_.push_back(GridResult(robot_saliency_map, {{"hasSaliencyLowerThan"}, {"hasSaliencyHigherThan"}}));
		}catch (tf2::ExtrapolationException ex){
			ROS_WARN("Extrapolation not possible");
		}
	}
}

PLUGINLIB_EXPORT_CLASS(rep_plugins::AgentsGazePlugin, rep_plugins::PluginBase)
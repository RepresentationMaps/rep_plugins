#ifndef REP_PLUGINS_SIMPLE_FACES_PLUGIN_H_
#define REP_PLUGINS_SIMPLE_FACES_PLUGIN_H_

#include <rep_plugins/rep_plugin_base.hpp>
#include <openvdb/openvdb.h>

#include <geometry_msgs/Point.h>

#include <hri/face.h>
#include <hri/hri.h>

namespace rep_plugins{
	class SimpleFacesPlugin : public rep_plugins::PluginBase{
		private:
			const double face_radius_ = 0.15;

			hri::HRIListener hri_listener_;

			static inline float intensity(const openvdb::Vec3d& centre, const openvdb::Vec3d& point, const float& radius){
				return static_cast<float>((std::abs(point[2]-centre[2])+radius)/(8*radius));
			}

		public:
			SimpleFacesPlugin(){}
			~SimpleFacesPlugin(){}
			void updateMap();
			void initialize(); // Here we initalize the HRIListener
	};
};
#endif
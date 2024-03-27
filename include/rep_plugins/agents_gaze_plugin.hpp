#ifndef REP_PLUGINS_AGENTS_GAZE_PLUGIN_H_
#define REP_PLUGINS_AGENTS_GAZE_PLUGIN_H_

#include <rep_plugins/rep_plugin_base.hpp>

#include <hri/face.h>
#include <hri/hri.h>

#include <math.h>

namespace rep_plugins{
	class AgentsGazePlugin : public rep_plugins::PluginBase{
		private:
			static constexpr float sigma_ = 0.1;
			static constexpr float amplitude_ = 0.2;
			static constexpr float epsilon_ = 1e-4;

			hri::HRIListener hri_listener_;

			static inline float intensity(openvdb::Vec3d point, float amplitude, float sigma, float epsilon){
				float theta = amplitude/2;
				float radius = static_cast<float>(point[0])*std::tan(theta)+epsilon;
				float distance = computeDistance(point, openvdb::Vec3d(point[0], 0.0, 0.0));
				float normalized_distance = distance/radius;
				//return 1/(sigma*std::sqrt(2*M_PI))*std::exp(-0.5*std::pow(normalized_distance/sigma, 2));
				return 0.3;
			}

		public:
			AgentsGazePlugin(){}
			~AgentsGazePlugin(){}

			void updateMap();
			void initialize(){
				hri_listener_.setReferenceFrame(ref_frame_);
			}
	};
};
#endif

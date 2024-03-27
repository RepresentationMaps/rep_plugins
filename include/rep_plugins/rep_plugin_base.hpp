#ifndef REP_PLUGINS_PLUGIN_BASE_H_
#define REP_PLUGINS_PLUGIN_BASE_H_

#include <openvdb/openvdb.h>
#include <openvdb/math/Math.h>
#include <openvdb/tools/GridTransformer.h>

#include <cmath>
#include <functional>
#include <memory>
#include <map>

#include <ros/ros.h>
#include <tf2_ros/transform_listener.h>

#include <boost/optional.hpp>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>

namespace rep_plugins
{
	struct GridResult{
		openvdb::GridBase::Ptr grid_ptr_;
		std::string frame; // For the moment this might be not used;
		std::string type; // We use this for aggregation purposes
		std::vector<std::vector<std::string>> data_format_; // examples: ["hasSaliencyLowerThan",""], ["hasSaliencyHigherThan", ""], ["containsObject", ""]

		GridResult(const openvdb::GridBase::Ptr& grid_ptr,
				   const std::vector<std::vector<std::string>>& data_format):
		grid_ptr_(grid_ptr),
		data_format_(data_format){}
	};

	class PluginBase
	{	protected:
			bool threaded_;

			std::string fixed_frame_ = "base_link";
			std::string memory_frame_; 

			float voxel_size_;
			openvdb::math::Vec3d offset_;
			openvdb::math::Transform::Ptr initial_transform_;

			float fov_;
			std::string ref_frame_;
			boost::optional<std::string> sensor_frame_;
			openvdb::FloatGrid::Ptr local_attention_map_;
			std::map<std::string, std::map<std::string, openvdb::GridBase::Ptr>> maps_;
			std::map<std::string, std::vector<openvdb::GridBase::Ptr>> maps_per_type_; // Same content as maps_, organized per type
			std::vector<GridResult> local_grid_results_;

			ros::NodeHandle n_;

			std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
			std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

			boost::optional<openvdb::Mat4d> global_transform_;
			openvdb::math::Transform full_transform_;

			// Testing solutions for faster transform
			boost::optional<openvdb::tools::GridTransformer> grid_transformer_;

			static inline float computeDistance(const openvdb::Vec3d& a, const openvdb::Vec3d& b){
				return std::sqrt(std::pow(a[0]-b[0], 2)+std::pow(a[1]-b[1], 2)+std::pow(a[2]-b[2], 2));
			}

			PluginBase();
			PluginBase(const bool& threaded);

			template<class T>
			inline std::shared_ptr<T> getAgentMapPtr(const std::string& agent, const std::string& map_name){
				return openvdb::gridPtrCast<T>(maps_[agent][map_name]);
			}

			openvdb::Coord localToGlobalTransform(const openvdb::Vec3d& current_coord);

			void transformAttentionMap(); // Transforms the pointclud applying sensor_frame_ -> ref_frame_

			virtual void habituationDecay(){}

			void addMap(const std::string& agent, const std::vector<std::string>& map);

			template<class T, class U>
			void const fillBox(T& grid,
							   openvdb::Vec3d bb_corner_1,
							   openvdb::Vec3d bb_corner_2,
							   const U value){
				constexpr bool check_base = std::is_base_of_v<openvdb::GridBase, T>;
				constexpr bool is_grid_base = std::is_same_v<openvdb::GridBase, T>;

				if constexpr(!check_base){
					std::cerr<<"Trying to operate over a non-tree object; aborting"<<std::endl;
					return;
				}

				if constexpr(is_grid_base){
					std::cerr<<"You are trying to pass a GridBase object; aborting"<<std::endl;
					return;
				}

				if constexpr(check_base && !is_grid_base){
					openvdb::Vec3d index_bb_1, index_bb_2;

					if (global_transform_){
						openvdb::Vec4d h_bb_corner_1(bb_corner_1[0], bb_corner_1[1], bb_corner_1[2], 1.0);
						openvdb::Vec4d h_bb_corner_2(bb_corner_2[0], bb_corner_2[1], bb_corner_2[2], 1.0);

						h_bb_corner_1 = (*global_transform_)*h_bb_corner_1;
						h_bb_corner_2 = (*global_transform_)*h_bb_corner_2;

						index_bb_1 = grid.worldToIndex(openvdb::Vec3d(h_bb_corner_1[0], h_bb_corner_1[1], h_bb_corner_1[2]));
						index_bb_2 = grid.worldToIndex(openvdb::Vec3d(h_bb_corner_2[0], h_bb_corner_2[1], h_bb_corner_2[2]));
					}else{
						index_bb_1 = grid.worldToIndex(bb_corner_1);
						index_bb_2 = grid.worldToIndex(bb_corner_2);
					}
					
					openvdb::math::CoordBBox cbbox(std::min(index_bb_1[0], index_bb_2[0]),
												   std::min(index_bb_1[1], index_bb_2[1]),
												   std::min(index_bb_1[2], index_bb_2[2]),
												   std::max(index_bb_1[0], index_bb_2[0]),
												   std::max(index_bb_1[1], index_bb_2[1]),
												   std::max(index_bb_1[2], index_bb_2[2]));
					
					grid.sparseFill(cbbox, value);

					grid.pruneGrid();
				}
			}

			template<class T, class U> // The template is the reason why the function is defined in the header file
			void pyramidCore(const int& i,
							 int& j,
							 int& k,
							 U& tree,
							 const float& theta_h,
							 const float& theta_v,
							 const openvdb::Vec3d& current_voxel_size,
							 const openvdb::math::Transform::Ptr linear_transform,
							 const openvdb::Vec3d& origin,
							 const openvdb::math::Quatd rotation_quat,
							 T intensity){
				/* Comments on design choice: this function was designed
				   to be used both in parallelized and non-parallelized
				   processes. One might say: why didn't you use the 3d
				   parallelization tools offered from tbb (tbb::blocked_range_3d)?
				   The problem is that the ranges are not blocked,
				   as the j and k limits depend on the value of i.
				*/
				constexpr bool base_check = std::is_base_of_v<openvdb::TreeBase, U>;
				constexpr bool same_type_check = std::is_same_v<openvdb::TreeBase, U>;

				if constexpr(!base_check){
					std::cerr<<"Trying to operate over a non-tree object; aborting"<<std::endl;
					return;
				}

				if constexpr(same_type_check){
					std::cerr<<"You are trying to pass a TreeBase object; aborting"<<std::endl;
					return;
				}

				if constexpr(base_check && !same_type_check){
					using VoxelValueType = typename U::ValueType;
					using GridAccessorType = typename openvdb::Grid<U>::Accessor;
					GridAccessorType acc(tree);

			        openvdb::Vec3d current_x = linear_transform->indexToWorld(openvdb::Coord(i, 0, 0));
					int j_start = -(static_cast<int>(std::tan(theta_h)*current_x.length()/current_voxel_size[1]));
					int k_start = -(static_cast<int>(std::tan(theta_v)*current_x.length()/current_voxel_size[2]));
			        for (j = j_start; j <= -j_start; ++j) {
			            for (k = k_start; k <= -k_start; ++k) {
			                openvdb::Vec3d current_y_coord = linear_transform->indexToWorld(openvdb::Coord(0.0, j, 0.0));
							float y_distance = computeDistance(origin, current_y_coord);
							openvdb::Vec3d current_z_coord = linear_transform->indexToWorld(openvdb::Coord(0.0, 0.0, k));
							float z_distance = computeDistance(origin, current_z_coord);
							if ((y_distance <= (std::tan(theta_h)*current_x.length()))
								&& (z_distance <= (std::tan(theta_v)*current_x.length()))){
								openvdb::Vec3d current_coord(i, j, k);
								current_coord = rotation_quat.rotateVector(current_coord);
								if (global_transform_){
									if constexpr (std::is_invocable_v<T, const openvdb::Vec3d&>)
										acc.setValue(localToGlobalTransform(current_coord), std::invoke(intensity, linear_transform->indexToWorld(openvdb::Coord(i, j, k))));
									else
										acc.setValue(localToGlobalTransform(current_coord), intensity); // Not explictly checking the type of the object
								}
								else{
									if constexpr (std::is_invocable_v<T, const openvdb::Vec3d&>)
										acc.setValue(openvdb::Coord(static_cast<int>(std::round(current_coord[0])),
																	 static_cast<int>(std::round(current_coord[1])),
																	 static_cast<int>(std::round(current_coord[2]))),
												 	 std::invoke(intensity, linear_transform->indexToWorld(openvdb::Coord(i, j, k))));
									else
										acc.setValue(openvdb::Coord(static_cast<int>(std::round(current_coord[0])),
																		 static_cast<int>(std::round(current_coord[1])),
																		 static_cast<int>(std::round(current_coord[2]))),
													 intensity);
								}
							}
			            }// end k
			        }// end j
			    }
			}

			template<class T, class U> // The template is the reason why the function is defined in the header file
			void coneCore(const int& i,
						  int& j,
						  int& k,
						  U& tree,
						  const float& theta,
						  const openvdb::Vec3d& current_voxel_size,
						  const openvdb::math::Transform::Ptr linear_transform,
						  const openvdb::Vec3d& origin_idx,
						  const openvdb::math::Quatd rotation_quat,
						  T intensity){
				/* Comments on design choice: see pyramidCore
				*/
				constexpr bool base_check = std::is_base_of_v<openvdb::TreeBase, U>;
				constexpr bool same_type_check = std::is_same_v<openvdb::TreeBase, U>;

				if constexpr(!base_check){
					std::cerr<<"Trying to operate over a non-tree object; aborting"<<std::endl;
					return;
				}

				if constexpr(same_type_check){
					std::cerr<<"You are trying to pass a TreeBase object; aborting"<<std::endl;
					return;
				}

				if constexpr(base_check && !same_type_check){
					using VoxelValueType = typename U::ValueType;
					using GridAccessorType = typename openvdb::Grid<U>::Accessor;
					GridAccessorType acc(tree);

					openvdb::Vec3d current_x = linear_transform->indexToWorld(openvdb::Coord(i, 0, 0));
					int j_start = -(static_cast<int>(std::tan(theta)*current_x.length()/current_voxel_size[1]));
					int k_start = -(static_cast<int>(std::tan(theta)*current_x.length()/current_voxel_size[2]));
			        for (j = j_start; j <= -j_start; ++j) {
			            for (k = k_start; k <= -k_start; ++k) {
			                openvdb::Vec3d current_yz_coord = linear_transform->indexToWorld(openvdb::Coord(0.0, j, k));
			                float distance = computeDistance(openvdb::Vec3d(0.0, 0.0, 0.0), current_yz_coord);
			                if (distance <= (std::tan(theta)*current_x.length())){
								openvdb::Vec3d current_coord(i, j, k);
								current_coord = rotation_quat.rotateVector(current_coord);
								current_coord = current_coord+origin_idx;
								VoxelValueType intensity_value;
								if constexpr (std::is_invocable_v<T, const openvdb::Vec3d&>)
										intensity_value = std::invoke(intensity, initial_transform_->indexToWorld(openvdb::Coord(i, j, k)));
									else
										intensity_value = intensity;
								if (global_transform_)
									acc.setValue(localToGlobalTransform(current_coord), intensity_value);
								else
									acc.setValue(openvdb::Coord(static_cast<int>(std::round(current_coord[0])),
																static_cast<int>(std::round(current_coord[1])),
																static_cast<int>(std::round(current_coord[2]))),
												 intensity_value);
							}
			            }// end k
			        }// end j
			    }
			}

			template<class T, class U>
			void addCone(U& grid,
						 const float& amplitude,
						 const float& length,
						 const openvdb::Vec3d& direction,
						 T intensity,
						 openvdb::Vec3d origin = openvdb::Vec3d(0.0, 0.0, 0.0)){
				if (length <= 0.0){
					ROS_WARN("The cone length has to be a strictly positive number.");
					return;
				}

				constexpr bool check_base = std::is_base_of_v<openvdb::GridBase, U>;
				constexpr bool is_grid_base = std::is_same_v<openvdb::GridBase, U>;

				if constexpr(!check_base){
					std::cerr<<"Trying to operate over a non-grid object; aborting"<<std::endl;
					return;
				}

				if constexpr(is_grid_base){
					std::cerr<<"You are trying to pass a GridBase object; aborting"<<std::endl;
					return;
				}

				if (check_base && !is_grid_base){
					using CurrentTreeType = typename U::TreeType;

					openvdb::math::Transform::Ptr linear_transform = initial_transform_;

					openvdb::Vec3d current_voxel_size = linear_transform->voxelSize();

					float theta = amplitude/2;
					float x_limit_index = length/current_voxel_size[0];
					
					openvdb::Vec3d x_axis(1.0, 0.0, 0.0);
					openvdb::Vec3d rotation_axis = x_axis.cross(direction);
					double angle = std::atan2(rotation_axis.length(), x_axis.dot(direction)); // Might be improved

					rotation_axis.normalize(); // Might be improved

					openvdb::math::Quatd rotation_quat(rotation_axis, angle);
					
					openvdb::Vec3d zero(0.0, 0.0, 0.0);
					openvdb::Vec3d origin_idx = grid.worldToIndex(origin);

					tbb::enumerable_thread_specific<CurrentTreeType> pool(grid.tree());

					// openvdb::util::NullInterrupter* mInterrupt = nullptr; // commented out as not used so far; might change in future release

			        auto kernel = [&](const tbb::blocked_range<int>& r) { // => r = tbb::blocked_range(0, x_limit_index, 1)
			            openvdb::Coord ijk;
			      		int &i = ijk[0], &j = ijk[1], &k = ijk[2];
			            CurrentTreeType &tree = (threaded_)?pool.local():grid.tree();
			            for (i = r.begin(); i != r.end(); ++i) {
			            	coneCore(i, j, k,
			            			 tree, theta,
			            			 current_voxel_size, linear_transform, origin_idx,
			            			 rotation_quat, intensity);
			            }// end i
			        };// kernel

			        ros::Time start_for = ros::Time::now();
			        auto iteration_range = tbb::blocked_range<int>(0, x_limit_index, 1);

			        if (threaded_){
			        	tbb::parallel_for(iteration_range, kernel);
				        using RangeT = tbb::blocked_range<typename tbb::enumerable_thread_specific<CurrentTreeType>::iterator>;
				        struct Op {
				            const bool mDelete;
				            CurrentTreeType *mTree;
				            Op(CurrentTreeType &tree) : mDelete(false), mTree(&tree) {}
				            Op(const Op& other, tbb::split) : mDelete(true), mTree(new CurrentTreeType(other.mTree->background())) {}
				            ~Op() { if (mDelete) delete mTree; }
				            void operator()(const RangeT &r) { for (auto i=r.begin(); i!=r.end(); ++i) this->merge(*i);}
				            void join(Op &other) { this->merge(*(other.mTree)); }
				            void merge(CurrentTreeType &tree) { mTree->merge(tree, openvdb::MERGE_ACTIVE_STATES); }
				        } op( grid.tree() );
				        tbb::parallel_reduce(RangeT(pool.begin(), pool.end(), 4), op);
			        }else{
			        	kernel(iteration_range);
			        }
			    }		
			}

			template<class T, class U> // The template is the reason why the function is defined in the header file
			void addPyramid(U& grid,
							const float& amplitude_h,
							const float& amplitude_v,
						 	const float& length,
						 	const float& voxel_size,
						 	const openvdb::Vec3d& direction,
						 	T intensity){
				/* This function builds a pyramid coming out of the origin frame
				   (that is, placed in the origin of the reference frame).
				*/
				if (length <= 0.0){
					ROS_WARN("The pyrdamid length has to be a strictly positive number.");
					return;
				}

				constexpr bool check_base = std::is_base_of_v<openvdb::GridBase, U>;
				constexpr bool is_grid_base = std::is_same_v<openvdb::GridBase, U>;

				if constexpr(!check_base){
					std::cerr<<"Trying to operate over a non-grid object; aborting"<<std::endl;
					return;
				}

				if constexpr(is_grid_base){
					std::cerr<<"You are trying to pass a GridBase object; aborting"<<std::endl;
					return;
				}

				if (check_base && !is_grid_base){
					using CurrentTreeType = typename U::TreeType;

					openvdb::math::Transform::Ptr linear_transform = initial_transform_;

					openvdb::Vec3d current_voxel_size = linear_transform->voxelSize();

					// Computing limits and angles
					float theta_h = amplitude_h/2;
					float theta_v = amplitude_v/2;
					float x_limit_index = length/current_voxel_size[0];
					
					openvdb::Vec3d x_axis(1.0, 0.0, 0.0);
					openvdb::Vec3d rotation_axis = x_axis.cross(direction);
					double angle = std::atan2(rotation_axis.length(), x_axis.dot(direction)); // Might be improved

					rotation_axis.normalize(); // Might be improved

					openvdb::math::Quatd rotation_quat(rotation_axis, angle);

					openvdb::Vec3d origin(0.0, 0.0, 0.0);

					tbb::enumerable_thread_specific<CurrentTreeType> pool(grid.tree());

			        auto kernel = [&](const tbb::blocked_range<int>& r) { // => r = tbb::blocked_range(0, x_limit_index, 1)
			            openvdb::Coord ijk;
			      		int &i = ijk[0], &j = ijk[1], &k = ijk[2];
			            CurrentTreeType &tree = (threaded_)?pool.local():grid.tree();
			            for (i = r.begin(); i != r.end(); ++i) {
			            	pyramidCore(i, j, k,
			            				tree, theta_h, theta_v,
			            				current_voxel_size, linear_transform, origin,
			            				rotation_quat, intensity);
			            }// end i
			        };// kernel

			        auto iteration_range = tbb::blocked_range<int>(0, x_limit_index, 1);

			        if (threaded_){
			        	tbb::parallel_for(iteration_range, kernel);
				        using RangeT = tbb::blocked_range<typename tbb::enumerable_thread_specific<CurrentTreeType>::iterator>;
				        struct Op {
				            const bool mDelete;
				            CurrentTreeType *mTree;
				            Op(CurrentTreeType &tree) : mDelete(false), mTree(&tree) {}
				            Op(const Op& other, tbb::split) : mDelete(true), mTree(new CurrentTreeType(other.mTree->background())) {}
				            ~Op() { if (mDelete) delete mTree; }
				            void operator()(const RangeT &r) { for (auto i=r.begin(); i!=r.end(); ++i) this->merge(*i);}
				            void join(Op &other) { this->merge(*(other.mTree)); }
				            void merge(CurrentTreeType &tree) { mTree->merge(tree, openvdb::MERGE_ACTIVE_STATES); }
				        } op( grid.tree() );
				        tbb::parallel_reduce(RangeT(pool.begin(), pool.end(), 4), op);
			        }else{
			        	kernel(iteration_range);
			        }
			    }
			}

			template<class T, class U>
			void sphereCore(const int& i,
							U& tree,
							const openvdb::Coord& ijk_min,
							const openvdb::Coord& ijk_max,
							const openvdb::math::Transform::Ptr linear_transform,
							const float& radius,
							const openvdb::Vec3d& centre,
							T intensity){
				constexpr bool base_check = std::is_base_of_v<openvdb::TreeBase, U>;
				constexpr bool same_type_check = std::is_same_v<openvdb::TreeBase, U>;

				if (!base_check){
					std::cerr<<"Trying to operate over a non-tree object; aborting"<<std::endl;
					return;
				}

				if (same_type_check){
					std::cerr<<"You are trying to pass a TreeBase object; aborting"<<std::endl;
					return;
				}

				int foo = 0; // debug

				if (base_check && !same_type_check){
					using VoxelValueType = typename U::ValueType;
					using GridAccessorType = typename openvdb::Grid<U>::Accessor;
					GridAccessorType acc(tree);

					openvdb::Coord ijk(i, ijk_min[1], ijk_min[2]);
					int& j = ijk[1];
					int& k = ijk[2]; 
					for(; j <= ijk_max[1]; ++j){
						k = ijk_min[2];
						for(; k <= ijk_max[2]; ++k){
							foo++;
							openvdb::Vec3d current_world_coord = linear_transform->indexToWorld(ijk);
							float distance = computeDistance(centre, current_world_coord);
							if (distance <= radius){
								VoxelValueType intensity_value;
								if constexpr (std::is_invocable_v<T, const openvdb::Vec3d&>)
									intensity_value = std::invoke(intensity, initial_transform_->indexToWorld(ijk));
								else
									intensity_value = intensity;
								if (global_transform_)
									acc.setValue(localToGlobalTransform(openvdb::Vec3d(i, j, k)), intensity_value);
								else
									acc.setValue(ijk, intensity_value);
							}
						} // end k
					} // end j
				}
			}

			template<class T, class U>
			void addSphere(U& grid,
						   const openvdb::Vec3d& centre,
						   const float& radius,
						   const float& voxel_size,
						   T intensity){
				constexpr bool check_base = std::is_base_of_v<openvdb::GridBase, U>;
				constexpr bool is_grid_base = std::is_same_v<openvdb::GridBase, U>;

				if constexpr(!check_base){
					std::cerr<<"You are trying to pass a non-grid type; aborting"<<std::endl;
					return;
				}

				if constexpr(is_grid_base){
					std::cerr<<"You are trying to pass a GridBase object; aborting"<<std::endl;
					return;
				}

				if (check_base && !is_grid_base){
					using CurrentTreeType = typename U::TreeType;

					// First of all, we get the centre index
					openvdb::math::Transform::Ptr linear_transform = initial_transform_;
					
					// Compute the index for the sphere bounds, as if it was a cube
					openvdb::Vec3d min_bound = centre - radius;
					openvdb::Vec3d min_bound_index_space = linear_transform->worldToIndex(min_bound);
					openvdb::Vec3d max_bound = centre + radius;
					openvdb::Vec3d max_bound_index_space = linear_transform->worldToIndex(max_bound);

					openvdb::Coord ijk_min(static_cast<int>(min_bound_index_space[0]), static_cast<int>(min_bound_index_space[1]), static_cast<int>(min_bound_index_space[2]));
					openvdb::Coord ijk_max(static_cast<int>(max_bound_index_space[0]), static_cast<int>(max_bound_index_space[1]), static_cast<int>(max_bound_index_space[2]));

					tbb::enumerable_thread_specific<CurrentTreeType> pool(grid.tree());

					// openvdb::util::NullInterrupter* mInterrupt = nullptr; // commented out as not used so far; might change in future release

			        auto kernel = [&](const tbb::blocked_range<int>& r) { // => r = tbb::blocked_range(0, x_limit_index, 1)
			        	openvdb::Coord ijk;
			        	int &i = ijk[0];

			            CurrentTreeType &tree = (threaded_)?pool.local():grid.tree();
			            for (i = r.begin(); i != r.end(); ++i) {
			            	sphereCore(i,
			            			   tree,
			            			   ijk_min,
			            			   ijk_max,
			            			   linear_transform, radius, centre,
			            			   intensity);
			            }// end i
			        };// kernel

			        auto iteration_range = tbb::blocked_range<int>(ijk_min[0], ijk_max[0], 1);

					if (threaded_){
			        	tbb::parallel_for(iteration_range, kernel);
				        using RangeT = tbb::blocked_range<typename tbb::enumerable_thread_specific<CurrentTreeType>::iterator>;
				        struct Op {
				            const bool mDelete;
				            CurrentTreeType *mTree;
				            Op(CurrentTreeType &tree) : mDelete(false), mTree(&tree) {}
				            Op(const Op& other, tbb::split) : mDelete(true), mTree(new CurrentTreeType(other.mTree->background())) {}
				            ~Op() { if (mDelete) delete mTree; }
				            void operator()(const RangeT &r) { for (auto i=r.begin(); i!=r.end(); ++i) this->merge(*i);}
				            void join(Op &other) { this->merge(*(other.mTree)); }
				            void merge(CurrentTreeType &tree) { mTree->merge(tree, openvdb::MERGE_ACTIVE_STATES); }
				        } op( grid.tree() );
				        tbb::parallel_reduce(RangeT(pool.begin(), pool.end(), 4), op);
			        }else{
			        	kernel(iteration_range);
			        }
			    }
			}

		public:
			virtual ~PluginBase();
			
			virtual void updateMap() = 0;
			void initializeBase(ros::NodeHandle& n, std::shared_ptr<tf2_ros::Buffer>& buffer){
				n_ = n;
				tf_buffer_ = buffer;
				n.param<bool>("/rep_plugins/threaded", threaded_, false);
			}
			void initializeMaps(const std::vector<std::string>& agents,
							    const std::vector<std::string>& maps);

			virtual void initialize(){};

			inline void updateFov(const float& fov){fov_ = fov;}
			inline void updateRefFrame(const std::string& ref_frame){ref_frame_ = ref_frame;}
			inline void updateFixedFrame(const std::string& fixed_frame){fixed_frame_ = fixed_frame;}
			inline void updateMemoryFrame(const std::string& memory_frame){memory_frame_ = memory_frame;}

			inline openvdb::FloatGrid::Ptr getLocalAttentionMap() {
				return local_attention_map_;}

			inline std::map<std::string, std::map<std::string, openvdb::GridBase::Ptr>> getAllMaps(){
				return maps_;
			}

			inline std::map<std::string, std::vector<openvdb::GridBase::Ptr>> getMapsPerType(){
				return maps_per_type_;
			}

			inline std::map<std::string, openvdb::GridBase::Ptr> getAgentMaps(const std::string& agent){
				auto agent_maps = maps_.find(agent);
				if (agent_maps == maps_.end()){
					return std::map<std::string, openvdb::GridBase::Ptr>();
				}
				return agent_maps->second;
			}

			inline std::vector<GridResult> const getExtendedGrids(){
				return local_grid_results_;
			}
	};
};
#endif
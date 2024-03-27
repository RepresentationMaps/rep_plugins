#include <ros/ros.h>
#include <string>

#include <rep_plugins/ResetContext.h>
#include <std_msgs/String.h>

void context_cb(const std_msgs::StringConstPtr& msg){
	rep_plugins::ResetContext request;
	request.request.request = msg->data;

	if(ros::service::call("/rep_plugins/object_detection/reset_context", request)){
		ROS_INFO_STREAM("Correctly handled the context resetting from bag file");
	}
}

int main(int argc, char** argv){
	ros::init(argc, argv, "context_republisher");

	ros::NodeHandle nh("~");

	ros::Subscriber sub = nh.subscribe<std_msgs::String>("/context_info", 1, context_cb);

	ros::spin();
}
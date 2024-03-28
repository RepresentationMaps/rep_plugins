# rep_plugins
Tools for Representation Maps Sources implementation as pluginlib plugins

> [!WARNING]  
> This is a research project and an official release is still
> under development. 

This package provides:
- the basic structures to develop `RMS` as `pluginlib` plugins;
- a group of ready-made `RMSs`.

## Available plugins

- **SimpleFaces plugin**: generating `RMs` based on the detected faces;
- **3DObjectDetection plugin**: generating `RMs` based on the detected objects (requires information as `vision_msgs/BoundingBox3D`);
- **AgentsGaze plugin**: generating `RMs` based on the detected gazes;

## Dependencies
- `ROS noetic`
- `openvdb >= 6.2.1` 
- `tbb >= 2020.1`
- `pcl >= 1.10.0`
- `yaml-cpp >= 0.6.2`
- `boost >= 1.71.0`
- `libhri`
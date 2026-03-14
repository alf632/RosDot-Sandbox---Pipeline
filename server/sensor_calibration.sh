#!/bin/bash

# Create a directory to hold the output configuration files
mkdir -p tf_configs

echo "Starting AprilTag auto-calibration routine..."

for i in {1..4}; do
    CAM="cam${i}"
    ENV_FILE="tf_configs/${CAM}_tf.env"
    
    echo "========================================="
    echo "Calibrating ${CAM}..."

    # 1. Start the AprilTag node for this camera in the background
    # Note: You will need a standard tags.yaml file so the node knows the tag sizes and IDs.
    docker run -d --rm --name apriltag_${CAM} --network host \
        -v $(pwd)/tags.yaml:/tmp/tags.yaml \
        ros:humble-ros-base \
        bash -c "source /opt/ros/humble/setup.bash && ros2 run apriltag_ros apriltag_node --ros-args -r image_rect:=/${CAM}/infra1/image_rect_raw -r camera_info:=/${CAM}/infra1/camera_info --params-file /tmp/tags.yaml"

    echo "Waiting 5 seconds for ${CAM} to detect the tag..."
    sleep 5 

    # 2. Grab the transform using tf2_echo and a 5-second timeout
    echo "Extracting transform from sandbox_origin to ${CAM}_link..."
    TF_OUTPUT=$(docker run --rm --network host ros:humble-ros-base bash -c "source /opt/ros/humble/setup.bash && timeout 5 ros2 run tf2_ros tf2_echo sandbox_origin ${CAM}_link" 2>/dev/null)

    # 3. Check if we actually saw the tag
    if [ -z "$TF_OUTPUT" ]; then
        echo "ERROR: Could not get transform for ${CAM}. Is the AprilTag visible and the IR emitter off?"
    else
        # 4. Parse the tf2_echo output
        # We remove brackets and commas, then use awk to grab the exact columns
        X=$(echo "$TF_OUTPUT" | grep "Translation:" | head -n 1 | tr -d '[],' | awk '{print $3}')
        Y=$(echo "$TF_OUTPUT" | grep "Translation:" | head -n 1 | tr -d '[],' | awk '{print $4}')
        Z=$(echo "$TF_OUTPUT" | grep "Translation:" | head -n 1 | tr -d '[],' | awk '{print $5}')
        
        ROLL=$(echo "$TF_OUTPUT" | grep "RPY (radian)" | head -n 1 | tr -d '[],' | awk '{print $6}')
        PITCH=$(echo "$TF_OUTPUT" | grep "RPY (radian)" | head -n 1 | tr -d '[],' | awk '{print $7}')
        YAW=$(echo "$TF_OUTPUT" | grep "RPY (radian)" | head -n 1 | tr -d '[],' | awk '{print $8}')

        # 5. Write out the environment file
        echo "TF_X=${X}" > $ENV_FILE
        echo "TF_Y=${Y}" >> $ENV_FILE
        echo "TF_Z=${Z}" >> $ENV_FILE
        echo "TF_ROLL=${ROLL}" >> $ENV_FILE
        echo "TF_PITCH=${PITCH}" >> $ENV_FILE
        echo "TF_YAW=${YAW}" >> $ENV_FILE

        echo "Success! Wrote coordinates to ${ENV_FILE}"
    fi

    # 6. Kill the temporary AprilTag container
    docker stop apriltag_${CAM} > /dev/null
done

echo "========================================="
echo "Calibration complete! You can now start the mapping pipeline."

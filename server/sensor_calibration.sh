#!/bin/bash

# Create a directory to hold the output configuration files
mkdir -p tf_configs

echo "Preparing Container"

docker images | grep -q apriltagcontainer || docker build -f apriltagDockerfile -t apriltagcontainer .

echo "Starting AprilTag auto-calibration routine..."

for i in {1..4}; do
    CAM="cam${i}"
    ENV_FILE="tf_configs/${CAM}_tf.env"
    
    echo "========================================="
    echo "Calibrating ${CAM}..."

    # 1. Start the AprilTag node for this camera in the background
    # Note: You will need a standard tags.yaml file so the node knows the tag sizes and IDs.
    docker run -d --rm --name apriltag_${CAM} --network host --ipc=host \
        -v $(pwd)/tags.yaml:/tmp/tags.yaml -e CAM=${CAM} -e ROS_DOMAIN_ID=42 \
        apriltagcontainer

    echo "Waiting 5 seconds for ${CAM} to detect the tag..."
    sleep 5 

    # 2. Grab the transform using tf2_echo and a 5-second timeout
    echo "Extracting transform from sandbox_origin to ${CAM}_link..."
    TF_OUTPUT=$(docker run --rm --network host --ipc=host -e ROS_DOMAIN_ID=42 ros:humble-ros-base \
      bash -c 'source /opt/ros/humble/setup.bash && \
      timeout 30 ros2 run tf2_ros tf2_echo sandbox_origin "$1" | \
      while IFS= read -r line; do \
        echo "$line"; \
        [[ "$line" == *"Quaternion (xyzw)"* ]] && break; \
      done' _ "${CAM}_link" 2>/dev/null)

    # 3. Check if we actually saw the tag
    if [ -z "$TF_OUTPUT" ]; then
        echo "ERROR: Could not get transform for ${CAM}. Is the AprilTag visible and the IR emitter off?"
    else
        # 4. Parse the tf2_echo output
        # We remove brackets and commas, then use awk to grab the exact columns
        X=$(echo "$TF_OUTPUT" | grep "Translation:" | head -n 1 | tr -d '[],' | awk '{print $3}')
        Y=$(echo "$TF_OUTPUT" | grep "Translation:" | head -n 1 | tr -d '[],' | awk '{print $4}')
        Z=$(echo "$TF_OUTPUT" | grep "Translation:" | head -n 1 | tr -d '[],' | awk '{print $5}')
        
        QX=$(echo "$TF_OUTPUT" | grep "Quaternion (xyzw)" | head -n 1 | tr -d '[],' | awk '{print $6}')
        QY=$(echo "$TF_OUTPUT" | grep "Quaternion (xyzw)" | head -n 1 | tr -d '[],' | awk '{print $7}')
        QZ=$(echo "$TF_OUTPUT" | grep "Quaternion (xyzw)" | head -n 1 | tr -d '[],' | awk '{print $8}')
	QW=$(echo "$TF_OUTPUT" | grep "Quaternion (xyzw)" | head -n 1 | tr -d '[],' | awk '{print $9}')

        # 5. Write out the environment file
        echo "TF_X=${X}" > $ENV_FILE
        echo "TF_Y=${Y}" >> $ENV_FILE
        echo "TF_Z=${Z}" >> $ENV_FILE
        echo "TF_QX=${QX}" >> $ENV_FILE
        echo "TF_QY=${QY}" >> $ENV_FILE
        echo "TF_QZ=${QZ}" >> $ENV_FILE
	echo "TF_QW=${QW}" >> $ENV_FILE

        echo "Success! Wrote coordinates to ${ENV_FILE}"
    fi

    # 6. Kill the temporary AprilTag container
    docker stop apriltag_${CAM} > /dev/null
done

echo "========================================="
echo "Calibration complete! You can now start the mapping pipeline."

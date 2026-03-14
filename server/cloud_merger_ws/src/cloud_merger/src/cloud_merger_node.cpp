#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/passthrough.h>

using std::placeholders::_1;
using std::placeholders::_2;
using std::placeholders::_3;
using std::placeholders::_4;

class CloudMergerNode : public rclcpp::Node {
public:
    CloudMergerNode() : Node("cloud_merger_node") {
        // Initialize TF2 Buffer and Listener
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // Setup the publisher for the final merged cloud
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("merged_filtered_cloud", 10);

        // Setup the 4 subscribers to the uncompressed depth clouds
        // (Assuming depth_image_proc is publishing them here)
        sub1_.subscribe(this, "/cam1/depth/points");
        sub2_.subscribe(this, "/cam2/depth/points");
        sub3_.subscribe(this, "/cam3/depth/points");
        sub4_.subscribe(this, "/cam4/depth/points");

        // Setup the approximate time synchronizer
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(10), sub1_, sub2_, sub3_, sub4_);
        sync_->registerCallback(std::bind(&CloudMergerNode::cloud_callback, this, _1, _2, _3, _4));

        RCLCPP_INFO(this->get_logger(), "Cloud Merger Node Started. Waiting for synchronized point clouds...");
    }

private:
    typedef message_filters::sync_policies::ApproximateTime<
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2,
        sensor_msgs::msg::PointCloud2, sensor_msgs::msg::PointCloud2> SyncPolicy;

    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub1_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub2_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub3_;
    message_filters::Subscriber<sensor_msgs::msg::PointCloud2> sub4_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;

    void cloud_callback(
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg1,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg2,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg3,
        const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg4) 
    {
        std::string target_frame = "sandbox_origin";
        pcl::PointCloud<pcl::PointXYZ>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        // Helper lambda to transform and append a cloud
        auto process_cloud = [&](const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg) {
            sensor_msgs::msg::PointCloud2 transformed_msg;
            try {
                // Transform cloud from camera frame to sandbox_origin
                transformed_msg = tf_buffer_->transform(*msg, target_frame, tf2::durationFromSec(0.1));
                
                pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
                pcl::fromROSMsg(transformed_msg, pcl_cloud);
                *merged_cloud += pcl_cloud; // Concatenate
            } catch (tf2::TransformException &ex) {
                RCLCPP_WARN(this->get_logger(), "Could not transform cloud: %s", ex.what());
            }
        };

        process_cloud(msg1);
        process_cloud(msg2);
        process_cloud(msg3);
        process_cloud(msg4);

        // Apply PassThrough Filter to remove ceilings and people's torsos/heads
        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PassThrough<pcl::PointXYZ> pass;
        pass.setInputCloud(merged_cloud);
        pass.setFilterFieldName("z");
        // Keep everything from -1.0m (deep sand) up to 0.5m (hands/forearms)
        pass.setFilterLimits(-1.0, 0.5); 
        pass.filter(*filtered_cloud);

        // Convert back to ROS message and publish
        sensor_msgs::msg::PointCloud2 output_msg;
        pcl::toROSMsg(*filtered_cloud, output_msg);
        output_msg.header.stamp = msg1->header.stamp; // Inherit timestamp from cam1
        output_msg.header.frame_id = target_frame;
        
        publisher_->publish(output_msg);
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<CloudMergerNode>());
    rclcpp::shutdown();
    return 0;
}

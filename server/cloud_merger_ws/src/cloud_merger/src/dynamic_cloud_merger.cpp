#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

using std::placeholders::_1;

class DynamicCloudMerger : public rclcpp::Node
{
public:
    DynamicCloudMerger() : Node("dynamic_cloud_merger")
    {
        // 1. Initialize TF2 Buffer and Listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 2. Declare parameter for dynamic camera topics
        this->declare_parameter<std::vector<std::string>>("camera_topics", 
            {"/cam1/pointcloud", "/cam2/pointcloud", "/cam3/pointcloud", "/cam4/pointcloud"});

	// Declare the timer interval parameter (Default: 33ms)
        this->declare_parameter<int>("timer_interval_ms", 33);
        
        std::vector<std::string> topics = this->get_parameter("camera_topics").as_string_array();
	int interval_ms = this->get_parameter("timer_interval_ms").as_int();

	RCLCPP_INFO(this->get_logger(), "Merger starting with interval: %d ms", interval_ms);

	// 3. Create dynamic subscribers using C++ lambdas
        for (const auto& topic : topics) {
            auto sub = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                topic, rclcpp::SensorDataQoS(),
                [this, topic](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                    // Just cache the latest message for this specific topic
                    std::lock_guard<std::mutex> lock(cache_mutex_);
                    latest_clouds_[topic] = msg;
                }
            );
            subscribers_.push_back(sub);
            RCLCPP_INFO(this->get_logger(), "Subscribed to: %s", topic.c_str());
        }

	// 1. Define a custom QoS specifically tuned for real-time Point Clouds
        rclcpp::QoS pub_qos = rclcpp::SensorDataQoS();
        pub_qos.keep_last(1);

        // 2. Set up Publisher Options to listen for QoS mismatch events
        rclcpp::PublisherOptions pub_options;
        pub_options.event_callbacks.incompatible_qos_callback =
            [this](rclcpp::QOSOfferedIncompatibleQoSInfo & info) {
                RCLCPP_ERROR(this->get_logger(),
                    "⚠️ QoS Mismatch! A node tried to subscribe, but its QoS settings are incompatible. Policy kind ID: %d",
                    info.last_policy_kind);
            };

        // 3. Create the publisher
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/sandbox/merged_cloud", pub_qos, pub_options);

        RCLCPP_INFO(this->get_logger(), "Publishing to: /sandbox/merged_cloud");

        // 5. Timer to run the merge operation at ~30Hz (33ms)
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(interval_ms),
            std::bind(&DynamicCloudMerger::timer_callback, this)
        );
    }

private:
    void timer_callback()
    {
	if (publisher_->get_subscription_count() == 0) {
            return;
        }

        std::lock_guard<std::mutex> lock(cache_mutex_);
        
        pcl::PointCloud<pcl::PointXYZ> final_merged_cloud;
        bool has_data = false;
        
        // Grab the current time for the output header
        rclcpp::Time current_time = this->now();

        // Loop through whatever is currently in our cache
        for (auto& [topic, msg] : latest_clouds_) {
            if (!msg) continue; // Skip if no data received yet or it was cleared

            sensor_msgs::msg::PointCloud2 transformed_msg;
            try {
                // Find the TF from the camera to the sandbox_origin
                auto transform = tf_buffer_->lookupTransform(
                    "sandbox_origin", msg->header.frame_id, tf2::TimePointZero);
                
                // Transform the ROS PointCloud2
                tf2::doTransform(*msg, transformed_msg, transform);
                
                // Convert to PCL and append to our master cloud
                pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
                pcl::fromROSMsg(transformed_msg, pcl_cloud);
                final_merged_cloud += pcl_cloud;
                
                has_data = true;
                
            } catch (const tf2::TransformException & ex) {
                RCLCPP_WARN(this->get_logger(), "TF Error on %s: %s", topic.c_str(), ex.what());
            }

            // Clear the pointer so we don't re-use stale data if the camera disconnects
            msg.reset(); 
        }

        // If we successfully processed at least one camera, publish the result!
        if (has_data) {
            sensor_msgs::msg::PointCloud2 output_msg;
            pcl::toROSMsg(final_merged_cloud, output_msg);
            
            output_msg.header.frame_id = "sandbox_origin";
            output_msg.header.stamp = current_time;
            
            publisher_->publish(output_msg);
        }
    }

    std::vector<rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subscribers_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::TimerBase::SharedPtr timer_;
    
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    std::map<std::string, sensor_msgs::msg::PointCloud2::SharedPtr> latest_clouds_;
    std::mutex cache_mutex_;
};

int main(int argc, char * argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DynamicCloudMerger>());
    rclcpp::shutdown();
    return 0;
}

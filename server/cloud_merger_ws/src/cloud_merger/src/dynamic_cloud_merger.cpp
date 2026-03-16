#include <memory>
#include <string>
#include <vector>
#include <map>
#include <mutex>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/transforms.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <opencv2/opencv.hpp>

class DynamicCloudMerger : public rclcpp::Node {
public:
    DynamicCloudMerger() : Node("dynamic_cloud_merger") {
        // --- Parameters ---
        this->declare_parameter("sandbox_width", 4.0);
        this->declare_parameter("sandbox_length", 6.0);
        this->declare_parameter("resolution_per_meter", 100);
        this->declare_parameter("max_step_size", 0.02);
        this->declare_parameter("camera_topics", std::vector<std::string>{});
        this->declare_parameter("published_topic", "sandbox/heightmap");
        this->declare_parameter("framerate", 30.0);
        this->declare_parameter("log_interval", 10.0);

        // --- TF & State Setup ---
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        int w = get_parameter("sandbox_width").as_double() * get_parameter("resolution_per_meter").as_int();
        int l = get_parameter("sandbox_length").as_double() * get_parameter("resolution_per_meter").as_int();
        current_heightmap_ = cv::Mat::zeros(l, w, CV_32FC1);

        // --- Subscriptions ---
        auto topics = get_parameter("camera_topics").as_string_array();
        for (const auto& topic : topics) {
            auto sub = create_subscription<sensor_msgs::msg::PointCloud2>(
                topic, 
                rclcpp::SensorDataQoS(), // Best Effort
                [this, topic](const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    latest_clouds_[topic] = msg;
                    cloud_counts_[topic]++;
                });
            subs_[topic] = sub;
            cloud_counts_[topic] = 0;
        }

        // --- Publisher ---
        pub_ = create_publisher<sensor_msgs::msg::Image>(
            get_parameter("published_topic").as_string(), 10);

        // --- Timers ---
        double interval_ms = 1000.0 / get_parameter("framerate").as_double();
        timer_ = create_wall_timer(
            std::chrono::duration<double, std::milli>(interval_ms),
            std::bind(&DynamicCloudMerger::process_and_publish, this));

        double log_s = get_parameter("log_interval").as_double();
        diag_timer_ = create_wall_timer(
            std::chrono::duration<double>(log_s),
            std::bind(&DynamicCloudMerger::log_diagnostics, this));
        
        last_sub_count_ = 0;
    }

private:
    void process_and_publish() {
        size_t current_subs = pub_->get_subscription_count();
        
        // Notify on connection changes
        if (current_subs != last_sub_count_) {
            if (current_subs > last_sub_count_) {
                RCLCPP_INFO(this->get_logger(), "New subscriber detected on %s", pub_->get_topic_name());
            } else {
                RCLCPP_INFO(this->get_logger(), "Subscriber disconnected from %s", pub_->get_topic_name());
            }
            last_sub_count_ = current_subs;
        }

        if (current_subs == 0) return;

        // Perform QoS Compatibility Check
        if (last_sub_count_ > 0 && subs_.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "Publisher active but no camera subscriptions configured!");
        }

        // --- Projection Logic ---
        int res = get_parameter("resolution_per_meter").as_int();
        double width = get_parameter("sandbox_width").as_double();
        double length = get_parameter("sandbox_length").as_double();
        float max_step = get_parameter("max_step_size").as_double();

        cv::Mat frame_buffer = cv::Mat::zeros(current_heightmap_.size(), CV_32FC1);
        cv::Mat count_buffer = cv::Mat::zeros(current_heightmap_.size(), CV_32FC1);

        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            if (latest_clouds_.empty()) return;

            for (auto const& [topic, msg] : latest_clouds_) {
                pcl::PointCloud<pcl::PointXYZ> pcl_cloud;
                pcl::fromROSMsg(*msg, pcl_cloud);

                pcl::PointCloud<pcl::PointXYZ> transformed_cloud;
                try {
                    auto tf = tf_buffer_->lookupTransform("sandbox_origin", msg->header.frame_id, tf2::TimePointZero);
                    pcl_ros::transformPointCloud(pcl_cloud, transformed_cloud, tf);
                } catch (tf2::TransformException &ex) {
                    continue; // Skip this cloud if TF isn't ready
                }

                for (const auto& pt : transformed_cloud.points) {
                    // Map 3D Sandbox Coords to 2D Image Coords
                    int x = (pt.x + (width / 2.0)) * res;
                    int y = (pt.y + (length / 2.0)) * res;

                    if (x >= 0 && x < frame_buffer.cols && y >= 0 && y < frame_buffer.rows) {
                        frame_buffer.at<float>(y, x) += pt.z;
                        count_buffer.at<float>(y, x) += 1.0f;
                    }
                }
            }
        }

        // Compute average height per pixel
        cv::Mat mean_height;
        cv::divide(frame_buffer, count_buffer, mean_height);

        // --- Post-Processing ---
        // 1. Create mask for inpainting (where count is 0)
        cv::Mat mask;
        cv::compare(count_buffer, 0, mask, cv::CMP_EQ);
        mask.convertTo(mask, CV_8U);

        // 2. Inpaint and Smooth
        cv::Mat processed;
        cv::inpaint(mean_height, mask, processed, 5, cv::INPAINT_NS);
        cv::GaussianBlur(processed, processed, cv::Size(7, 7), 1.5);

        // 3. Temporal Interpolation (the "Anti-Jump" logic)
        cv::Mat delta = processed - current_heightmap_;
        cv::Mat clamped_delta;
        // Clamp the change per frame to max_step_size
        cv::threshold(cv::abs(delta), clamped_delta, max_step, max_step, cv::THRESH_TRUNC);
        
        // Apply the sign of the original delta to the clamped magnitude
        cv::Mat sign;
        cv::divide(delta, cv::abs(delta) + 1e-6, sign); 
        current_heightmap_ += clamped_delta.mul(sign);

        // --- Publish ---
        auto out_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", current_heightmap_).toImageMsg();
        out_msg->header.stamp = this->now();
        out_msg->header.frame_id = "sandbox_origin";
        pub_->publish(*out_msg);
    }

    void log_diagnostics() {
        double interval = get_parameter("log_interval").as_double();
        RCLCPP_INFO(this->get_logger(), "--- Periodic Report (Last %.1f s) ---", interval);
        
        if (subs_.empty()) {
            RCLCPP_ERROR(this->get_logger(), " No camera topics subscribed! Check 'camera_topics' parameter.");
        }

        for (auto& [topic, count] : cloud_counts_) {
            if (count == 0) {
                RCLCPP_WARN(this->get_logger(), " Topic [%s]: 0 messages! Possible QoS mismatch or camera disconnect.", topic.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), " Topic [%s]: %d msgs (Avg %.1f Hz)", 
                    topic.c_str(), count, (double)count / interval);
            }
            count = 0; // Reset for next interval
        }
    }

    // Members
    std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr> subs_;
    std::map<std::string, sensor_msgs::msg::PointCloud2::SharedPtr> latest_clouds_;
    std::map<std::string, int> cloud_counts_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp::TimerBase::SharedPtr diag_timer_;
    
    cv::Mat current_heightmap_;
    size_t last_sub_count_;
    std::mutex data_mutex_;
};


int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<DynamicCloudMerger>();

  // Using a MultiThreadedExecutor is recommended if you have 4x cameras
  // to ensure callbacks don't block the processing timer.
  rclcpp::executors::MultiThreadedExecutor executor;
  executor.add_node(node);
  executor.spin();

  rclcpp::shutdown();
  return 0;
}

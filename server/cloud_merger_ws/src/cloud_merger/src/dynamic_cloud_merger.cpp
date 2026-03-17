#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <shared_mutex>
#include <map>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "cv_bridge/cv_bridge.hpp"

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

struct ThreadBuffer {
    cv::Mat accum;
    cv::Mat count;
};

static thread_local std::shared_ptr<ThreadBuffer> local_buf = nullptr;

class DynamicCloudMerger : public rclcpp::Node {
public:
    DynamicCloudMerger() : Node("dynamic_cloud_merger") {
        // --- Parameters ---
        this->declare_parameter("sandbox_width", 4.0);
        this->declare_parameter("sandbox_length", 6.0);
        this->declare_parameter("resolution_per_meter", 100);
        this->declare_parameter("max_step_size", 0.02);
        this->declare_parameter("camera_prefixes", std::vector<std::string>{"/cam1", "/cam2", "/cam3", "/cam4"});
        this->declare_parameter("published_topic", "sandbox/heightmap");
        this->declare_parameter("framerate", 30.0);
        this->declare_parameter("log_interval", 10.0);

        // --- Hardware/TF Setup ---
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        int w = get_parameter("sandbox_width").as_double() * get_parameter("resolution_per_meter").as_int();
        int l = get_parameter("sandbox_length").as_double() * get_parameter("resolution_per_meter").as_int();
        current_heightmap_ = cv::Mat::zeros(l, w, CV_32FC1);

        auto prefixes = get_parameter("camera_prefixes").as_string_array();
        for (const auto& pre : prefixes) {
            setup_camera(pre);
        }

        pub_ = create_publisher<sensor_msgs::msg::Image>(get_parameter("published_topic").as_string(), 10);

        // --- Timers ---
        double interval_ms = 1000.0 / get_parameter("framerate").as_double();
        timer_ = create_wall_timer(std::chrono::duration<double, std::milli>(interval_ms),
                                   std::bind(&DynamicCloudMerger::process_and_publish, this));

        double log_s = get_parameter("log_interval").as_double();
        diag_timer_ = create_wall_timer(std::chrono::duration<double>(log_s),
                                        std::bind(&DynamicCloudMerger::log_diagnostics, this));
        
        last_sub_count_ = 0;
    }

private:
    void setup_camera(const std::string& pre) {
        std::string info_topic = pre + "/depth/camera_info";
        std::string depth_topic = pre + "/depth/image_rect_raw";

        RCLCPP_INFO(this->get_logger(), "Subscribing to camera: %s", pre.c_str());

        info_subs_[pre] = create_subscription<sensor_msgs::msg::CameraInfo>(
            info_topic, 10, [this, pre](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                std::unique_lock lock(lut_mutex_);
                info_counts_[pre]++;
                if (luts_.find(pre) == luts_.end()) {
                    generate_lut(pre, msg);
                }
            });

        depth_subs_[pre] = create_subscription<sensor_msgs::msg::Image>(
            depth_topic, rclcpp::SensorDataQoS(),
            [this, pre](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::unique_lock lock(data_mutex_);
                latest_depths_[pre] = msg;
                depth_counts_[pre]++;
            });
            
        depth_counts_[pre] = 0;
        info_counts_[pre] = 0;
    }

    void generate_lut(const std::string& id, const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        cv::Mat lut(msg->height, msg->width, CV_32FC2);
        float fx = msg->k[0], fy = msg->k[4], cx = msg->k[2], cy = msg->k[5];
        for (int v = 0; v < (int)msg->height; v++) {
            for (int u = 0; u < (int)msg->width; u++) {
                lut.at<cv::Vec2f>(v, u) = cv::Vec2f((u - cx) / fx, (v - cy) / fy);
            }
        }
        luts_[id] = lut;
        RCLCPP_INFO(this->get_logger(), "Successfully generated deprojection LUT for %s", id.c_str());
    }

    void process_and_publish() {
        if (pub_->get_subscription_count() == 0) return;
    
        const float sb_w = get_parameter("sandbox_width").as_double();
        const float sb_l = get_parameter("sandbox_length").as_double();
        const int res = get_parameter("resolution_per_meter").as_int();
        const float max_step = get_parameter("max_step_size").as_double();
        const cv::Size hm_size = current_heightmap_.size();
    
        cv::Mat final_accum = cv::Mat::zeros(hm_size, CV_32FC1);
        cv::Mat final_count = cv::Mat::zeros(hm_size, CV_32FC1);
        std::mutex merge_mutex;
    
        std::vector<std::string> active_ids;
        {
            std::shared_lock data_lock(data_mutex_);
            for(auto const& [id, msg] : latest_depths_) active_ids.push_back(id);
        }
    
        // Process each camera in parallel
        cv::parallel_for_(cv::Range(0, active_ids.size()), [&](const cv::Range& r) {
            for (int i = r.start; i < r.end; i++) {
                std::string id = active_ids[i];
                
                // Local buffers for this specific camera
                cv::Mat cam_accum = cv::Mat::zeros(hm_size, CV_32FC1);
                cv::Mat cam_count = cv::Mat::zeros(hm_size, CV_32FC1);
    
                sensor_msgs::msg::Image::SharedPtr msg;
                cv::Mat lut;
                {
                    std::shared_lock data_lock(data_mutex_);
                    std::shared_lock lut_lock(lut_mutex_);
                    msg = latest_depths_[id];
                    lut = luts_[id];
                }
    
                Eigen::Isometry3f tf;
                try {
                    auto ts = tf_buffer_->lookupTransform("sandbox_origin", msg->header.frame_id, tf2::TimePointZero);
                    tf = tf2::transformToEigen(ts).cast<float>();
                } catch (...) { continue; }
    
                cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "16UC1");
                
                for (int v = 0; v < cv_ptr->image.rows; v++) {
                    const uint16_t* d_row = cv_ptr->image.ptr<uint16_t>(v);
                    const cv::Vec2f* l_row = lut.ptr<cv::Vec2f>(v);
                    for (int u = 0; u < cv_ptr->image.cols; u++) {
                        if (d_row[u] == 0) continue;
                        
                        float z = d_row[u] * 0.001f;
                        Eigen::Vector3f p = tf * Eigen::Vector3f(l_row[u][0] * z, l_row[u][1] * z, z);
    
                        int gx = (p.x() + (sb_w / 2.0f)) * res;
                        int gy = (p.y() + (sb_l / 2.0f)) * res;
    
                        if (gx >= 0 && gx < hm_size.width && gy >= 0 && gy < hm_size.height) {
                            cam_accum.at<float>(gy, gx) += p.z();
                            cam_count.at<float>(gy, gx) += 1.0f;
                        }
                    }
                }
    
                // Merging this camera's results into the global buffer
                std::lock_guard<std::mutex> lock(merge_mutex);
                final_accum += cam_accum;
                final_count += cam_count;
            }
        });
    
        // --- Validation Check ---
        double minV, maxV;
        cv::minMaxLoc(final_count, &minV, &maxV);
        if (maxV <= 0) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No points mapped to grid. Max count is 0.");
            return; 
        }
    
        cv::Mat mean_h;
        cv::divide(final_accum, final_count, mean_h);
        
        // Fill holes where count was 0
        cv::Mat mask;
        cv::compare(final_count, 0, mask, cv::CMP_EQ);
        mask.convertTo(mask, CV_8U);
        cv::inpaint(mean_h, mask, mean_h, 3, cv::INPAINT_NS);
        cv::GaussianBlur(mean_h, mean_h, cv::Size(5, 5), 1.0);
    
        // Temporal Interpolation
        cv::Mat delta = mean_h - current_heightmap_;
        cv::Mat clamped_delta, sign;
        cv::threshold(cv::abs(delta), clamped_delta, max_step, max_step, cv::THRESH_TRUNC);
        
        // Manual signum to avoid divide by zero
        sign = cv::Mat::zeros(delta.size(), CV_32FC1);
        sign.setTo(1.0, delta > 0);
        sign.setTo(-1.0, delta < 0);
        
        current_heightmap_ += clamped_delta.mul(sign);
    
        // Final minMax check for logs
        cv::minMaxLoc(current_heightmap_, &minV, &maxV);
        RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Heightmap ready. Range: [%.2f, %.2f]", minV, maxV);
    
        auto out_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", current_heightmap_).toImageMsg();
        out_msg->header.stamp = this->now();
        pub_->publish(*out_msg);
    }

    void log_diagnostics() {
        double interval = get_parameter("log_interval").as_double();
        RCLCPP_INFO(this->get_logger(), "--- Periodic Diagnostic Report (Last %.1fs) ---", interval);
        
        auto prefixes = get_parameter("camera_prefixes").as_string_array();
        for (const auto& pre : prefixes) {
            int d_count = depth_counts_[pre];
            int i_count = info_counts_[pre];
            
            if (d_count == 0) {
                RCLCPP_WARN(this->get_logger(), "  [%s] CRITICAL: No depth images received! Check QoS or connection.", pre.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "  [%s] Depth: %d msgs (%.1f Hz) | Info: %d msgs", 
                            pre.c_str(), d_count, d_count / interval, i_count);
            }
            depth_counts_[pre] = 0;
            info_counts_[pre] = 0;
        }
        RCLCPP_INFO(this->get_logger(), "------------------------------------------------");
    }

    std::shared_mutex data_mutex_, lut_mutex_;
    std::map<std::string, cv::Mat> luts_;
    std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> depth_subs_;
    std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr> info_subs_;
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> latest_depths_;
    
    // Diagnostic Counters
    std::map<std::string, int> depth_counts_;
    std::map<std::string, int> info_counts_;
    size_t last_sub_count_;

    cv::Mat current_heightmap_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr timer_, diag_timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<DynamicCloudMerger>();
    rclcpp::executors::MultiThreadedExecutor exec;
    exec.add_node(node);
    exec.spin();
    rclcpp::shutdown();
    return 0;
}

#include <map>
#include <mutex>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <opencv2/opencv.hpp>
#include <Eigen/Geometry>

namespace sandbox_components {

struct CameraPipeline {
    cv::Mat lut;
    sensor_msgs::msg::CameraInfo info;
    bool ready = false;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub;
};

class CameraReprojector : public rclcpp::Node {
public:
    explicit CameraReprojector(const rclcpp::NodeOptions & options) 
    : Node("camera_reprojector", rclcpp::NodeOptions(options).use_intra_process_comms(true)) {
        this->declare_parameter("sandbox_width", 4.0);
        this->declare_parameter("sandbox_length", 6.0);
        this->declare_parameter("output_width", 600);
        this->declare_parameter("output_height", 400);
        this->declare_parameter("target_namespace", "/");

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("local_heightmap", rclcpp::SensorDataQoS());

        // 1. Create a dedicated callback group for the processing timer
        timer_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        // 2. Assign the processing timer to this new group
        process_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), 
            std::bind(&CameraReprojector::combine_and_publish, this),
            timer_cb_group_  // <-- This tells ROS to run this in a separate thread!
        );
        
        discovery_timer_ = this->create_wall_timer(std::chrono::seconds(2), std::bind(&CameraReprojector::discover_cameras, this));
    }

private:
    void discover_cameras() {
        std::string target_ns = this->get_parameter("target_namespace").as_string();
        auto topics = this->get_topic_names_and_types();
        for (auto const& [name, types] : topics) {
            if (name.find("/depth/image_rect_raw") != std::string::npos && name.find(target_ns) == 0) {
                if (pipelines_.find(name) == pipelines_.end()) {
                    create_pipeline(name);
                }
            }
        }
    }

    void create_pipeline(const std::string& topic) {
        auto pipe = std::make_shared<CameraPipeline>();
        pipe->depth_sub = this->create_subscription<sensor_msgs::msg::Image>(
            topic, rclcpp::SensorDataQoS(),
            [this, topic](const sensor_msgs::msg::Image::SharedPtr msg) {
                std::lock_guard<std::mutex> lock(data_mutex_);
                latest_frames_[topic] = msg;
            });
            
        std::string info_topic = topic;
        info_topic.replace(info_topic.find("image_rect_raw"), 14, "camera_info");
        
        pipe->info_sub = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            info_topic, 10, [this, pipe](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
                if (!pipe->ready) {
                    pipe->lut = cv::Mat(msg->height, msg->width, CV_32FC2);
                    float fx = msg->k[0], fy = msg->k[4], cx = msg->k[2], cy = msg->k[5];
                    for (int v = 0; v < (int)msg->height; v++) {
                        for (int u = 0; u < (int)msg->width; u++) {
                            pipe->lut.at<cv::Vec2f>(v, u) = cv::Vec2f((u - cx) / fx, (v - cy) / fy);
                        }
                    }
                    pipe->ready = true;
                }
            });
            
        pipelines_[topic] = pipe;
        RCLCPP_INFO(this->get_logger(), "Pipeline initialized for: %s", topic.c_str());
    }

    void combine_and_publish() {
        if (pub_->get_subscription_count() == 0 || latest_frames_.empty()) return;

        int out_w = this->get_parameter("output_width").as_int();
        int out_h = this->get_parameter("output_height").as_int();
        float sb_w = this->get_parameter("sandbox_width").as_double();
        float sb_l = this->get_parameter("sandbox_length").as_double();

        cv::Mat combined_data = cv::Mat::zeros(out_h, out_w, CV_32FC2);
        std::mutex merge_mutex;

        std::vector<std::string> active_topics;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            for (auto const& [t, msg] : latest_frames_) active_topics.push_back(t);
        }

        cv::parallel_for_(cv::Range(0, active_topics.size()), [&](const cv::Range& r) {
            for (int i = r.start; i < r.end; i++) {
                std::string topic = active_topics[i];
                auto pipe = pipelines_[topic];
                if (!pipe || !pipe->ready) continue;

                sensor_msgs::msg::Image::SharedPtr msg;
                {
                    std::lock_guard<std::mutex> lock(data_mutex_);
                    msg = latest_frames_[topic];
                }

                Eigen::Isometry3f tf;
                try {
                    auto ts = tf_buffer_->lookupTransform("sandbox_origin", msg->header.frame_id, tf2::TimePointZero);
                    tf = tf2::transformToEigen(ts).cast<float>();
                } catch (const tf2::TransformException & ex) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000, 
                    "TF Error preventing reprojection: %s", ex.what());
                    continue;
                }

                cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "16UC1");
                cv::Mat cam_data = cv::Mat::zeros(out_h, out_w, CV_32FC2);

                for (int v = 0; v < cv_ptr->image.rows; v++) {
                    const uint16_t* d_row = cv_ptr->image.ptr<uint16_t>(v);
                    const cv::Vec2f* l_row = pipe->lut.ptr<cv::Vec2f>(v);
                    for (int u = 0; u < cv_ptr->image.cols; u++) {
                        if (d_row[u] == 0) continue;
                        float z = d_row[u] * 0.001f;
                        Eigen::Vector3f p = tf * Eigen::Vector3f(l_row[u][0] * z, l_row[u][1] * z, z);
                        int gx = ((p.x() / sb_w) + 0.5f) * out_w;
                        int gy = ((p.y() / sb_l) + 0.5f) * out_h;
                        if (gx >= 0 && gx < out_w && gy >= 0 && gy < out_h) {
                            float* ptr = (float*)cam_data.ptr(gy, gx);
                            ptr[0] += p.z();
                            ptr[1] += 1.0f;
                        }
                    }
                }
                std::lock_guard<std::mutex> lock(merge_mutex);
                combined_data += cam_data;
            }
        });

        auto unique_msg = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(std_msgs::msg::Header(), "32FC2", combined_data).toImageMsg(*unique_msg);
        pub_->publish(std::move(unique_msg));
    }

    std::mutex data_mutex_;
    std::map<std::string, std::shared_ptr<CameraPipeline>> pipelines_;
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> latest_frames_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr discovery_timer_, process_timer_;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
};
} // namespace sandbox_components
RCLCPP_COMPONENTS_REGISTER_NODE(sandbox_components::CameraReprojector)

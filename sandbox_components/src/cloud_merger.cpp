#include <mutex>
#include <map>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>

namespace sandbox_components {

class CloudMerger : public rclcpp::Node {
public:
    explicit CloudMerger(const rclcpp::NodeOptions & options) 
    : Node("cloud_merger", rclcpp::NodeOptions(options).use_intra_process_comms(true)) {
        this->declare_parameter("output_width", 600);
        this->declare_parameter("output_height", 400);
        this->declare_parameter("temporal_smoothing_factor", 5.0); // Multiplier for delta
        this->declare_parameter("temporal_min_alpha", 0.05);       // Floor for noise suppression

        int w = this->get_parameter("output_width").as_int();
        int h = this->get_parameter("output_height").as_int();
        current_heightmap_ = cv::Mat::zeros(h, w, CV_32FC1);

        pub_ = this->create_publisher<sensor_msgs::msg::Image>("/sandbox/heightmap", rclcpp::SensorDataQoS());
        discovery_timer_ = this->create_wall_timer(std::chrono::seconds(2), std::bind(&CloudMerger::discover_topics, this));

        timer_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33), 
            std::bind(&CloudMerger::merge_and_publish, this),
            timer_cb_group_
        );
    }

private:
    void discover_topics() {
        auto topics = this->get_topic_names_and_types();
        for (auto const& [name, types] : topics) {
            if (name.find("local_heightmap") != std::string::npos) {
                if (subs_.find(name) == subs_.end()) {
                    RCLCPP_INFO(this->get_logger(), "Merger subscribed to: %s", name.c_str());
                    subs_[name] = this->create_subscription<sensor_msgs::msg::Image>(
                        name, rclcpp::SensorDataQoS(), [this, name](const sensor_msgs::msg::Image::SharedPtr msg) {
                            std::lock_guard<std::mutex> lock(mtx_);
                            latest_maps_[name] = msg;
                        });
                }
            }
        }
    }

    void merge_and_publish() {
        if (latest_maps_.empty()) return;

        int w = this->get_parameter("output_width").as_int();
        int h = this->get_parameter("output_height").as_int();
        cv::Mat final_accum = cv::Mat::zeros(h, w, CV_32FC1);
        cv::Mat final_count = cv::Mat::zeros(h, w, CV_32FC1);

        {
            std::lock_guard<std::mutex> lock(mtx_);
            for (auto const& [topic, msg] : latest_maps_) {
                cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "32FC2");
                cv::Mat planes[2];
                cv::split(cv_ptr->image, planes);
                final_accum += planes[0];
                final_count += planes[1];
            }
        }

        cv::Mat mean_h;
        cv::divide(final_accum, final_count, mean_h);
        
        cv::Mat mask;
        cv::compare(final_count, 0, mask, cv::CMP_EQ);
        mean_h.setTo(0.0f, mask);
        mask.convertTo(mask, CV_8U);
        cv::inpaint(mean_h, mask, mean_h, 3, cv::INPAINT_NS);
        cv::GaussianBlur(mean_h, mean_h, cv::Size(5, 5), 1.0);

        // --- NON-LINEAR ADAPTIVE EXPONENTIAL SMOOTHING ---
        float smoothing_factor = this->get_parameter("temporal_smoothing_factor").as_double();
        float min_alpha = this->get_parameter("temporal_min_alpha").as_double();

        // 1. Calculate raw delta and absolute delta
        cv::Mat delta = mean_h - current_heightmap_;
        cv::Mat abs_delta = cv::abs(delta);
        
        // 2. Base alpha
        cv::Mat alpha = abs_delta * smoothing_factor;
        
        // 3. THE MAGIC: Square the alpha matrix. 
        // This severely penalizes small values (noise) while keeping large values near 1.0
        cv::pow(alpha, 2.0, alpha);
        
        // 4. Clamp alpha so it never exceeds 1.0 (overshoot) and never drops below min_alpha (frozen)
        cv::threshold(alpha, alpha, 1.0, 1.0, cv::THRESH_TRUNC); 
        cv::max(alpha, min_alpha, alpha);

        // 5. Apply the weighted step: current += delta * alpha
        current_heightmap_ += delta.mul(alpha);

        // SAFETY NET: If a NaN somehow sneaks through, catch it so the map auto-heals
        cv::patchNaNs(current_heightmap_, 0.0f);

        auto unique_msg = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", current_heightmap_).toImageMsg(*unique_msg);
        pub_->publish(std::move(unique_msg));
    }

    std::mutex mtx_;
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> latest_maps_;
    std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subs_;
    cv::Mat current_heightmap_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    rclcpp::TimerBase::SharedPtr timer_, discovery_timer_;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
};
} // namespace sandbox_components
RCLCPP_COMPONENTS_REGISTER_NODE(sandbox_components::CloudMerger)

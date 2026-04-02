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
        this->declare_parameter("output_width", 256);
        this->declare_parameter("output_height", 256);

        // Bilateral spatial filter (replaces Gaussian blur; preserves sand relief edges)
        this->declare_parameter("bilateral_d", 7);                // Neighborhood diameter in pixels
        this->declare_parameter("bilateral_sigma_color", 0.005);  // Height delta (m) treated as edge; lower = sharper relief
        this->declare_parameter("bilateral_sigma_space", 3.0);    // Spatial falloff in pixels

        // Slow layer: stable terrain + physics base (non-linear exponential smoothing)
        this->declare_parameter("slow_smoothing_factor", 2.0);    // Multiplier for delta before squaring
        this->declare_parameter("slow_min_alpha", 0.01);          // Floor alpha; keeps the layer from freezing

        // Fast layer: hands and placed objects (decays back toward slow layer when stimulus leaves)
        this->declare_parameter("fast_alpha", 0.40);              // Chase alpha when object detected (~5 frames to respond)
        this->declare_parameter("fast_detection_threshold", 0.015); // Min height delta (m) to classify as a real object
        this->declare_parameter("fast_decay_alpha", 0.10);        // Decay rate back toward slow layer when no object (~10 frames)

        int w = this->get_parameter("output_width").as_int();
        int h = this->get_parameter("output_height").as_int();
        slow_layer_ = cv::Mat::zeros(h, w, CV_32FC1);
        fast_layer_ = cv::Mat::zeros(h, w, CV_32FC1);

        // /sandbox/heightmap         — visual composite (terrain + transient objects); backward-compatible
        // /sandbox/heightmap/physics — slow layer only (ultra-smooth; for water/physics simulation)
        pub_visual_   = this->create_publisher<sensor_msgs::msg::Image>("/sandbox/heightmap",         rclcpp::SensorDataQoS());
        pub_physics_  = this->create_publisher<sensor_msgs::msg::Image>("/sandbox/heightmap/physics", rclcpp::SensorDataQoS());

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

        // Fill void regions (zero count) with simple dilation-based fill rather than inpaint
        cv::Mat void_mask;
        cv::compare(final_count, 0, void_mask, cv::CMP_EQ);
        mean_h.setTo(0.0f, void_mask);

        cv::Mat dilated;
        cv::dilate(mean_h, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
        void_mask.convertTo(void_mask, CV_8U);
        dilated.copyTo(mean_h, void_mask);

        // Bilateral filter: smooths flat areas while preserving sand relief edges
        int bd              = this->get_parameter("bilateral_d").as_int();
        double bs_color     = this->get_parameter("bilateral_sigma_color").as_double();
        double bs_space     = this->get_parameter("bilateral_sigma_space").as_double();
        cv::Mat mean_h_filtered;
        cv::bilateralFilter(mean_h, mean_h_filtered, bd, bs_color, bs_space);
        mean_h = mean_h_filtered;

        // --- SLOW LAYER: stable terrain + physics base ---
        // Non-linear adaptive exponential smoothing.
        // Small deltas (noise) get severely suppressed; large deltas (real terrain change) pass through.
        float slow_sf    = (float)this->get_parameter("slow_smoothing_factor").as_double();
        float slow_min_a = (float)this->get_parameter("slow_min_alpha").as_double();

        cv::Mat slow_delta     = mean_h - slow_layer_;
        cv::Mat abs_slow_delta = cv::abs(slow_delta);
        cv::Mat slow_alpha     = abs_slow_delta * slow_sf;
        cv::pow(slow_alpha, 2.0, slow_alpha);
        cv::threshold(slow_alpha, slow_alpha, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::max(slow_alpha, slow_min_a, slow_alpha);

        slow_layer_ += slow_delta.mul(slow_alpha);
        cv::patchNaNs(slow_layer_, 0.0f);

        // --- FAST LAYER: objects and hands ---
        // Chases large deltas quickly; decays back toward slow_layer_ when stimulus disappears.
        float fast_a      = (float)this->get_parameter("fast_alpha").as_double();
        float fast_thresh = (float)this->get_parameter("fast_detection_threshold").as_double();
        float fast_decay  = (float)this->get_parameter("fast_decay_alpha").as_double();

        cv::Mat fast_delta     = mean_h - fast_layer_;
        cv::Mat abs_fast_delta = cv::abs(fast_delta);

        // Binary mask: 1.0 where a real object is present, 0.0 where not
        cv::Mat object_mask;
        cv::threshold(abs_fast_delta, object_mask, fast_thresh, 1.0f, cv::THRESH_BINARY);

        // Where object detected: chase the measurement
        // Where no object: relax back toward the slow (terrain) layer
        cv::Mat fast_update =
            fast_delta.mul(object_mask) * fast_a +
            (slow_layer_ - fast_layer_).mul(1.0f - object_mask) * fast_decay;
        fast_layer_ += fast_update;
        cv::patchNaNs(fast_layer_, 0.0f);

        // --- COMPOSITE OUTPUT (visual) ---
        // Base is the slow layer; fast layer is blended in where it differs significantly.
        // The blend mask is smoothed to avoid hard spatial edges.
        cv::Mat layer_diff = cv::abs(fast_layer_ - slow_layer_);
        cv::Mat blend_weight;
        cv::threshold(layer_diff, blend_weight, 0.005f, 1.0f, cv::THRESH_BINARY);
        cv::GaussianBlur(blend_weight, blend_weight, cv::Size(9, 9), 2.0);
        cv::Mat visual_output = slow_layer_ + (fast_layer_ - slow_layer_).mul(blend_weight);
        cv::patchNaNs(visual_output, 0.0f);

        // Publish physics topic (slow layer only — maximally smooth for water sim)
        auto physics_msg = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", slow_layer_).toImageMsg(*physics_msg);
        pub_physics_->publish(std::move(physics_msg));

        // Publish visual topic (composite — terrain + transient objects/hands)
        auto visual_msg = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", visual_output).toImageMsg(*visual_msg);
        pub_visual_->publish(std::move(visual_msg));
    }

    std::mutex mtx_;
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> latest_maps_;
    std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subs_;
    cv::Mat slow_layer_;   // Stable terrain; drives physics/water sim
    cv::Mat fast_layer_;   // Transient objects and hands; decays without stimulus
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_visual_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_physics_;
    rclcpp::TimerBase::SharedPtr timer_, discovery_timer_;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
};
} // namespace sandbox_components
RCLCPP_COMPONENTS_REGISTER_NODE(sandbox_components::CloudMerger)

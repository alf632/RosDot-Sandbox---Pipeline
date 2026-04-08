#include <mutex>
#include <map>
#include <vector>
#include <algorithm>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>

namespace sandbox_components {

// Three-layer temporal pipeline:
//
//  terrain_layer_  — non-linear exponential smoothing; tracks sand relief (seconds),
//                    strongly suppresses frame-to-frame noise; feeds visual output.
//
//  physics_layer_  — linear EMA of terrain_layer_; much slower, ultra-smooth;
//                    immune to sand reshaping transients; feeds /sandbox/heightmap/physics.
//
//  fast_layer_     — tracks large sudden deltas (hands, placed objects) with high alpha,
//                    decays back toward terrain_layer_ when stimulus is gone;
//                    blended over terrain_layer_ in the visual composite.

class CloudMerger : public rclcpp::Node {
public:
    explicit CloudMerger(const rclcpp::NodeOptions & options)
    : Node("cloud_merger", rclcpp::NodeOptions(options).use_intra_process_comms(true)) {
        this->declare_parameter("output_width", 256);
        this->declare_parameter("output_height", 256);

        // Bilateral spatial filter (replaces Gaussian blur; preserves sand relief edges)
        this->declare_parameter("bilateral_d", 7);                // Neighborhood diameter in pixels
        this->declare_parameter("bilateral_sigma_color", 0.005);  // Height delta (m) treated as an edge; lower = sharper relief
        this->declare_parameter("bilateral_sigma_space", 3.0);    // Spatial falloff in pixels

        // Terrain layer: tracks sand relief, suppresses noise.
        // The squaring trick means: alpha ≈ (|delta| * factor)^2, floored at min_alpha.
        // A 3mm noise spike at factor=5 gets alpha=(0.003*5)^2=0.0002 → min_alpha wins → 0.05
        // A 30mm sand hill  at factor=5 gets alpha=(0.030*5)^2=0.022  → min_alpha wins → 0.05
        // A 200mm hand      at factor=5 gets alpha=(0.200*5)^2=1.0    → clamped to 1.0
        // Net effect: noise and sand both converge at min_alpha rate; hands jump instantly.
        this->declare_parameter("terrain_smoothing_factor", 5.0);  // Match original pipeline
        this->declare_parameter("terrain_min_alpha", 0.05);        // Floor blend rate per frame

        // Physics layer: linear EMA of terrain_layer_. Decouples water-sim smoothness from
        // terrain tracking speed. Lower = smoother but more lag behind sand reshaping.
        this->declare_parameter("physics_alpha", 0.02);            // ~50 frames (1.7 s) to 63% of a step change

        // Fast layer: hands and placed objects.
        this->declare_parameter("fast_alpha", 0.40);               // Chase alpha when object detected (~5 frames)
        this->declare_parameter("fast_detection_threshold", 0.008);// Min delta (m) to count as an object (8 mm)
        this->declare_parameter("fast_decay_alpha", 0.10);         // Decay toward terrain layer when object gone (~10 frames)

        // Per-cell outlier rejection for multi-device setups (>2 heightmap sources).
        // When ≥ outlier_min_sources contribute to a cell, the median height is computed
        // and cameras deviating by more than outlier_rejection_threshold are discarded.
        this->declare_parameter("outlier_rejection_threshold", 0.020);  // metres
        this->declare_parameter("outlier_min_sources", 3);              // minimum sources to enable rejection

        int w = this->get_parameter("output_width").as_int();
        int h = this->get_parameter("output_height").as_int();
        terrain_layer_ = cv::Mat::zeros(h, w, CV_32FC1);
        physics_layer_ = cv::Mat::zeros(h, w, CV_32FC1);
        fast_layer_    = cv::Mat::zeros(h, w, CV_32FC1);

        // /sandbox/heightmap         — visual composite (terrain + objects/hands); backward-compatible
        // /sandbox/heightmap/physics — ultra-smooth physics base; no transient objects
        pub_visual_  = this->create_publisher<sensor_msgs::msg::Image>("/sandbox/heightmap",         rclcpp::SensorDataQoS());
        pub_physics_ = this->create_publisher<sensor_msgs::msg::Image>("/sandbox/heightmap/physics", rclcpp::SensorDataQoS());

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
        float outlier_thresh = (float)this->get_parameter("outlier_rejection_threshold").as_double();
        int   outlier_min    = this->get_parameter("outlier_min_sources").as_int();

        cv::Mat final_accum = cv::Mat::zeros(h, w, CV_32FC1);
        cv::Mat final_count = cv::Mat::zeros(h, w, CV_32FC1);

        // Count sources and decide strategy
        int n_sources;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            n_sources = (int)latest_maps_.size();
        }

        bool use_outlier_rejection =
            (outlier_thresh > 0.0f && n_sources >= outlier_min);

        if (!use_outlier_rejection) {
            // Simple accumulation (original path — ≤2 sources or rejection disabled)
            std::lock_guard<std::mutex> lock(mtx_);
            for (auto const& [topic, msg] : latest_maps_) {
                cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "32FC2");
                cv::Mat planes[2];
                cv::split(cv_ptr->image, planes);
                final_accum += planes[0];
                final_count += planes[1];
            }
        } else {
            // Per-cell median-based outlier rejection
            std::vector<cv::Mat> src_heights, src_counts;
            {
                std::lock_guard<std::mutex> lock(mtx_);
                for (auto const& [topic, msg] : latest_maps_) {
                    cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "32FC2");
                    cv::Mat planes[2];
                    cv::split(cv_ptr->image, planes);
                    src_heights.push_back(planes[0].clone());
                    src_counts.push_back(planes[1].clone());
                }
            }

            int n_src = (int)src_heights.size();
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    struct Entry { float mean_h; int idx; };
                    std::vector<Entry> entries;
                    entries.reserve(n_src);
                    for (int s = 0; s < n_src; s++) {
                        float cnt = src_counts[s].at<float>(y, x);
                        if (cnt > 0.5f) {
                            entries.push_back({src_heights[s].at<float>(y, x) / cnt, s});
                        }
                    }
                    if ((int)entries.size() < outlier_min) {
                        for (auto const& e : entries) {
                            final_accum.at<float>(y, x) += src_heights[e.idx].at<float>(y, x);
                            final_count.at<float>(y, x) += src_counts[e.idx].at<float>(y, x);
                        }
                    } else {
                        std::sort(entries.begin(), entries.end(),
                                  [](auto const& a, auto const& b){ return a.mean_h < b.mean_h; });
                        float median = entries[entries.size() / 2].mean_h;
                        for (auto const& e : entries) {
                            if (std::abs(e.mean_h - median) <= outlier_thresh) {
                                final_accum.at<float>(y, x) += src_heights[e.idx].at<float>(y, x);
                                final_count.at<float>(y, x) += src_counts[e.idx].at<float>(y, x);
                            }
                        }
                    }
                }
            }
        }

        cv::Mat mean_h;
        cv::divide(final_accum, final_count, mean_h);

        // Fill void regions (zero count) with dilation-based fill (avoids inpaint artifacts)
        cv::Mat void_mask;
        cv::compare(final_count, 0, void_mask, cv::CMP_EQ);
        mean_h.setTo(0.0f, void_mask);
        cv::Mat dilated;
        cv::dilate(mean_h, dilated, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)));
        void_mask.convertTo(void_mask, CV_8U);
        dilated.copyTo(mean_h, void_mask);

        // Bilateral filter: smooths flat areas while preserving sand relief edges
        int bd          = this->get_parameter("bilateral_d").as_int();
        double bs_color = this->get_parameter("bilateral_sigma_color").as_double();
        double bs_space = this->get_parameter("bilateral_sigma_space").as_double();
        cv::Mat mean_h_filtered;
        cv::bilateralFilter(mean_h, mean_h_filtered, bd, bs_color, bs_space);
        mean_h = mean_h_filtered;

        // --- TERRAIN LAYER ---
        // Non-linear adaptive smoothing: alpha = clamp((|delta|*factor)^2, min_alpha, 1.0)
        // Noise and small features converge slowly at min_alpha; large step changes pass instantly.
        float t_sf     = (float)this->get_parameter("terrain_smoothing_factor").as_double();
        float t_min_a  = (float)this->get_parameter("terrain_min_alpha").as_double();

        cv::Mat t_delta     = mean_h - terrain_layer_;
        cv::Mat abs_t_delta = cv::abs(t_delta);
        cv::Mat t_alpha     = abs_t_delta * t_sf;
        cv::pow(t_alpha, 2.0, t_alpha);
        cv::threshold(t_alpha, t_alpha, 1.0, 1.0, cv::THRESH_TRUNC);
        cv::max(t_alpha, t_min_a, t_alpha);

        terrain_layer_ += t_delta.mul(t_alpha);
        cv::patchNaNs(terrain_layer_, 0.0f);

        // --- PHYSICS LAYER ---
        // Simple linear EMA of terrain_layer_. Extra smoothing decouples water-sim quality
        // from terrain tracking speed. Adjust physics_alpha to taste.
        float p_alpha = (float)this->get_parameter("physics_alpha").as_double();
        physics_layer_ += (terrain_layer_ - physics_layer_) * p_alpha;
        cv::patchNaNs(physics_layer_, 0.0f);

        // --- FAST LAYER: objects and hands ---
        // Chases large deltas quickly; decays back toward terrain_layer_ when gone.
        float fast_a      = (float)this->get_parameter("fast_alpha").as_double();
        float fast_thresh = (float)this->get_parameter("fast_detection_threshold").as_double();
        float fast_decay  = (float)this->get_parameter("fast_decay_alpha").as_double();

        cv::Mat fast_delta     = mean_h - fast_layer_;
        cv::Mat abs_fast_delta = cv::abs(fast_delta);

        cv::Mat object_mask;
        cv::threshold(abs_fast_delta, object_mask, fast_thresh, 1.0f, cv::THRESH_BINARY);

        cv::Mat fast_update =
            fast_delta.mul(object_mask) * fast_a +
            (terrain_layer_ - fast_layer_).mul(1.0f - object_mask) * fast_decay;
        fast_layer_ += fast_update;
        cv::patchNaNs(fast_layer_, 0.0f);

        // --- COMPOSITE VISUAL OUTPUT ---
        // Base is terrain_layer_; fast_layer_ is blended in where it differs significantly.
        // Blend mask is Gaussian-smoothed to avoid hard spatial edges around objects.
        cv::Mat layer_diff = cv::abs(fast_layer_ - terrain_layer_);
        cv::Mat blend_weight;
        cv::threshold(layer_diff, blend_weight, 0.005f, 1.0f, cv::THRESH_BINARY);
        cv::GaussianBlur(blend_weight, blend_weight, cv::Size(9, 9), 2.0);
        cv::Mat visual_output = terrain_layer_ + (fast_layer_ - terrain_layer_).mul(blend_weight);
        cv::patchNaNs(visual_output, 0.0f);

        // Publish physics (extra-smooth, no transient objects)
        auto physics_msg = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", physics_layer_).toImageMsg(*physics_msg);
        pub_physics_->publish(std::move(physics_msg));

        // Publish visual composite (terrain + hands/objects)
        auto visual_msg = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(std_msgs::msg::Header(), "32FC1", visual_output).toImageMsg(*visual_msg);
        pub_visual_->publish(std::move(visual_msg));
    }

    std::mutex mtx_;
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> latest_maps_;
    std::map<std::string, rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr> subs_;
    cv::Mat terrain_layer_;  // Tracks sand relief; noise-suppressed via non-linear smoothing
    cv::Mat physics_layer_;  // Ultra-smooth EMA of terrain_layer_; for water/physics sim
    cv::Mat fast_layer_;     // Transient objects/hands blended over terrain in visual output
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_visual_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_physics_;
    rclcpp::TimerBase::SharedPtr timer_, discovery_timer_;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
};
} // namespace sandbox_components
RCLCPP_COMPONENTS_REGISTER_NODE(sandbox_components::CloudMerger)

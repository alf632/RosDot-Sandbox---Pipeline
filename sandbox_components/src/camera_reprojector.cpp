#include <map>
#include <mutex>
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <algorithm>
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
        this->declare_parameter("depth_min_mm", 300);   // Reject depth readings closer than this
        this->declare_parameter("depth_max_mm", 1500);  // Reject depth readings farther than this
        this->declare_parameter("median_blur_kernel", 3); // Odd integer; 1 disables, 3 or 5 recommended
        // Minimum number of depth rays that must hit a grid cell (per camera) for it to count.
        // 0.5 → accept any single hit (original behaviour, recommended for most setups).
        // 1.5 → require ≥2 hits (rejects oblique/edge cells but can cause holes in blanket/box coverage).
        this->declare_parameter("sparse_hit_threshold", 0.5);
        // Directory containing per-camera TF JSON files with optional _quality_weight field.
        this->declare_parameter("calibration_dir", "/tmp/calibrations/tf_configs");
        // Max per-cell height deviation from median before a camera is rejected (metres).
        // Only active when >2 cameras contribute to a cell.  0 = disabled.
        this->declare_parameter("outlier_rejection_threshold", 0.020);

        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
        pub_ = this->create_publisher<sensor_msgs::msg::Image>("local_heightmap", rclcpp::SensorDataQoS());

        timer_cb_group_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

        process_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(33),
            std::bind(&CameraReprojector::combine_and_publish, this),
            timer_cb_group_
        );

        discovery_timer_ = this->create_wall_timer(std::chrono::seconds(2), std::bind(&CameraReprojector::discover_cameras, this));

        // Load weights now, then reload every 30 s
        reload_camera_weights();
        weight_timer_ = this->create_wall_timer(std::chrono::seconds(30),
            std::bind(&CameraReprojector::reload_camera_weights, this));
    }

private:
    // ── camera serial extraction ────────────────────────────────────────────
    // Works for strings like "cam_101622075200_link" or
    // "cam_101622075200_depth_optical_frame" — returns "101622075200".
    static std::string extract_serial(const std::string& s) {
        auto pos = s.find("cam_");
        if (pos == std::string::npos) return "";
        pos += 4;
        auto end = s.find('_', pos);
        if (end == std::string::npos) end = s.size();
        return s.substr(pos, end - pos);
    }

    // ── weight loading from calibration JSON files ──────────────────────────
    void reload_camera_weights() {
        std::string cal_dir = this->get_parameter("calibration_dir").as_string();
        std::map<std::string, float> new_weights;
        try {
            for (auto const& entry : std::filesystem::directory_iterator(cal_dir)) {
                if (entry.path().extension() != ".json") continue;
                std::string serial = extract_serial(entry.path().stem().string());
                if (serial.empty()) continue;
                float w = parse_quality_weight(entry.path().string());
                new_weights[serial] = w;
            }
        } catch (const std::filesystem::filesystem_error&) {
            // Directory doesn't exist yet — not an error during early boot.
        }
        if (!new_weights.empty()) {
            std::lock_guard<std::mutex> lock(weight_mutex_);
            camera_weights_ = std::move(new_weights);
        }
    }

    // Minimal JSON value extraction — finds "_quality_weight": <number> in file.
    static float parse_quality_weight(const std::string& path) {
        std::ifstream f(path);
        if (!f.is_open()) return 1.0f;
        std::string line;
        while (std::getline(f, line)) {
            auto pos = line.find("\"_quality_weight\"");
            if (pos == std::string::npos) continue;
            pos = line.find(':', pos);
            if (pos == std::string::npos) continue;
            try {
                return std::stof(line.substr(pos + 1));
            } catch (...) {}
        }
        return 1.0f;  // default: full weight
    }

    float get_camera_weight(const std::string& frame_id) {
        std::string serial = extract_serial(frame_id);
        if (serial.empty()) return 1.0f;
        std::lock_guard<std::mutex> lock(weight_mutex_);
        auto it = camera_weights_.find(serial);
        return (it != camera_weights_.end()) ? it->second : 1.0f;
    }

    // ── camera discovery ────────────────────────────────────────────────────
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

    // ── main processing loop ────────────────────────────────────────────────
    void combine_and_publish() {
        if (pub_->get_subscription_count() == 0 || latest_frames_.empty()) return;

        int out_w = this->get_parameter("output_width").as_int();
        int out_h = this->get_parameter("output_height").as_int();
        float sb_w = this->get_parameter("sandbox_width").as_double();
        float sb_l = this->get_parameter("sandbox_length").as_double();
        uint16_t min_d = (uint16_t)this->get_parameter("depth_min_mm").as_int();
        uint16_t max_d = (uint16_t)this->get_parameter("depth_max_mm").as_int();
        int blur_k = this->get_parameter("median_blur_kernel").as_int();
        float sparse_thresh = (float)this->get_parameter("sparse_hit_threshold").as_double();
        float outlier_thresh = (float)this->get_parameter("outlier_rejection_threshold").as_double();

        std::vector<std::string> active_topics;
        {
            std::lock_guard<std::mutex> lock(data_mutex_);
            for (auto const& [t, msg] : latest_frames_) active_topics.push_back(t);
        }

        int num_cams = (int)active_topics.size();

        // Per-camera results: heightmap (CV_32FC2) + weight
        struct CamResult {
            cv::Mat data;       // empty if camera skipped
            float weight = 1.0f;
        };
        std::vector<CamResult> cam_results(num_cams);

        // Process each camera in parallel
        cv::parallel_for_(cv::Range(0, num_cams), [&](const cv::Range& r) {
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
                std::string frame_id;
                try {
                    frame_id = msg->header.frame_id;
                    auto ts = tf_buffer_->lookupTransform("sandbox_origin", frame_id, tf2::TimePointZero);
                    tf = tf2::transformToEigen(ts).cast<float>();
                } catch (const tf2::TransformException & ex) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                    "TF Error preventing reprojection: %s", ex.what());
                    continue;
                }

                cv_bridge::CvImageConstPtr cv_ptr = cv_bridge::toCvShare(msg, "16UC1");

                // Apply spatial median blur to remove hot pixels and flying pixels
                cv::Mat depth_img;
                if (blur_k > 1) {
                    cv::medianBlur(cv_ptr->image, depth_img, blur_k);
                } else {
                    depth_img = cv_ptr->image;
                }

                cv::Mat cam_data = cv::Mat::zeros(out_h, out_w, CV_32FC2);

                for (int v = 0; v < depth_img.rows; v++) {
                    const uint16_t* d_row = depth_img.ptr<uint16_t>(v);
                    const cv::Vec2f* l_row = pipe->lut.ptr<cv::Vec2f>(v);
                    for (int u = 0; u < depth_img.cols; u++) {
                        uint16_t raw_d = d_row[u];
                        // Reject zero, below-minimum, and above-maximum depth readings
                        if (raw_d < min_d || raw_d > max_d) continue;
                        float z = raw_d * 0.001f;
                        Eigen::Vector3f p = tf * Eigen::Vector3f(l_row[u][0] * z, l_row[u][1] * z, z);
                        int gx = cvRound(((p.x() / sb_w) + 0.5f) * out_w);
                        int gy = cvRound(((p.y() / sb_l) + 0.5f) * out_h);
                        if (gx >= 0 && gx < out_w && gy >= 0 && gy < out_h) {
                            float* ptr = (float*)cam_data.ptr(gy, gx);
                            ptr[0] += p.z();
                            ptr[1] += 1.0f;
                        }
                    }
                }

                // Sparse-hit mask: discard cells below the minimum hit count.
                // Default threshold=0.5 keeps all cells with ≥1 hit (same as no mask).
                // Raise to 1.5 to require ≥2 hits if single-hit noise is a problem.
                if (sparse_thresh > 0.0f) {
                    cv::Mat cam_planes[2];
                    cv::split(cam_data, cam_planes);
                    cv::Mat valid_mask;
                    cv::threshold(cam_planes[1], valid_mask, sparse_thresh, 1.0f, cv::THRESH_BINARY);
                    cam_planes[0] = cam_planes[0].mul(valid_mask);
                    cam_planes[1] = cam_planes[1].mul(valid_mask);
                    cv::merge(cam_planes, 2, cam_data);
                }

                cam_results[i].data = cam_data;
                cam_results[i].weight = get_camera_weight(frame_id);
            }
        });

        // ── combine per-camera heightmaps ───────────────────────────────────
        cv::Mat combined_data = cv::Mat::zeros(out_h, out_w, CV_32FC2);

        // Count cameras that actually produced data
        int active_cams = 0;
        for (auto const& cr : cam_results) {
            if (!cr.data.empty()) active_cams++;
        }

        if (active_cams <= 2 || outlier_thresh <= 0.0f) {
            // ≤2 cameras or outlier rejection disabled: weighted sum
            for (auto const& cr : cam_results) {
                if (cr.data.empty()) continue;
                if (std::abs(cr.weight - 1.0f) < 1e-6f) {
                    combined_data += cr.data;
                } else {
                    combined_data += cr.data * cr.weight;
                }
            }
        } else {
            // >2 cameras: per-cell median-based outlier rejection
            // Split each camera's data into height-sum and count planes
            std::vector<int> valid_indices;
            std::vector<cv::Mat> cam_heights, cam_counts;
            std::vector<float> cam_weights;
            for (int i = 0; i < num_cams; i++) {
                if (cam_results[i].data.empty()) continue;
                valid_indices.push_back(i);
                cv::Mat planes[2];
                cv::split(cam_results[i].data, planes);
                cam_heights.push_back(planes[0]);
                cam_counts.push_back(planes[1]);
                cam_weights.push_back(cam_results[i].weight);
            }

            int n_valid = (int)valid_indices.size();
            cv::Mat out_accum = cv::Mat::zeros(out_h, out_w, CV_32FC1);
            cv::Mat out_count = cv::Mat::zeros(out_h, out_w, CV_32FC1);

            for (int y = 0; y < out_h; y++) {
                for (int x = 0; x < out_w; x++) {
                    // Collect per-camera mean heights for this cell
                    struct CellEntry { float mean_h; int idx; };
                    std::vector<CellEntry> entries;
                    entries.reserve(n_valid);
                    for (int c = 0; c < n_valid; c++) {
                        float cnt = cam_counts[c].at<float>(y, x);
                        if (cnt > 0.5f) {
                            entries.push_back({cam_heights[c].at<float>(y, x) / cnt, c});
                        }
                    }

                    if (entries.size() < 3) {
                        // Not enough cameras for outlier detection — weighted sum
                        for (auto const& e : entries) {
                            float w = cam_weights[e.idx];
                            out_accum.at<float>(y, x) += cam_heights[e.idx].at<float>(y, x) * w;
                            out_count.at<float>(y, x) += cam_counts[e.idx].at<float>(y, x) * w;
                        }
                    } else {
                        // Compute median height
                        std::sort(entries.begin(), entries.end(),
                                  [](auto const& a, auto const& b){ return a.mean_h < b.mean_h; });
                        float median = entries[entries.size() / 2].mean_h;

                        // Accumulate cameras within threshold of median
                        for (auto const& e : entries) {
                            if (std::abs(e.mean_h - median) <= outlier_thresh) {
                                float w = cam_weights[e.idx];
                                out_accum.at<float>(y, x) += cam_heights[e.idx].at<float>(y, x) * w;
                                out_count.at<float>(y, x) += cam_counts[e.idx].at<float>(y, x) * w;
                            }
                        }
                    }
                }
            }

            cv::Mat merged[2] = {out_accum, out_count};
            cv::merge(merged, 2, combined_data);
        }

        auto unique_msg = std::make_unique<sensor_msgs::msg::Image>();
        cv_bridge::CvImage(std_msgs::msg::Header(), "32FC2", combined_data).toImageMsg(*unique_msg);
        pub_->publish(std::move(unique_msg));
    }

    std::mutex data_mutex_;
    std::mutex weight_mutex_;
    std::map<std::string, std::shared_ptr<CameraPipeline>> pipelines_;
    std::map<std::string, sensor_msgs::msg::Image::SharedPtr> latest_frames_;
    std::map<std::string, float> camera_weights_;   // serial → quality weight
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_;
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    rclcpp::TimerBase::SharedPtr discovery_timer_, process_timer_, weight_timer_;
    rclcpp::CallbackGroup::SharedPtr timer_cb_group_;
};
} // namespace sandbox_components
RCLCPP_COMPONENTS_REGISTER_NODE(sandbox_components::CameraReprojector)

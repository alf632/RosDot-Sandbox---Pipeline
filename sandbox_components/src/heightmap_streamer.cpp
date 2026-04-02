#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "rclcpp/rclcpp.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>

namespace sandbox_components {

class HeightmapStreamer : public rclcpp::Node {
public:
    explicit HeightmapStreamer(const rclcpp::NodeOptions & options) 
    : Node("heightmap_streamer", rclcpp::NodeOptions(options).use_intra_process_comms(true)) {
        this->declare_parameter("udp_ip", "127.0.0.1");
        this->declare_parameter("udp_port", 5005);
        this->declare_parameter("z_offset", 0.25);
        this->declare_parameter("z_range", 0.50);
        // Which heightmap topic to stream.
        // Use "/sandbox/heightmap" (visual, default) or "/sandbox/heightmap/physics" (smooth, for water sim).
        this->declare_parameter("source_topic", std::string("/sandbox/heightmap"));

        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        int send_buff = 1024 * 1024;
        setsockopt(sock_, SOL_SOCKET, SO_SNDBUF, &send_buff, sizeof(send_buff));

        memset(&servaddr_, 0, sizeof(servaddr_));
        servaddr_.sin_family = AF_INET;
        servaddr_.sin_port = htons(this->get_parameter("udp_port").as_int());
        servaddr_.sin_addr.s_addr = inet_addr(this->get_parameter("udp_ip").as_string().c_str());

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            this->get_parameter("source_topic").as_string(),
            rclcpp::SensorDataQoS(), std::bind(&HeightmapStreamer::image_callback, this, std::placeholders::_1));
    }
    ~HeightmapStreamer() { if (sock_ >= 0) close(sock_); }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "32FC1");
            cv::Mat gray_img;

            float offset = this->get_parameter("z_offset").as_double();
            float range  = this->get_parameter("z_range").as_double();
            // Map [-offset, range-offset] to [0, 255]
            float alpha = 255.0f / range, beta = offset * alpha;
            cv_ptr->image.convertTo(gray_img, CV_8UC1, alpha, beta);

            std::vector<uchar> buf;
            cv::imencode(".png", gray_img, buf, {cv::IMWRITE_PNG_COMPRESSION, 3});
            sendto(sock_, buf.data(), buf.size(), 0, (const struct sockaddr *)&servaddr_, sizeof(servaddr_));
        } catch (...) { RCLCPP_ERROR(this->get_logger(), "Stream Error"); }
    }

    int sock_;
    struct sockaddr_in servaddr_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};
} // namespace sandbox_components
RCLCPP_COMPONENTS_REGISTER_NODE(sandbox_components::HeightmapStreamer)

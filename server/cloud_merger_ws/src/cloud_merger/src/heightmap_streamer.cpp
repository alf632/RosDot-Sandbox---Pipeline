#include <memory>
#include <string>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp"
#include <opencv2/opencv.hpp>

class HeightmapStreamer : public rclcpp::Node {
public:
    HeightmapStreamer() : Node("heightmap_streamer") {
        this->declare_parameter("udp_ip", "127.0.0.1");
        this->declare_parameter("udp_port", 5005);
        this->declare_parameter("input_topic", "sandbox/heightmap");
	this->declare_parameter("height_offset", 0.25f);
	this->declare_parameter("height_range", 0.5f);

        // Socket setup
        sock_ = socket(AF_INET, SOCK_DGRAM, 0);
        
        // Increase socket send buffer for large PNGs just in case
        int send_buff = 1024 * 1024; // 1MB
        setsockopt(sock_, SOL_SOCKET, SO_SNDBUF, &send_buff, sizeof(send_buff));

        memset(&servaddr_, 0, sizeof(servaddr_));
        servaddr_.sin_family = AF_INET;
        servaddr_.sin_port = htons(get_parameter("udp_port").as_int());
        servaddr_.sin_addr.s_addr = inet_addr(get_parameter("udp_ip").as_string().c_str());

        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            get_parameter("input_topic").as_string(),
            10,
            std::bind(&HeightmapStreamer::image_callback, this, std::placeholders::_1)
        );

        RCLCPP_INFO(this->get_logger(), "Streaming PNGs to %s:%ld", 
            get_parameter("udp_ip").as_string().c_str(), 
            get_parameter("udp_port").as_int());
    }

    ~HeightmapStreamer() {
        if (sock_ >= 0) close(sock_);
    }

private:
    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg) {
        try {
            // 1. Convert ROS image to OpenCV
            cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "32FC1");

	    float offset = get_parameter("height_offset").as_double();
            float range = get_parameter("height_range").as_double();; // total height of sandbox
            float alpha = 255.0f / range;
            float beta = offset * alpha;
            
            // 2. Normalize to 8-bit (0-255) or 16-bit (0-65535) for PNG
            // Godot's load_png_from_buffer works best with standard grayscale.
            cv::Mat gray_img;
            
            // Scale by 255 to fill the 8-bit range.
            cv_ptr->image.convertTo(gray_img, CV_8UC1, alpha, beta);

            // 3. Encode to PNG in memory
            std::vector<uchar> buf;
            std::vector<int> params = {cv::IMWRITE_PNG_COMPRESSION, 3}; // Low compression (3) = faster CPU
            cv::imencode(".png", gray_img, buf, params);

            // 4. Send as a single UDP packet
            // Note: Max UDP size is technically 65,507 bytes. 
            // A 600x400 grayscale PNG is typically 10KB - 40KB, which fits!
            if (buf.size() > 65507) {
                RCLCPP_WARN(this->get_logger(), "PNG too large for UDP: %zu bytes", buf.size());
                return;
            }

            sendto(sock_, buf.data(), buf.size(), 0,
                   (const struct sockaddr *)&servaddr_, sizeof(servaddr_));

        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
        }
    }

    int sock_;
    struct sockaddr_in servaddr_;
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
};

int main(int argc, char ** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<HeightmapStreamer>());
    rclcpp::shutdown();
    return 0;
}

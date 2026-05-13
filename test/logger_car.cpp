#include <ros/ros.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <signal.h>
#include <fstream>
#include <iomanip>
#include <vector>
#include <string>

static bool g_shutdown = false;
static void sigint_handler(int) { g_shutdown = true; ros::shutdown(); }

class CarLogger {
public:
    CarLogger(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
        pnh.param<std::string>("output_filename", output_filename_, "/tmp/car_output.csv");
        std::string topic;
        pnh.param<std::string>("topic", topic, "/car_output");
        sub_ = nh.subscribe(topic, 200, &CarLogger::callback, this);
        ROS_INFO("[CarLogger] Subscribing to %s → %s", topic.c_str(), output_filename_.c_str());
    }

    ~CarLogger() { save(); }

private:
    void callback(const rosneuro_msgs::NeuroFrame::ConstPtr& msg) {
        if (first_seq_ < 0) {
            first_seq_    = (int)msg->header.seq;
            n_channels_   = msg->eeg.info.nchannels;
            ROS_INFO("[CarLogger] first_seq=%d  nchannels=%d", first_seq_, n_channels_);
        }
        flat_data_.insert(flat_data_.end(), msg->eeg.data.begin(), msg->eeg.data.end());
    }

    void save() {
        if (flat_data_.empty()) { ROS_WARN("[CarLogger] No data to save."); return; }

        int total_samples = (int)flat_data_.size() / n_channels_;

        std::ofstream f(output_filename_);
        if (!f) { ROS_ERROR("[CarLogger] Cannot open %s", output_filename_.c_str()); return; }
        // 9 significant digits covers float32's full ~7.2-digit precision
        f << std::setprecision(9);
        for (int r = 0; r < total_samples; r++) {
            for (int c = 0; c < n_channels_; c++) {
                if (c > 0) f << ",";
                f << flat_data_[r * n_channels_ + c];
            }
            f << "\n";
        }
        ROS_INFO("[CarLogger] Saved %d x %d to %s", total_samples, n_channels_, output_filename_.c_str());

        std::string seq_file = output_filename_.substr(0, output_filename_.rfind(".csv")) + "_first_seq.txt";
        std::ofstream fs(seq_file);
        fs << first_seq_ << "\n";
        ROS_INFO("[CarLogger] Saved first_seq=%d to %s", first_seq_, seq_file.c_str());
    }

    ros::Subscriber sub_;
    std::string output_filename_;
    std::vector<float> flat_data_;
    int n_channels_ = 0;
    int first_seq_  = -1;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "car_logger", ros::init_options::NoSigintHandler);
    signal(SIGINT, sigint_handler);
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    CarLogger logger(nh, pnh);
    ros::spin();
    return 0;
}

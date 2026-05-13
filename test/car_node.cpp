#include <ros/ros.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <Eigen/Dense>
#include "rosneuro_filters_car/Car.hpp"

class CarNode {
public:
    CarNode(ros::NodeHandle& nh, ros::NodeHandle& pnh) {
        pnh.param<std::vector<std::string>>("EOG_ch_names", eog_ch_names_, {});
        std::string topic_out;
        pnh.param<std::string>("topic_out", topic_out, "/car_output");

        sub_ = nh.subscribe("/neurodata", 1, &CarNode::callback, this);
        pub_ = nh.advertise<rosneuro_msgs::NeuroFrame>(topic_out, 1);
        ROS_INFO("[CarNode] EOG_ch_names: %zu channel(s), publishing to %s",
                 eog_ch_names_.size(), topic_out.c_str());
    }

private:
    void callback(const rosneuro_msgs::NeuroFrame::ConstPtr& msg) {
        if (!labels_resolved_) {
            std::vector<std::string> labels(msg->eeg.info.labels.begin(),
                                            msg->eeg.info.labels.end());
            car_.configure(labels, eog_ch_names_);
            labels_resolved_ = true;
            ROS_INFO("[CarNode] Labels resolved. nchannels=%d nsamples=%d",
                     msg->eeg.info.nchannels, msg->eeg.info.nsamples);
        }

        int ns = msg->eeg.info.nsamples;
        int nc = msg->eeg.info.nchannels;
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
            in(msg->eeg.data.data(), ns, nc);

        rosneuro::DynamicMatrix<float> filtered = car_.apply(in.cast<float>());

        rosneuro_msgs::NeuroFrame out = *msg;
        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out_rm = filtered;
        out.eeg.data.assign(out_rm.data(), out_rm.data() + out_rm.size());
        pub_.publish(out);
    }

    ros::Subscriber sub_;
    ros::Publisher  pub_;
    rosneuro::Car<float> car_;
    std::vector<std::string> eog_ch_names_;
    bool labels_resolved_ = false;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "car_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    CarNode node(nh, pnh);
    ros::spin();
    return 0;
}

#include <ros/ros.h>
#include <rosneuro_msgs/NeuroFrame.h>
#include <Eigen/Dense>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

static const std::vector<std::string> CH_LABELS_32 = {
    "Fp1","Fp2","F3","Fz","F4","FC1","FC2","C3","Cz","C4",
    "CP1","CP2","P3","Pz","P4","POz","O1","O2","CPz","F1",
    "F2","FC5","FCz","FC6","C1","C2","CP5","CP6","P5","P1","P2","P6"
};

static Eigen::MatrixXd readCSV(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open: " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        std::vector<double> row;
        std::string tok;
        while (std::getline(ss, tok, ','))
            row.push_back(std::stod(tok));
        if (!row.empty()) rows.push_back(row);
    }
    Eigen::MatrixXd m(rows.size(), rows[0].size());
    for (int r = 0; r < (int)rows.size(); r++)
        for (int c = 0; c < (int)rows[0].size(); c++)
            m(r, c) = rows[r][c];
    return m;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "test_publisher_car");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    std::string csv_filename;
    int n_samples;
    double sample_rate;

    if (!pnh.getParam("csv_file", csv_filename))   { ROS_ERROR("'csv_file' not set.");    return 1; }
    if (!pnh.getParam("chunk_size", n_samples))    { ROS_ERROR("'chunk_size' not set.");  return 1; }
    if (!pnh.getParam("sample_rate", sample_rate)) { ROS_ERROR("'sample_rate' not set."); return 1; }

    ROS_INFO("Loading: %s", csv_filename.c_str());
    Eigen::MatrixXd full_data;
    try { full_data = readCSV(csv_filename); }
    catch (const std::exception& e) { ROS_ERROR("CSV error: %s", e.what()); return 1; }

    int n_channels    = full_data.cols();
    int total_samples = full_data.rows();
    ROS_INFO("Loaded: %d samples x %d channels.", total_samples, n_channels);

    std::vector<std::string> ch_labels(CH_LABELS_32.begin(),
                                       CH_LABELS_32.begin() + std::min(n_channels, (int)CH_LABELS_32.size()));
    for (int i = (int)ch_labels.size(); i < n_channels; i++)
        ch_labels.push_back("Ch" + std::to_string(i + 1));

    ros::Publisher pub = nh.advertise<rosneuro_msgs::NeuroFrame>("/neurodata", 1);
    ros::Rate loop_rate(sample_rate / n_samples);

    ROS_INFO("Waiting for subscriber on /neurodata ...");
    while (ros::ok() && pub.getNumSubscribers() == 0) {
        ros::Duration(0.5).sleep();
        ROS_INFO_THROTTLE(5.0, "Still waiting...");
    }
    ROS_INFO("Subscriber connected. Starting publication.");

    int current_sample = 0;
    uint32_t seq = 0;
    while (ros::ok()) {
        if (current_sample + n_samples > total_samples) { ROS_INFO("End of CSV file."); break; }

        rosneuro_msgs::NeuroFrame msg;
        msg.header.stamp       = ros::Time::now();
        msg.header.seq         = seq;
        msg.sr                 = sample_rate;
        msg.eeg.info.nchannels = n_channels;
        msg.eeg.info.nsamples  = n_samples;
        msg.eeg.info.labels    = ch_labels;

        Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> chunk =
            full_data.block(current_sample, 0, n_samples, n_channels).cast<float>();
        msg.eeg.data.assign(chunk.data(), chunk.data() + chunk.size());

        pub.publish(msg);
        ROS_INFO_THROTTLE(1.0, "Published seq %u", seq);

        current_sample += n_samples;
        seq++;
        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}

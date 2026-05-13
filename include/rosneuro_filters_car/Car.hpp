#ifndef ROSNEURO_FILTERS_CAR_HPP
#define ROSNEURO_FILTERS_CAR_HPP

#include <Eigen/Dense>
#include <gtest/gtest_prod.h>
#include <vector>
#include <string>
#include <algorithm>
#include <rosneuro_filters/Filter.hpp>

namespace rosneuro {
    template <typename T>
    class Car : public Filter<T> {
        public:
            Car(void);
            ~Car(void) {};

            bool configure(void);
            bool configure(const std::string& param_name) {
                return Filter<T>::configure(param_name);
            }

            bool configure(const std::vector<int>& eog_ch) {
                this->eog_ch_excl_ = eog_ch;
                this->mask_excl_.clear();
                return true;
            }

            // Resolve EOG channel names to indices using the provided label list
            bool configure(const std::vector<std::string>& ch_labels,
                           const std::vector<std::string>& eog_names) {
                this->eog_ch_names_ = eog_names;
                this->eog_ch_excl_.clear();
                this->mask_excl_.clear();
                for (const auto& name : eog_names) {
                    for (int i = 0; i < (int)ch_labels.size(); i++) {
                        std::string a = name, b = ch_labels[i];
                        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
                        std::transform(b.begin(), b.end(), b.begin(), ::tolower);
                        if (a == b) { this->eog_ch_excl_.push_back(i); break; }
                    }
                }
                return true;
            }

            DynamicMatrix<T> apply(const DynamicMatrix<T>& in);
            FRIEND_TEST(CarTestSuite, TestCarName);

        private:
            std::vector<int>         eog_ch_excl_;
            std::vector<bool>        mask_excl_;
            std::vector<std::string> eog_ch_names_;
    };

    template<typename T>
    Car<T>::Car(void) {
        this->name_ = "car";
    }

    template<typename T>
    bool Car<T>::configure(void) {
        this->mask_excl_.clear();
        this->eog_ch_excl_.clear();

        std::vector<std::string> eog_ch_names;
        if (Filter<T>::getParam(std::string("EOG_ch_names"), eog_ch_names)) {
            this->eog_ch_names_ = eog_ch_names;
            ROS_INFO("[%s] EOG_ch_names loaded (%zu channels, name-based – call configure(labels,names) to resolve)",
                     this->name().c_str(), eog_ch_names.size());
            return true;
        }

        std::vector<double> eog_ch;
        if (Filter<T>::getParam(std::string("EOG_ch"), eog_ch)) {
            this->eog_ch_excl_.resize(eog_ch.size());
            for (size_t i = 0; i < eog_ch.size(); i++)
                this->eog_ch_excl_[i] = static_cast<int>(eog_ch[i] - 1);
            ROS_INFO("[%s] EOG_ch loaded (%zu channels, index-based)", this->name().c_str(), eog_ch.size());
            return true;
        }

        ROS_WARN("[%s] Neither EOG_ch_names nor EOG_ch found – CAR applied to all channels",
                 this->name().c_str());
        return true;
    }

    template<typename T>
    DynamicMatrix<T> Car<T>::apply(const DynamicMatrix<T>& in) {
        long n_samples = in.rows();
        long n_channels = in.cols();

        if (this->eog_ch_excl_.empty()) {
            return in - (in.rowwise().mean()).replicate(1, n_channels);
        }

        if (this->mask_excl_.size() != n_channels) {
            this->mask_excl_.assign(n_channels, false);
            for (int idx : this->eog_ch_excl_) {
                if (idx >= 0 && idx < n_channels) {
                    this->mask_excl_[idx] = true;
                }
            }
        }

        Eigen::Matrix<T, Eigen::Dynamic, 1> common_average = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(n_samples);
        int valid_channels_count = 0;

        for (long i = 0; i < n_channels; ++i) {
            if (!this->mask_excl_[i]) {
                common_average += in.col(i);
                valid_channels_count++;
            }
        }

        if (valid_channels_count > 0) {
            common_average /= static_cast<T>(valid_channels_count);
        } else {
            return in;
        }

        DynamicMatrix<T> out = in; 

        for (long i = 0; i < n_channels; ++i) {
            out.col(i) -= common_average;
        }
        
        return out;
    }
}

#endif
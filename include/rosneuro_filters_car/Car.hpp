#ifndef ROSNEURO_FILTERS_CAR_HPP
#define ROSNEURO_FILTERS_CAR_HPP

#include <Eigen/Dense>
#include <gtest/gtest_prod.h>
#include <vector>
#include <algorithm>
#include <rosneuro_filters/Filter.hpp>

namespace rosneuro {
    template <typename T>
    class Car : public Filter<T> {
        public:
            Car(void);
            ~Car(void) {};

            bool configure(void);
            bool configure(std::vector<int> channels_exclude); 
            
            DynamicMatrix<T> apply(const DynamicMatrix<T>& in);
            FRIEND_TEST(CarTestSuite, TestCarName);

        private:
            std::vector<int> eog_ch_excl_;
            std::vector<bool> mask_excl_; 
    };

    template<typename T>
    Car<T>::Car(void) {
        this->name_ = "car";
    }

    template<typename T>
    bool Car<T>::configure(void) {
        this->eog_ch_excl_.clear();
        this->mask_excl_.clear();
        return true;
    }

    template<typename T>
    bool Car<T>::configure(std::vector<int> channels_exclude) {
        this->eog_ch_excl_ = channels_exclude;
        this->mask_excl_.clear();

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
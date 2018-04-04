
#include "featureMatching.h"

//STL
#include <functional>
#include <future>
#include <stdexcept>
#include <vector>

//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "config.h"

namespace StVO {

int matchNNR(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12) {

    int matches = 0;
    matches_12.resize(desc1.rows, -1);

    std::vector<std::vector<cv::DMatch>> matches_;
    cv::Ptr<cv::BFMatcher> bfm = cv::BFMatcher::create(cv::NORM_HAMMING, false); // cross-check
    bfm->knnMatch(desc1, desc2, matches_, 2);

    if (desc1.rows != matches_.size())
        throw std::runtime_error("[MapHandler->matchNNR] Different size for matches and descriptors!");

    for (int idx = 0; idx < desc1.rows; ++idx) {
        if (matches_[idx][0].distance < matches_[idx][1].distance * nnr) {
            matches_12[idx] = matches_[idx][0].trainIdx;
            matches++;
        }
    }

    return matches;
}

int match(const cv::Mat &desc1, const cv::Mat &desc2, float nnr, std::vector<int> &matches_12) {

    matches_12.clear();
    if (Config::bestLRMatches()) {
        int matches;
        std::vector<int> matches_21;
        if (Config::lrInParallel()) {
            auto match_12 = std::async(std::launch::async, &matchNNR,
                                  std::cref(desc1), std::cref(desc2), nnr, std::ref(matches_12));
            auto match_21 = std::async(std::launch::async, &matchNNR,
                                  std::cref(desc2), std::cref(desc1), nnr, std::ref(matches_21));
            matches = match_12.get();
            match_21.wait();
        } else {
            matches = matchNNR(desc1, desc2, nnr, matches_12);
            matchNNR(desc2, desc1, nnr, matches_21);
        }

        for (int i1 = 0; i1 < matches_12.size(); ++i1) {
            int &i2 = matches_12[i1];
            if (i2 >= 0 && matches_21[i2] != i1) {
                i2 = -1;
                matches--;
            }
        }

        return matches;
    } else
        return matchNNR(desc1, desc2, nnr, matches_12);
}

} // namesapce StVO

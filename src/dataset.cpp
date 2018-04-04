/*****************************************************************************
**      Stereo VO and SLAM by combining point and line segment features     **
******************************************************************************
**                                                                          **
**  Copyright(c) 2016-2018, Ruben Gomez-Ojeda, University of Malaga         **
**  Copyright(c) 2016-2018, David Zuñiga-Noël, University of Malaga         **
**  Copyright(c) 2016-2018, MAPIR group, University of Malaga               **
**                                                                          **
**  This program is free software: you can redistribute it and/or modify    **
**  it under the terms of the GNU General Public License (version 3) as     **
**  published by the Free Software Foundation.                              **
**                                                                          **
**  This program is distributed in the hope that it will be useful, but     **
**  WITHOUT ANY WARRANTY; without even the implied warranty of              **
**  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the            **
**  GNU General Public License for more details.                            **
**                                                                          **
**  You should have received a copy of the GNU General Public License       **
**  along with this program.  If not, see <http://www.gnu.org/licenses/>.   **
**                                                                          **
*****************************************************************************/

#include "dataset.h"

//STL
#include <algorithm>
#include <functional>
#include <limits>
#include <list>
#include <stdexcept>
#include <string>

//Boost
#include <boost/regex.hpp> //Note: using boost regex instead of C++11 regex as it isn't supported by the compiler until gcc 4.9
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>

//OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

//YAML
#include <yaml-cpp/yaml.h>

#include "pinholeStereoCamera.h"

namespace StVO {

void getSortedImages(const boost::filesystem::path &img_dir, std::function<bool(const std::string &)> filter,
                     std::function<bool(const std::string &, const std::string &)> comparator, std::vector<std::string> &img_paths) {

    // get a sorted list of files in the img directories
    if (!boost::filesystem::exists(img_dir) ||
            !boost::filesystem::is_directory(img_dir))
        throw std::runtime_error("[Dataset] Invalid images subfolder");

    // get all files in the img directories
    std::list<std::string> all_imgs;
    for(auto& entry : boost::make_iterator_range(boost::filesystem::directory_iterator(img_dir), {})) {
        boost::filesystem::path filename_path = entry.path().filename();
        if (boost::filesystem::is_regular_file(entry.status()) &&
                (filename_path.extension() == ".png"  ||
                 filename_path.extension() == ".jpg"  ||
                 filename_path.extension() == ".jpeg" ||
                 filename_path.extension() == ".pnm"  ||
                 filename_path.extension() == ".tiff")) {
            all_imgs.push_back(filename_path.string());
        }
    }

    // sort
    img_paths.clear();
    img_paths.reserve(all_imgs.size());
    for (const std::string &filename : all_imgs)
        if (!filter(filename)) img_paths.push_back(filename);

    if (img_paths.empty())
        throw std::runtime_error("[Dataset] Invalid image names?");

    std::sort(img_paths.begin(), img_paths.end(), comparator);

    for (std::string &filename : img_paths)
        filename = (img_dir / filename).string();
}

Dataset::Dataset(const std::string &dataset_path, const PinholeStereoCamera &cam, int offset, int nmax, int step)
    : cam(cam) {

    boost::filesystem::path dataset_base(dataset_path);
    if (!boost::filesystem::exists(dataset_base) ||
            !boost::filesystem::is_directory(dataset_base))
        throw std::runtime_error("[Dataset] Invalid directory");

    boost::filesystem::path dataset_params = dataset_base / "dataset_params.yaml";
    if (!boost::filesystem::exists(dataset_params))
        throw std::runtime_error("[Dataset] Dataset parameters not found");
    YAML::Node dataset_config = YAML::LoadFile(dataset_params.string());

    // setup image directories
    boost::filesystem::path img_l_dir = dataset_base / dataset_config["images_subfolder_l"].as<std::string>();
    boost::filesystem::path img_r_dir = dataset_base / dataset_config["images_subfolder_r"].as<std::string>();

    boost::regex expression("^[^0-9]*([0-9]+\\.?+[0-9]*)[^0-9]*\\.[a-z]{3,4}$");
    boost::cmatch what;

    auto filename_filter = [&expression, &what](const std::string &s) {
        return !boost::regex_match(s.c_str(), what, expression);
    };

    auto sort_by_number = [&expression, &what](const std::string &a, const std::string &b) {
        double n1, n2;

        if (boost::regex_match(a.c_str(), what, expression))
            n1 = std::stod(what[1]);
        else
            throw std::runtime_error("[Dataset] Unexpected behaviour while sorting filenames");

        if (boost::regex_match(b.c_str(), what, expression))
            n2 = std::stod(what[1]);
        else
            throw std::runtime_error("[Dataset] Unexpected behaviour while sorting filenames");

        return (n1 < n2);
    };

    std::vector<std::string> img_l_paths, img_r_paths;
    getSortedImages(img_l_dir, filename_filter, sort_by_number, img_l_paths);
    getSortedImages(img_r_dir, filename_filter, sort_by_number, img_r_paths);

    if (img_l_paths.size() != img_r_paths.size())
        throw std::runtime_error("[Dataset] Left and right images");

    // decimate sequence
    offset = std::max(0, offset);
    nmax = (nmax <= 0) ? std::numeric_limits<int>::max() : nmax;
    step = std::max(1, step);
    for (int i = 0, ctr = 0; (i + offset) < img_l_paths.size() && ctr < nmax; i += step, ctr++) {
        images_l.push_back(img_l_paths[i + offset]);
        images_r.push_back(img_r_paths[i + offset]);
    }
}

Dataset::~Dataset() {

}

bool Dataset::nextFrame(cv::Mat &img_l, cv::Mat &img_r) {
    if (!hasNext()) return false;

    img_l = cv::imread(images_l.front(), CV_LOAD_IMAGE_UNCHANGED);
    img_r = cv::imread(images_r.front(), CV_LOAD_IMAGE_UNCHANGED);
    cam.rectifyImagesLR(img_l, img_l, img_r, img_r);
    images_l.pop_front();
    images_r.pop_front();

    return (!img_l.empty() && !img_r.empty());
}

bool Dataset::hasNext() {
    return !(images_l.empty() || images_r.empty());
}

} // namespace StVO


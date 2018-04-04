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

#pragma once

//STL
#include <list>
#include <string>

//OpenCV
#include <opencv2/core.hpp>

#include "pinholeStereoCamera.h"

namespace StVO {

class Dataset {
public:

    // Constructor
    Dataset(const std::string &dataset_path, const PinholeStereoCamera &cam, int offset = 0, int nmax = 0, int step = 1);

    // Destrcutor
    virtual ~Dataset();

    // Reads next frame in the dataset sequence, returning true if image was successfully loaded
    bool nextFrame(cv::Mat &img_l, cv::Mat &img_r);

    // Returns if there are images still available in the sequence
    inline bool hasNext();

private:

    std::list<std::string> images_l, images_r;
    const PinholeStereoCamera &cam;
};

} // namespace StVO


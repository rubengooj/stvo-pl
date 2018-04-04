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

#include "timer.h"

//STL
#include <stdexcept>
#include <chrono>

namespace StVO {

Timer::Timer(double scale) : started(false), scale(scale) { }
Timer::~Timer() { }

void Timer::start() {

    started = true;
    start_t = std::chrono::high_resolution_clock::now();
}

double Timer::stop() {

    std::chrono::high_resolution_clock::time_point end_t = std::chrono::high_resolution_clock::now();

    if (!started)
        throw std::logic_error("[Timer] Stop called without previous start");

    started = false;
    std::chrono::duration<double, std::nano> elapsed_ns = end_t - start_t;
    return elapsed_ns.count()*scale;
}

} // namespace StVO

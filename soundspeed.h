#include "hip/hip_runtime.h"
/**
 * @author      Christoph Schaefer cm.schaefer@gmail.com
 *
 * @section     LICENSE
 * Copyright (c) 2019 Christoph Schaefer
 *
 * This file is part of miluphcuda.
 *
 * miluphcuda is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * miluphcuda is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with miluphcuda.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef _SOUNDSPEED_H
#define _SOUNDSPEED_H


/**
 * @brief Compute sound speed depending on equation of state.
 * @details For materials with constant sound speed it is set in `calculateSoundSpeed()`.
 */
__global__ void calculateSoundSpeed(void);

/**
 * @brief Initialize sound speed for all materials.
 */
__global__ void initializeSoundspeed(void);

#endif

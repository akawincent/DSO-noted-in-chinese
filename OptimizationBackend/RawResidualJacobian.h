/*
 * @Author: akawincent 3511606256@qq.com
 * @Date: 2023-02-08 09:49:36
 * @LastEditors: akawincent 3511606256@qq.com
 * @LastEditTime: 2023-02-19 18:28:05
 * @FilePath: \DSO github\DSO-noted-in-chinese\OptimizationBackend\RawResidualJacobian.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

 
#include "util/NumType.h"

namespace dso
{
struct RawResidualJacobian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// ================== new structure: save independently =============.
	VecNRf resF;	//对应r21  这里是1*8的向量 把一个点的邻域pattern内每个点的残差都保存了

	// the two rows of d[x,y]/d[xi].
	//x2像素坐标对T21求导
	Vec6f Jpdxi[2];			// 2x6

	// the two rows of d[x,y]/d[C].
	//x2像素坐标对相机内参求导
	VecCf Jpdc[2];			// 2x4

	// the two rows of d[x,y]/d[idepth].
	//x2像素坐标对逆深度求导(这里的逆深度是针对host frame的)
	Vec2f Jpdd;				// 2x1

	// the two columns of d[r]/d[x,y].
	//r21对x2像素坐标求导 r21是1*8向量
	VecNRf JIdx[2];			// 8x2

	// = the two columns of d[r] / d[ab]
	//r21对相对光度参数a21和b21求导 
	//JabF[0] = dr21/da21 JabF[1] = dr21/db21
	VecNRf JabF[2];			// 8x2

	// = JIdx^T * JIdx (inner product). Only as a shorthand.
	//dr21/dx2^T * dr21/dx2 = 2x8 * 8x2 = 2x2
	Mat22f JIdx2;				// 2x2

	// = Jab^T * JIdx (inner product). Only as a shorthand.
	//JabF^T * dr21/dx2 = 2x8 * 8x2 = 2x2
	Mat22f JabJIdx;			// 2x2

	// = Jab^T * Jab (inner product). Only as a shorthand.
	//JabF^T * JabF = 2x8 * 8x2 = 2x2
	Mat22f Jab2;			// 2x2

};
}


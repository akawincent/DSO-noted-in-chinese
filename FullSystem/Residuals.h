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

 
#include "util/globalCalib.h"
#include "vector"
 
#include "util/NumType.h"
#include <iostream>
#include <fstream>
#include "util/globalFuncs.h"
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{
class PointHessian;
class FrameHessian;
class CalibHessian;

class EFResidual;

//枚举优化状态变量的类型:激活 FEJ固定点 边缘化
enum ResLocation {ACTIVE=0, LINEARIZED, MARGINALIZED, NONE};
//point的枚举类型:好点 OOB OUTLIER
enum ResState {IN=0, OOB, OUTLIER};

struct FullJacRowT
{
	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
};

/********************* 关键帧优化中的残差类 ****************************/
class PointFrameResidual
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	//存着EFResidual对象
	EFResidual* efResidual;
	//实例化计数器
	static int instanceCounter;
	//优化中当前状态量和更新后状态量的类型
	ResState state_NewState;
	ResState state_state;
	//优化中的能量值
	double state_energy;
	double state_NewEnergy;
	double state_NewEnergyWithOutlier;

	//设置状态量的类型
	void setState(ResState s) {state_state = s;}

	//持有激活点
	PointHessian* point;
	//持有构成该残差的主导帧
	FrameHessian* host;
	//持有构成该残差的目标帧
	FrameHessian* target;
	//优化用到导数都存在J里
	RawResidualJacobian* J;

	bool isNew;

	//host帧上的8-pattern邻域点投影到target帧上后的像素坐标
	Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
	//host帧上的点投影到target帧上后的像素坐标以及相对于target帧上的逆深度
	Vec3f centerProjectedTo;

	~PointFrameResidual();
	PointFrameResidual();
	PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_);
	double linearize(CalibHessian* HCalib);

	//重置状态量
	void resetOOB()
	{	
		//能量设为0
		state_NewEnergy = state_energy = 0;
		//状态量更新值设为OUTLIER
		state_NewState = ResState::OUTLIER;
		//当前状态量设为好点
		setState(ResState::IN);
	};
	void applyRes( bool copyJacobians);

	void debugPlot();

	void printRows(std::vector<VecX> &v, VecX &r, int nFrames, int nPoints, int M, int res);
};
}


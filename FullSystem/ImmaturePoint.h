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
 
#include "FullSystem/HessianBlocks.h"
namespace dso
{


struct ImmaturePointTemporaryResidual
{
public:
	ResState state_state;
	double state_energy;
	ResState state_NewState;
	double state_NewEnergy;
	FrameHessian* target;
};

//点的状态
enum ImmaturePointStatus {
	//表示最近一次在极线上进行成功匹配
	IPS_GOOD=0,					// traced well and good
	//表示追踪结束，或者是追踪的点在投用时已经到图像外面了
	IPS_OOB,					// OOB: end tracking & marginalize!
	//表示追踪失败
	IPS_OUTLIER,				// energy too high: if happens again: outlier!
	//表示虽然成功追踪 但是视差太小(不利于深度滤波) 可以跳过
	IPS_SKIPPED,				// traced well and good (but not actually traced).
	//表示因为一些坏条件导致追踪失败
	IPS_BADCONDITION,			// not traced because of bad condition.
	//表示还没有进行过深度追踪
	IPS_UNINITIALIZED			// not even traced once.
};			


class ImmaturePoint
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	//static values
	//MAX_RES_PER_POINT = 8
	//存储该像素邻域8个点的像素值
	float color[MAX_RES_PER_POINT];
	//邻域8个像素点的权重wp
	float weights[MAX_RES_PER_POINT];
	
	//点以及其邻域点的梯度平方和
	Mat22f gradH;
	Vec2f gradH_ev;
	Mat22f gradH_eig;
	float energyTH;
	//像素坐标
	float u,v;
	//点的host frame
	FrameHessian* host;
	int idxInImmaturePoints;
	//特征点质量
	float quality;
	//选点时的种类
	float my_type;

	//逆深度的上下限度
	float idepth_min;
	float idepth_max;

	ImmaturePoint(int u_, int v_, FrameHessian* host_, float type, CalibHessian* HCalib);
	~ImmaturePoint();

	ImmaturePointStatus traceOn(FrameHessian* frame, const Mat33f &hostToFrame_KRKi, const Vec3f &hostToFrame_Kt, const Vec2f &hostToFrame_affine, CalibHessian* HCalib, bool debugPrint=false);

	ImmaturePointStatus lastTraceStatus; 	//上一次极线跟踪的点的状态
	Vec2f lastTraceUV;						//上一次极线跟踪的点的像素坐标
	float lastTracePixelInterval;			//上一次极线跟踪的不确定度

	float idepth_GT;

	double linearizeResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float &Hdd, float &bd,
			float idepth);
	float getdPixdd(
			CalibHessian *  HCalib,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

	float calcResidual(
			CalibHessian *  HCalib, const float outlierTHSlack,
			ImmaturePointTemporaryResidual* tmpRes,
			float idepth);

	private:
};

}


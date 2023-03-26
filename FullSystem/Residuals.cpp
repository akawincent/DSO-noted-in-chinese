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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
int PointFrameResidual::instanceCounter = 0;


long runningResID=0;


PointFrameResidual::PointFrameResidual(){assert(false); instanceCounter++;}

PointFrameResidual::~PointFrameResidual(){assert(efResidual==0); instanceCounter--; delete J;}

//PointFrameResidual初始化
PointFrameResidual::PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_) :
	point(point_),
	host(host_),
	target(target_)
{
	efResidual=0;
	instanceCounter++;
	resetOOB();
	//准备导数
	J = new RawResidualJacobian();
	assert(((long)J)%16==0);

	isNew=true;
}

/****************** 准备后端优化中使用的导数 *******************/
double PointFrameResidual::linearize(CalibHessian* HCalib)
{
	state_NewEnergyWithOutlier=-1;
	//残差当前状态为OOB 把新状态也设为OOB  直接返回当前状态的能量
	if(state_state == ResState::OOB)
		{ state_NewState = ResState::OOB; return state_energy; }

	//拿到这个残差对应的host和target帧之间的预先计算值 
	FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);

	float energyLeft=0;
	
	const Eigen::Vector3f* dIl = target->dI;				//目标帧的像素信息
	//const float* const Il = target->I;

	//优化中随着增量更新变化的
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;		//K*旋转*K^-1
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;			//K*平移

	//线性化点处固定的位姿
	const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;			//旋转部分	
	const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;			//平移部分

	const float * const color = point->color;				//像素灰度值
	const float * const weights = point->weights;			//像素的权重

	Vec2f affLL = precalc->PRE_aff_mode;					//相对灰度变换
	float b0 = precalc->PRE_b0_mode;						//光度参数b


	Vec6f d_xi_x, d_xi_y;
	Vec4f d_C_x, d_C_y;
	float d_d_x, d_d_y;
	//里面一系列的SCALE因子是干什么的
	{
		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		//如果从host到target的投影不成功  就把该residual设为OOB 直接返回他的能量
		//这里的投影用的是线性化点处固定的位姿
		if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
				PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{ state_NewState = ResState::OOB; return state_energy; }

		//(point在target帧上的像素横坐标,point在target帧上的像素纵坐标,point相对于target帧上的逆深度)
		centerProjectedTo = Vec3f(Ku, Kv, new_idepth);

		/*********** 所有的导数的数学推导参考:https://www.cnblogs.com/JingeTU/p/8395046.html *************/
		//这里的Jacobian都是线性化点处的导数 在一次优化中的所有迭代里都是不变的

		// diff d_idepth
		//dx2/dp1
		d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*SCALE_IDEPTH*HCalib->fxl();
		d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*SCALE_IDEPTH*HCalib->fyl();

		// diff calib
		//dx2/dC  这里的C是相机内参矩阵
		d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));
		d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) * HCalib->fyli();
		d_C_x[0] = KliP[0]*d_C_x[2];
		d_C_x[1] = KliP[1]*d_C_x[3];

		d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) * HCalib->fxli();
		d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
		d_C_y[0] = KliP[0]*d_C_y[2];
		d_C_y[1] = KliP[1]*d_C_y[3];

		d_C_x[0] = (d_C_x[0]+u)*SCALE_F;
		d_C_x[1] *= SCALE_F;
		d_C_x[2] = (d_C_x[2]+1)*SCALE_C;
		d_C_x[3] *= SCALE_C;

		d_C_y[0] *= SCALE_F;
		d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
		d_C_y[2] *= SCALE_C;
		d_C_y[3] = (d_C_y[3]+1)*SCALE_C;

		//dx2/dT21(T是李代数)
		d_xi_x[0] = new_idepth*HCalib->fxl();
		d_xi_x[1] = 0;
		d_xi_x[2] = -new_idepth*u*HCalib->fxl();
		d_xi_x[3] = -u*v*HCalib->fxl();
		d_xi_x[4] = (1+u*u)*HCalib->fxl();
		d_xi_x[5] = -v*HCalib->fxl();

		d_xi_y[0] = 0;
		d_xi_y[1] = new_idepth*HCalib->fyl();
		d_xi_y[2] = -new_idepth*v*HCalib->fyl();
		d_xi_y[3] = -(1+v*v)*HCalib->fyl();
		d_xi_y[4] = u*v*HCalib->fyl();
		d_xi_y[5] = u*HCalib->fyl();
	}

	//把上述计算结果赋给J
	{
		J->Jpdxi[0] = d_xi_x;
		J->Jpdxi[1] = d_xi_y;

		J->Jpdc[0] = d_C_x;
		J->Jpdc[1] = d_C_y;

		J->Jpdd[0] = d_d_x;
		J->Jpdd[1] = d_d_y;

	}

	float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
	float JabJIdx_00=0, JabJIdx_01=0, JabJIdx_10=0, JabJIdx_11=0;
	float JabJab_00=0, JabJab_01=0, JabJab_11=0;

	float wJI2_sum = 0;
	//考虑8-pattern邻域 构建残差
	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
		//这里的投影要用变化的位姿 需要让residual数值产生变化
		if(!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{ state_NewState = ResState::OOB; return state_energy; }

		projectedTo[idx][0] = Ku;
		projectedTo[idx][1] = Kv;

		//计算residual的数值  
        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);


		float drdA = (color[idx]-b0);
		if(!std::isfinite((float)hitColor[0]))
		{ state_NewState = ResState::OOB; return state_energy; }

		//邻域点的权重
		float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
        w = 0.5f*(w + weights[idx]);

		//Huber鲁棒核函数
		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		//能量
		energyLeft += w*w*hw *residual*residual*(2-hw);

		//这一部分导数是不需要FEJ的
		{
			if(hw < 1) hw = sqrtf(hw);
			hw = hw*w;

			hitColor[1]*=hw;
			hitColor[2]*=hw;

			//r21  这里是经过 hw加权的
			J->resF[idx] = residual*hw;

			//JIdx = dr21/dx2 = wh * [gx gy] 
			//图像梯度是不需要线性固定的
			J->JIdx[0][idx] = hitColor[1];
			J->JIdx[1][idx] = hitColor[2];

			//JabF[0] = dr21/da21 JabF[1] = dr21/db21
			J->JabF[0][idx] = drdA*hw;
			J->JabF[1][idx] = hw;

			JIdxJIdx_00+=hitColor[1]*hitColor[1];
			JIdxJIdx_11+=hitColor[2]*hitColor[2];
			JIdxJIdx_10+=hitColor[1]*hitColor[2];

			JabJIdx_00+= drdA*hw * hitColor[1];
			JabJIdx_01+= drdA*hw * hitColor[2];
			JabJIdx_10+= hw * hitColor[1];
			JabJIdx_11+= hw * hitColor[2];

			JabJab_00+= drdA*drdA*hw*hw;
			JabJab_01+= drdA*hw*hw;
			JabJab_11+= hw*hw;


			wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]);

			if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;
			if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;
		}
	}
	//给Hessian矩阵做准备
	J->JIdx2(0,0) = JIdxJIdx_00;
	J->JIdx2(0,1) = JIdxJIdx_10;
	J->JIdx2(1,0) = JIdxJIdx_10;
	J->JIdx2(1,1) = JIdxJIdx_11;
	J->JabJIdx(0,0) = JabJIdx_00;
	J->JabJIdx(0,1) = JabJIdx_01;
	J->JabJIdx(1,0) = JabJIdx_10;
	J->JabJIdx(1,1) = JabJIdx_11;
	J->Jab2(0,0) = JabJab_00;
	J->Jab2(0,1) = JabJab_01;
	J->Jab2(1,0) = JabJab_01;
	J->Jab2(1,1) = JabJab_11;

	state_NewEnergyWithOutlier = energyLeft;

	//能量太大了 就不要了
	if(energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2)
	{
		energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
		state_NewState = ResState::OUTLIER;
	}
	else
	{
		state_NewState = ResState::IN;
	}

	state_NewEnergy = energyLeft;
	//返回这个r的能量
	return energyLeft;
}



void PointFrameResidual::debugPlot()
{
	if(state_state==ResState::OOB) return;
	Vec3b cT = Vec3b(0,0,0);

	if(freeDebugParam5==0)
	{
		float rT = 20*sqrt(state_energy/9);
		if(rT<0) rT=0; if(rT>255)rT=255;
		cT = Vec3b(0,255-rT,rT);
	}
	else
	{
		if(state_state == ResState::IN) cT = Vec3b(255,0,0);
		else if(state_state == ResState::OOB) cT = Vec3b(255,255,0);
		else if(state_state == ResState::OUTLIER) cT = Vec3b(0,0,255);
		else cT = Vec3b(255,255,255);
	}

	for(int i=0;i<patternNum;i++)
	{
		if((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0]-3 && projectedTo[i][1] < hG[0]-3 ))
			target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1],cT);
	}
}



void PointFrameResidual::applyRes(bool copyJacobians)
{
	if(copyJacobians)
	{
		if(state_state == ResState::OOB)
		{
			assert(!efResidual->isActiveAndIsGoodNEW);
			return;	// can never go back from OOB
		}
		//好的残差 就把efResidual的is设为true 并调用takeDataF
		if(state_NewState == ResState::IN)// && )
		{
			efResidual->isActiveAndIsGoodNEW=true;
			efResidual->takeDataF();
		}
		else
		{
			efResidual->isActiveAndIsGoodNEW=false;
		}
	}
	//state_state = state_NewState
	setState(state_NewState);
	//能量也更新一下
	state_energy = state_NewEnergy;
}
}

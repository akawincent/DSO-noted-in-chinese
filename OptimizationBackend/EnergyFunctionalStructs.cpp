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


#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

/******************* 计算JpJdF ********************/
void EFResidual::takeDataF()
{
	std::swap<RawResidualJacobian*>(J, data->J);

	Vec2f JI_JI_Jd = J->JIdx2 * J->Jpdd;

	for(int i=0;i<6;i++)
		JpJdF[i] = J->Jpdxi[0][i]*JI_JI_Jd[0] + J->Jpdxi[1][i] * JI_JI_Jd[1];
	//8x1的向量 前6x1为 dr21/dT21 * dr21/dp1  后2x1为 dr21/d光度参数 * dr21/dp1
	JpJdF.segment<2>(6) = J->JabJIdx*J->Jpdd;
}


//将FrameHessian中的state给到EFFrame中的delta量
void EFFrame::takeData()
{
	//这里的data是这个帧的FrameHessian类
	//prior获取先验量，用于构造涉及边缘化的能量的高斯牛顿近似的Hessian矩阵
	prior = data->getPrior().head<8>();
	//获取delta量 = 此刻优化时的增量和 - 边缘化时刻的增量和
	delta = data->get_state_minus_stateZero().head<8>();
	//边缘化时的优化累积起来的增量和
	delta_prior =  (data->get_state() - data->getPriorZero()).head<8>();

//	Vec10 state_zero =  data->get_state_zero();
//	state_zero.segment<3>(0) = SCALE_XI_TRANS * state_zero.segment<3>(0);
//	state_zero.segment<3>(3) = SCALE_XI_ROT * state_zero.segment<3>(3);
//	state_zero[6] = SCALE_A * state_zero[6];
//	state_zero[7] = SCALE_B * state_zero[7];
//	state_zero[8] = SCALE_A * state_zero[8];
//	state_zero[9] = SCALE_B * state_zero[9];
//
//	std::cout << "state_zero: " << state_zero.transpose() << "\n";
	assert(data->frameID != -1);
	//给个id
	frameID = data->frameID;
}




void EFPoint::takeData()
{
	priorF = data->hasDepthPrior ? setting_idepthFixPrior*SCALE_IDEPTH*SCALE_IDEPTH : 0;
	if(setting_solverMode & SOLVER_REMOVE_POSEPRIOR) priorF=0;

	deltaF = data->idepth-data->idepth_zero;
}


void EFResidual::fixLinearizationF(EnergyFunctional* ef)
{
	Vec8f dp = ef->adHTdeltaF[hostIDX+ef->nFrames*targetIDX];

	// compute Jp*delta
	__m128 Jp_delta_x = _mm_set1_ps(J->Jpdxi[0].dot(dp.head<6>())
								   +J->Jpdc[0].dot(ef->cDeltaF)
								   +J->Jpdd[0]*point->deltaF);
	__m128 Jp_delta_y = _mm_set1_ps(J->Jpdxi[1].dot(dp.head<6>())
								   +J->Jpdc[1].dot(ef->cDeltaF)
								   +J->Jpdd[1]*point->deltaF);
	__m128 delta_a = _mm_set1_ps((float)(dp[6]));
	__m128 delta_b = _mm_set1_ps((float)(dp[7]));

	for(int i=0;i<patternNum;i+=4)
	{
		// PATTERN: rtz = resF - [JI*Jp Ja]*delta.
		__m128 rtz = _mm_load_ps(((float*)&J->resF)+i);
		rtz = _mm_sub_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(J->JIdx))+i),Jp_delta_x));
		rtz = _mm_sub_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(J->JIdx+1))+i),Jp_delta_y));
		rtz = _mm_sub_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(J->JabF))+i),delta_a));
		rtz = _mm_sub_ps(rtz,_mm_mul_ps(_mm_load_ps(((float*)(J->JabF+1))+i),delta_b));
		_mm_store_ps(((float*)&res_toZeroF)+i, rtz);
	}

	isLinearized = true;
}

}

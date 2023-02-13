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
#include "FullSystem/ImmaturePoint.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "FullSystem/CoarseTracker.h"

namespace dso
{


/******************** 选定被边缘化掉的帧的策略 *******************/
//note:传进来的这个关键帧newFH好像在函数体内没用到 
//	   可以理解成当关键帧newFH进来后 这个函数为了维护滑动窗口并把newFH加入窗口需要选择之前的一些帧进行边缘化
void FullSystem::flagFramesForMarginalization(FrameHessian* newFH)
{
	//setting_maxFrames = 7 滑动窗口最多维护7个帧
	//setting_minFrames = 5 滑动窗口最少要维护5个帧
	//setting_minFrameAge = 1
	if(setting_minFrameAge > setting_maxFrames)
	{
		//多出的帧全部边缘化
		for(int i=setting_maxFrames;i<(int)frameHessians.size();i++)
		{
			FrameHessian* fh = frameHessians[i-setting_maxFrames];
			fh->flaggedForMarginalization = true;
		}
		return;
	}

	//标记为边缘化帧的个数
	int flagged = 0;
	// marginalize all frames that have not enough points.
	for(int i=0;i<(int)frameHessians.size();i++)
	{
		FrameHessian* fh = frameHessians[i];
		//激活点+未成熟点
		int in = fh->pointHessians.size() + fh->immaturePoints.size();
		//边缘化掉的点+丢掉的点
		int out = fh->pointHessiansMarginalized.size() + fh->pointHessiansOut.size();
		//计算出了(fh-1)帧和fh帧之间的相对光度参数 a21 b21
		Vec2 refToFh=AffLight::fromToVecExposure(frameHessians.back()->ab_exposure, fh->ab_exposure,
				frameHessians.back()->aff_g2l(), fh->aff_g2l());
		
		/*********************边缘化条件*****************/
		/*	
			1.fh帧的in点数量少于0.05*点的总数量
			2.fh与(fh-1)帧之间的a21超过0.7
			3.容器内所有的帧数量 - 已经被边缘化的帧数量 > 滑动窗口维护帧的最小数量
		*/
		//setting_minPointsRemaining = 0.05
		//setting_maxLogAffFacInWindow = 0.7
		//setting_minFrames = 5 滑动窗口最少要维护5个帧
		//in < setting_minPointsRemaining *(in+out) 这个条件就是论文中Keyframe Marginalization的第一点规则
		if( (in < setting_minPointsRemaining *(in+out) || fabs(logf((float)refToFh[0])) > setting_maxLogAffFacInWindow)
				&& ((int)frameHessians.size())-flagged > setting_minFrames)
		{
//			printf("MARGINALIZE frame %d, as only %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
			fh->flaggedForMarginalization = true;
			flagged++;
		}
		else
		{
//			printf("May Keep frame %d, as %'d/%'d points remaining (%'d %'d %'d %'d). VisInLast %'d / %'d. traces %d, activated %d!\n",
//					fh->frameID, in, in+out,
//					(int)fh->pointHessians.size(), (int)fh->immaturePoints.size(),
//					(int)fh->pointHessiansMarginalized.size(), (int)fh->pointHessiansOut.size(),
//					visInLast, outInLast,
//					fh->statistics_tracesCreatedForThisFrame, fh->statistics_pointsActivatedForThisFrame);
		}
	}

	// marginalize one.
	//无论如何都至少要边缘化一个帧 为关键帧newFH让出位置
	//这个条件代表着窗口已经满了并且还有关键帧等着进窗口里
	//这时候要用论文中Keyframe Marginalization的第三点规则来选择需要被边缘化的帧
	if((int)frameHessians.size()-flagged >= setting_maxFrames)
	{
		double smallestScore = 1;
		FrameHessian* toMarginalize = 0;
		FrameHessian* latest = frameHessians.back();

		for(FrameHessian* fh : frameHessians)
		{
			//保留了latest帧不被边缘化
			//这里就是论文中Keyframe Marginalization的第二点规则  最新帧newFH和第二新帧latest不被边缘化
			if(fh->frameID > latest->frameID-setting_minFrameAge || fh->frameID == 0) continue;
			//if(fh==frameHessians.front() == 0) continue;
			
			//距离评分
			//这样的启发式函数可以让关键帧有更好的3D空间分布 关键帧之间离着更紧
			double distScore = 0;
			for(FrameFramePrecalc &ffh : fh->targetPrecalc)
			{	
				//这里也体现了论文中Keyframe Marginalization的第二点规则
				if(ffh.target->frameID > latest->frameID-setting_minFrameAge+1 || ffh.target == ffh.host) continue;
				//累加所有fh以后的帧到fh之间距离的倒数
				distScore += 1/(1e-5+ffh.distanceLL);
			}
			//再乘以fh帧到最新一帧的距离的平方根  离着最新帧越远 这个数越小
			distScore *= -sqrtf(fh->targetPrecalc.back().distanceLL);

			//distScore最小的被边缘化掉
			if(distScore < smallestScore)
			{
				smallestScore = distScore;
				toMarginalize = fh;
			}
		}

//		printf("MARGINALIZE frame %d, as it is the closest (score %.2f)!\n",
//				toMarginalize->frameID, smallestScore);
		toMarginalize->flaggedForMarginalization = true;
		flagged++;
	}

//	printf("FRAMES LEFT: ");
//	for(FrameHessian* fh : frameHessians)
//		printf("%d ", fh->frameID);
//	printf("\n");
}




void FullSystem::marginalizeFrame(FrameHessian* frame)
{
	// marginalize or remove all this frames points.

	assert((int)frame->pointHessians.size()==0);


	ef->marginalizeFrame(frame->efFrame);

	// drop all observations of existing points in that frame.

	for(FrameHessian* fh : frameHessians)
	{
		if(fh==frame) continue;

		for(PointHessian* ph : fh->pointHessians)
		{
			for(unsigned int i=0;i<ph->residuals.size();i++)
			{
				PointFrameResidual* r = ph->residuals[i];
				if(r->target == frame)
				{
					if(ph->lastResiduals[0].first == r)
						ph->lastResiduals[0].first=0;
					else if(ph->lastResiduals[1].first == r)
						ph->lastResiduals[1].first=0;


					if(r->host->frameID < r->target->frameID)
						statistics_numForceDroppedResFwd++;
					else
						statistics_numForceDroppedResBwd++;

					ef->dropResidual(r->efResidual);
					deleteOut<PointFrameResidual>(ph->residuals,i);
					break;
				}
			}
		}
	}



    {
        std::vector<FrameHessian*> v;
        v.push_back(frame);
        for(IOWrap::Output3DWrapper* ow : outputWrapper)
            ow->publishKeyframes(v, true, &Hcalib);
    }


	frame->shell->marginalizedAt = frameHessians.back()->shell->id;
	frame->shell->movedByOpt = frame->w2c_leftEps().norm();

	deleteOutOrder<FrameHessian>(frameHessians, frame);
	for(unsigned int i=0;i<frameHessians.size();i++)
		frameHessians[i]->idx = i;




	setPrecalcValues();
	ef->setAdjointsF(&Hcalib);
}




}

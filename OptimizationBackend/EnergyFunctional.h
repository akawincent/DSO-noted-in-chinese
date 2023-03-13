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
#include "util/IndexThreadReduce.h"
#include "vector"
#include <math.h>
#include "map"


namespace dso
{

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessian;


class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;
class AccumulatedTopHessian;
class AccumulatedTopHessianSSE;
class AccumulatedSCHessian;
class AccumulatedSCHessianSSE;


extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;


/*****************存储着用于后端优化的量:帧 点 残差关系*****************/
class EnergyFunctional {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	friend class EFFrame;
	friend class EFPoint;
	friend class EFResidual;
	friend class AccumulatedTopHessian;
	friend class AccumulatedTopHessianSSE;
	friend class AccumulatedSCHessian;
	friend class AccumulatedSCHessianSSE;

	EnergyFunctional();
	~EnergyFunctional();

	//向后端优化中加入残差关系
	EFResidual* insertResidual(PointFrameResidual* r);
	//向后端优化中加入关键帧
	EFFrame* insertFrame(FrameHessian* fh, CalibHessian* Hcalib);
	//向后端优化中加入激活点
	EFPoint* insertPoint(PointHessian* ph);

	//扔掉不好的残差关系
	void dropResidual(EFResidual* r);
	//边缘化帧
	void marginalizeFrame(EFFrame* fh);
	//移除点
	void removePoint(EFPoint* ph);



	void marginalizePointsF();
	void dropPointsF();
	void solveSystemF(int iteration, double lambda, CalibHessian* HCalib);
	double calcMEnergyF();
	double calcLEnergyF_MT();


	void makeIDX();

	void setDeltaF(CalibHessian* HCalib);
	//设置帧间的伴随矩阵，因为固定住线性化点了，因此要把增量从线性化点映射到相对位姿空间上
	void setAdjointsF(CalibHessian* Hcalib);

	//保存着插入到后端优化的关键帧  这里存的是Frame用于后端优化的信息 也就是EEFrame类
	std::vector<EFFrame*> frames;
	//nPoints:窗口内的成熟点数量
	//nFrames:窗口内的帧数量
	//nResiduals:窗口内残差关系的数量
	int nPoints, nFrames, nResiduals;

	//边缘化后的留下的先验信息
	MatXX HM;
	VecX bM;

	int resInA, resInL, resInM;
	MatXX lastHS;
	VecX lastbS;
	VecX lastX;
	std::vector<VecX> lastNullspaces_forLogging;
	std::vector<VecX> lastNullspaces_pose;
	std::vector<VecX> lastNullspaces_scale;
	std::vector<VecX> lastNullspaces_affA;
	std::vector<VecX> lastNullspaces_affB;

	IndexThreadReduce<Vec10>* red;


	std::map<uint64_t,
	  Eigen::Vector2i,
	  std::less<uint64_t>,
	  Eigen::aligned_allocator<std::pair<const uint64_t, Eigen::Vector2i>>
	  > connectivityMap;

private:

	VecX getStitchedDeltaF() const;

	void resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT);
    void resubstituteFPt(const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid);

	void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
	void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
	void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

	void calcLEnergyPt(int min, int max, Vec10* stats, int tid);

	void orthogonalize(VecX* b, MatXX* H);
	//将host和target帧上的变化量转化到相对位移上
	Mat18f* adHTdeltaF;

	//向量空间的矩阵伴随
	Mat88* adHost;
	Mat88* adTarget;
	//将host和target的变化量转化到相对位姿
	Mat88f* adHostF;
	Mat88f* adTargetF;


	VecC cPrior;
	VecCf cDeltaF;
	VecCf cPriorF;

	AccumulatedTopHessianSSE* accSSE_top_L;
	AccumulatedTopHessianSSE* accSSE_top_A;


	AccumulatedSCHessianSSE* accSSE_bot;

	std::vector<EFPoint*> allPoints;
	std::vector<EFPoint*> allPointsToMarg;

	float currentLambda;
};
}


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
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include "util/settings.h"
#include "vector"
#include <math.h>

namespace dso
{
struct CalibHessian;
struct FrameHessian;

//存储图像中关于三维点的信息的数据结构体
struct Pnt
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// index in jacobian. never changes (actually, there is no reason why).??
	//三维点投影到firstFrame的图像坐标
	float u,v;

	// idepth / isgood / energy during optimization.
	//三维点在优化的当前迭代中的逆深度
	float idepth;
	//三维点在优化的当前迭代中是否为好点
	bool isGood;
	//三维点在优化的当前迭代中构成的残差项
	Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
	//三维点在优化的下一次迭代中计算更新出来的逆深度
	float idepth_new;
	//三维点在优化的下一次迭代中计算更新出来的是否为好点
	bool isGood_new;
	//三维点在优化的下一次迭代中计算更新出来的能量项
	Vec2f energy_new;

	//逆深度的真值(送入后端BA优化)
	float iR;
	float iRSumNum;

	float lastHessian;
	float lastHessian_new;

	// max stepsize for idepth (corresponding to max. movement in pixel-space).
	float maxstep;

	// idx (x+y*w) of closest point one pyramid level above.
	//当前点在上一层金字塔图像的父点
	int parent;
	//上一层金字塔中点的逆深度
	float parentDist;

	// idx (x+y*w) of up to 10 nearest points in pixel space.
	//当前点的10个最近邻像素点在points数组中的索引
	int neighbours[10];
	//邻域的逆深度
	float neighboursDist[10];
	//在图像中是什么尺度下被提取出来的
	float my_type;
	//外点阈值
	float outlierTH;
};

/**********************DSO进行最初第一帧和第二帧之间初始化操作的类*******************************/
class CoarseInitializer {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	//构造函数 初始化w和h两个参数
	CoarseInitializer(int w, int h);
	//析构函数
	~CoarseInitializer();


	void setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian);
	bool trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps);
	void calcTGrads(FrameHessian* newFrameHessian);

	//帧的编号
	int frameID;
	//是否固定光度变换参数
	bool fixAffine;
	//是否打印调试信息
	bool printDebug;
	//存储在图像金字塔不同等级的所有三维点及其领域(pattern)
	Pnt* points[PYR_LEVELS];
	//不同等级图像的三维点的个数
	int numPoints[PYR_LEVELS];
	//当前帧到新一帧的光度变换参数
	AffLight thisToNext_aff;
	//当前帧到新一帧的姿态变换李群
	SE3 thisToNext;

	//初始化时第一帧的数据信息
	FrameHessian* firstFrame;
	//初始化时第二帧的数据信息
	FrameHessian* newFrame;
private:
	//相机内参矩阵
	Mat33 K[PYR_LEVELS];
	//K_inverse:相机内参矩阵的逆
	Mat33 Ki[PYR_LEVELS];
	//世界坐标系转变到相机坐标系的x，y坐标焦距
	double fx[PYR_LEVELS];
	double fy[PYR_LEVELS];
	//K_inverse中的f_inverse
	double fxi[PYR_LEVELS];
	double fyi[PYR_LEVELS];
	//世界坐标系转变到相机坐标系的x，y坐标常数校正
	double cx[PYR_LEVELS];
	double cy[PYR_LEVELS];
	//K_inverse中的c_inverse
	double cxi[PYR_LEVELS];
	double cyi[PYR_LEVELS];
	//w和h是图像的宽和高
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];

	//构造相机内参矩阵
	void makeK(CalibHessian* HCalib);

	bool snapped;
	int snappedAt;

	// pyramid images & levels on all levels
	//当前帧与新帧的图像金字塔每一级图像像素信息
	Eigen::Vector3f* dINew[PYR_LEVELS];
	Eigen::Vector3f* dIFist[PYR_LEVELS];
	//8*8的float型对角矩阵
	Eigen::DiagonalMatrix<float, 8> wM;

	// temporary buffers for H and b.
	//暂时存储当前帧和新帧b向量的矩阵
	Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
	Vec10f* JbBuffer_new;

	Accumulator9 acc9;
	Accumulator9 acc9SC;

	//图像导数
	Vec3f dGrads[PYR_LEVELS];

	float alphaK;
	float alphaW;
	float regWeight;
	float couplingWeight;

	Vec3f calcResAndGS(
			int lvl,
			Mat88f &H_out, Vec8f &b_out,
			Mat88f &H_out_sc, Vec8f &b_out_sc,
			const SE3 &refToNew, AffLight refToNew_aff,
			bool plot);
	Vec3f calcEC(int lvl); // returns OLD NERGY, NEW ENERGY, NUM TERMS.
	void optReg(int lvl);

	void propagateUp(int srcLvl);
	void propagateDown(int srcLvl);
	float rescale();

	void resetPoints(int lvl);
	void doStep(int lvl, float lambda, Vec8f inc);
	void applyStep(int lvl);

	void makeGradients(Eigen::Vector3f** data);

    void debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps);
	void makeNN();
};




struct FLANNPointcloud
{
    inline FLANNPointcloud() {num=0; points=0;}
    inline FLANNPointcloud(int n, Pnt* p) :  num(n), points(p) {}
	int num;
	Pnt* points;
	inline size_t kdtree_get_point_count() const { return num; }
	inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
	{
		const float d0=p1[0]-points[idx_p2].u;
		const float d1=p1[1]-points[idx_p2].v;
		return d0*d0+d1*d1;
	}

	inline float kdtree_get_pt(const size_t idx, int dim) const
	{
		if (dim==0) return points[idx].u;
		else return points[idx].v;
	}
	template <class BBOX>
		bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
};

}



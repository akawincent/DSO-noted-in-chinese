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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"


#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

/****************DSO初始化流程类CoarseInitializer的构造函数*********************************/
//  parameter:  ww ————> 图像的宽; hh ————> 图像的高 
// default parameter: thisToNext_aff(0,0) ————> 光度变换参数 a21 = 0 b21 = 0;
//					  thisToNext —————> SE(3)姿态变换	
CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3())
{
	//初始化图像每个金字塔等级的点以及点的个数
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];

	//初始化时的frameID为-1
	frameID=-1;
	//固定光度参数
	fixAffine=true;
	printDebug=false;

	//Eigen库中.diagonal()将矩阵对角线元素作为向量返回
	//姿态变换矩阵旋转部分的尺度
	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	//姿态变换矩阵平移部分的尺度	
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}

CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}

/****** 追踪第一帧和第二帧之间的相机位姿并重新估计第一帧中三维点points的深度 ******/
/* parameter:
		newFrameHessian ——————> 传入的帧数据结构 赋给了CoarseInitializer::newFrame 
								是前端跟踪初始化过程的第二个帧
*/
bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	newFrame = newFrameHessian;

    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = {5,5,10,30,50};
	//能量项的权重
	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

	//snapped此时为false
	//初始化两帧之间的位姿thisToNext的平移部分translation()为0 初始化每个三维点真实逆深度为1 初始化光度参数
	if(!snapped)
	{
		//平移部分设置为0了 但旋转部分是定义后未初始化的
		//这里是一个trick?
		thisToNext.translation().setZero();
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			int npts = numPoints[lvl];
			//初始化一些没在setFirst中初始化的量
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}

	//refToNew_current 姿态变换矩阵 （平移部分初始化为0）
	SE3 refToNew_current = thisToNext;
	//refToNew_aff_current 光度变换参数 （初始化a1和b1为0）
	AffLight refToNew_aff_current = thisToNext_aff;

	//初始化时第一帧和第二帧的曝光时间都大于0 则给光度变换参数估计一个初值
	//这时候曝光时间已知 所以refToNew_aff_current在这里存储的光度参数是(a21,b21)而不是(a1,b1)?
	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		//a21 = log(t2/t1)   b21 = 0
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.

	//从金字塔顶层图像开始遍历
	Vec3f latestRes = Vec3f::Zero();
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{
		if(lvl<pyrLevelsUsed-1)
			propagateDown(lvl+1);

		//定义增量方程的H和b矩阵以及Schur Complement
		Mat88f H,Hsc; Vec8f b,bsc;
		//再一次为points[lvl]初始化  同时将最高层的坏点转化为好点 在这里也将idepth = idepth_new
		resetPoints(lvl);
		//计算得到当前层迭代优化前的总energy项以及H b Hsc bsc
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
		//生效calcResAndGS中计算出来的新的优化信息 energy isGood lastHessian idepth等
		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		//记录迭代优化的失败次数
		int fails=0;

		//printDebug初始化为false
		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

		/************ 迭代优化求解姿态变换矩阵、光度变换参数以及逆深度 ***********/
		int iteration=0;
		while(true)
		{
			Mat88f Hl = H;
			//加入阻尼因子lamda = 0.1 就是LM法 (u*I + H)x = b
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			//Schur Complement
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			//why?
			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));

			//求解增量
			Vec8f inc;
			//初始化为true
			if(fixAffine)
			{
				inc.head<6>() = - (wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
				//光度参数的增量固定为0
				inc.tail<2>().setZero();
			}
			else
				inc = - (wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.

			/****************** 这里将更新后的状态变量暂时存储在new里面 *******************/
			//更新姿态变换矩阵
			SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
			//更新光度变换参数
			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			//更新每个point的逆深度 pts[i].idepth_new
			doStep(lvl, lambda, inc);

			//状态变量更新后 再求一次更新后的总energy以及增量方程的H和b及其舒尔补
			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);

			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);

			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						inc.norm());
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}

			//如果本次优化更新后的energy小于上一次计算出的energy 则接受本次优化 更新状态变量 
			if(accept)
			{

				if(resNew[1] == alphaK*numPoints[lvl])
					snapped = true;
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				optReg(lvl);
				lambda *= 0.5;		//降低阻尼因子 放慢优化的步伐
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;			//累积失败次数
				lambda *= 4;		//增大阻尼因子 放大优化的步伐
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;

			//迭代停止条件  增量值非常小 or 迭代次数大于设置的最大迭代次数 or 连续迭代失败次数两次以上
			if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;

				quitOpt = true;
			}


			if(quitOpt) break;
			iteration++;
		}
		//优化结束后最终的能量
		latestRes = resOld;
	}

	//保存优化结果 (金字塔每一层都迭代优化后的结果)
	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i);

	//帧编号累加
	frameID++;
	if(!snapped) snappedAt=0;

	if(snapped && snappedAt==0)
		snappedAt = frameID;

    debugPlot(0,wraps);

	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;



	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
//函数说明：CoareInitializer对象中calcResAndGS函数的具体实现
/*paramater list: 
				lvl ————>图像金字塔的等级
				H_out ————> 当前帧残差关于各参数的雅可比矩阵相乘得到的Hessian矩阵
				b_out ————> 当前帧残差关于个参数的雅可比矩阵与当前误差项乘积
				H_out_sc ————> 经过Schur补处理后的Hessian矩阵
				b_out_sc ————> 经过Schur补处理后的b矩阵
				refToNew ————> 到新一帧的姿态变换李群SE(3)
				refToNew_aff ————> 到新一帧的光度变换(a21,b21)
*/
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
{
	//wl为该金字塔等级图像的宽  hl为为该金字塔等级图像的高
	int wl = w[lvl], hl = h[lvl];
	//第一帧的像素信息(初始化时的参考帧) 
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	//后面传进来的帧的像素信息
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

	//帧间姿态变换的旋转部分乘以该图像金字塔等级的相机内参的逆！！！！
	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();	//cast类型转换为浮点型
	//帧间姿态变换的平移部分
	Vec3f t = refToNew.translation().cast<float>();		//cast类型转换为浮点型
	//r2new_aff 保存着光度变换参数 exp(a1)和b1
	//若firstFrame和newFrame的曝光时间已知 则r2new_aff = (exp(a21),b21)
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	//当前图像等级的相机内参矩阵参数
	//在CoarseInitializer::setFirst中的makeK函数中对fx fy cx cy赋值了
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	//用于计算11*11的hessian矩阵 JWJ^T 内部采用EIGEN_ALIGNED_OPERATOR加速运算
	Accumulator11 E;
	//acc9是9*9的hessian矩阵
	acc9.initialize();
	E.initialize();

	//第一帧图像金字塔等级中的点的个数 是firstFrame在setFirst函数中被赋值的
	int npts = numPoints[lvl];
	//第一帧该等级图像中所有三维点的信息 是firstFrame在setFirst函数中被赋值的
	Pnt* ptsl = points[lvl];
	for(int i=0;i<npts;i++)
	{
		//依次遍历图像中所有的点
		Pnt* point = ptsl+i;
		//逆深度的最大调整步长?
		point->maxstep = 1e10;
		//该点不是一个好点
		if(!point->isGood)
		{
			//把能量项第一个加入到E中
			E.updateSingle((float)(point->energy[0]));
			//energy在迭代更新中不变
			point->energy_new = point->energy;
			//在迭代更新中 该point仍然为false
			point->isGood_new = false;
			//跳过该点 计算H和b不考虑该点
			continue;
		}

		//VecNRf为8*1的向量 存储8-pattern邻域中每个点对位姿、光度、深度的Jacobian信息
        VecNRf dp0;
        VecNRf dp1;
        VecNRf dp2;
        VecNRf dp3;
        VecNRf dp4;
        VecNRf dp5;
        VecNRf dp6;
        VecNRf dp7;
        VecNRf dd;

		//r这个8*1的向量是存储8-pattern邻域中每个点的误差项
        VecNRf r;
		
		//JbBuffer is temporary buffers for H and b
		//This step do initialization for JbBuffer_new
		//JbBuffer_new is updated in the next iteration of optimization
		JbBuffer_new[i].setZero();

		// sum over all residuals.
		bool isGood = true;
		//定义一个energy 后面计算这个energy 一个point对应一个energy
		float energy=0;
		//该点领域的8个点 论文中的8-pattern
		for(int idx=0;idx<patternNum;idx++)
		{
			/**************************** projection ******************************/

			//当前点的8-pattern邻域与当前点的坐标差值
			int dx = patternP[idx][0];	//patternP[idx][0] = staticpattern[8][idx][0]
			int dy = patternP[idx][1];  //patternP[idx][1] = staticpattern[8][idx][1]

			//ponit->u+dx,point->v+dy：8-Pattern中点的像素坐标
			//pt = [X2/Z1,Y2/Z1,Z2/Z1]  
			//这里的idepth_new是迭代后更新的Z1  如果是优化中的初始化则idepth_new = idepth
			Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new; //源代码

			//x2_regular = idepth_2 * pt;idepth_2 = 1/Z2
			float u = pt[0] / pt[2];	//点投影到newFrame上的归一化平面x坐标
			float v = pt[1] / pt[2];    //点投影到newFrame上的归一化平面y坐标
			float Ku = fxl * u + cxl;   //点投影到newFrame上像素坐标x轴
			float Kv = fyl * v + cyl;   //点投影到newFrame上像素坐标y轴
			// new_idepth = Z2 点在newFrame中的深度
			float new_idepth = point->idepth_new/pt[2];   

			//如果点投影到新帧中的坐标处于新帧图像边缘或是逆深度不合理 则此点为坏点 不参与优化
			if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0))
			{
				isGood = false;
				break;
			}

			/****************************data association******************************/

			//获取点投影到newframe的(Ku,Kv)像素点处的灰度值以及图像梯度
			//hitColor[0]:灰度值  hitColor[1]:x轴图像梯度 hitColor[2]:y轴图像梯度
			Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
			//Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

			//获取点投影到firstFrame的(point->u+dx,point->v+dy)像素点处的灰度值
			//float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
			float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

			//不在浮点数范围之后 设该点为坏点
			if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0]))
			{
				isGood = false;
				break;
			}

			/****************************Photometric error******************************/
			//I2[X]- (a21*I1[X] + b21)
			float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
			//hw:Huber鲁棒核函数 setting_huberTH：Huber阈值
			//不超过threshold是r*r L2损失  超过了则2*r - th^2 L1损失
			float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
			//邻域的8个点都累积到energy
			energy += hw *residual*residual*(2-hw);

			/*用于计算residual对inverse_depth的雅可比矩阵 参考https://www.cnblogs.com/JingeTU/p/8203606.html
				dxdd = 1/inverse_depth_1 * inverse_depth_2 * (t21_x - u2'*t21_z)
				dydd = 1/inverse_depth_1 * inverse_depth_2 * (t21_y - v2'*t21_z)
			*/
			//源代码
			float dxdd = (t[0]-t[2]*u)/pt[2];
			float dydd = (t[1]-t[2]*v)/pt[2];

			//超过了Huber阈值 hw是energy的加权因子 开根号就得到了residual的加权因子 
			//误差求导模型中是有这个加权因子的
			if(hw < 1) hw = sqrtf(hw);
			//hitColor[1]:x轴图像梯度gx  hitColor[2]:y轴图像梯度gy
			float dxInterp = hw*hitColor[1]*fxl;		//wh*gx*fx
			float dyInterp = hw*hitColor[2]*fyl;		//wh*gy*fy

			/*残差residual关于SE(3)姿态变换的雅可比矩阵 参考https://www.cnblogs.com/JingeTU/p/8203606.html
				dp0[] = wh*gx*fx*inverse_depth_2
				dp1[] = wh*gy*fy*inverse_depth_2
				dp2[] = wh*(-gx*fx*inverse_depth_2*u2'-gy*fy*inverse_depth_2*v2')
				dp3[] = wh*(-gx*fx*u2'*v2'-gy*fy*(1+v2')^2)
				dp4[] = wh*(-gx*fx*(1+v2')^2-gy*fy*u2'*v2')
				dp5[] = wh*(-gx*fx*v2'+gy*fy*u2')
			*/
			dp0[idx] = new_idepth * dxInterp;				
			dp1[idx] = new_idepth * dyInterp;
			dp2[idx] = - new_idepth * (u * dxInterp + v * dyInterp);
			dp3[idx] = - u * v * dxInterp - (1 + v * v) * dyInterp;
			dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
			dp5[idx] = - v * dxInterp + u * dyInterp;

			/*残差residual关于光度变换参数的雅可比矩阵
			/*初始化时设置了a1和b1都是0 所以只需要针对a2和b2求导进行优化就可以了 
			/*这里不关心a21和b21 优化要解出来的是a2和b2
				dp6[] = -wh*a21*rlR   res对a2求导
				dp7[] = -wh           res对b2求导
			*/
			dp6[idx] = - hw*r2new_aff[0] * rlR;
			dp7[idx] = - hw*1;

			/* 残差residual关于逆深度的雅可比矩阵 */
			dd[idx] = dxInterp * dxdd  + dyInterp * dydd;
			/* 存储残差residual 用于计算增量方程的b */
			r[idx] = hw*residual;

			//maxstep可以理解为移动单位像素，深度的变化
			float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm();
			if(maxstep < point->maxstep) point->maxstep = maxstep;

			// immediately compute dp*dd' and dd*dd' in JbBuffer1.
			/*
				JbBuffer_new[i][0]~[7] 将pattern中所有的Jx21 * Jp累加
				JbBuffer_new[i][8] 将pattern中所有的Jp * r21 相加
				JbBuffer_new[i][9] 将pattern中所有的 Jp *Jp 相加
				！！！！这里的Jx Jp r都是针对一个point的！！！！
			*/
			JbBuffer_new[i][0] += dp0[idx]*dd[idx];
			JbBuffer_new[i][1] += dp1[idx]*dd[idx];
			JbBuffer_new[i][2] += dp2[idx]*dd[idx];
			JbBuffer_new[i][3] += dp3[idx]*dd[idx];
			JbBuffer_new[i][4] += dp4[idx]*dd[idx];
			JbBuffer_new[i][5] += dp5[idx]*dd[idx];
			JbBuffer_new[i][6] += dp6[idx]*dd[idx];
			JbBuffer_new[i][7] += dp7[idx]*dd[idx];
			JbBuffer_new[i][8] += r[idx]*dd[idx];
			JbBuffer_new[i][9] += dd[idx]*dd[idx];
		}

		//该点不是好点或者能量项大于阈值
		if(!isGood || energy > point->outlierTH*20)
		{
			//这里的point->energy[0]应该是0 加入到E
			E.updateSingle((float)(point->energy[0]));
			point->isGood_new = false;
			point->energy_new = point->energy;
			continue;
		}

		// add into energy.
		//将该点包括其领域的累加energy加入到E
		E.updateSingle(energy);
		//更新是否为好点
		point->isGood_new = true;
		//将energy添加到该点point->energy_new[0]
		point->energy_new[0] = energy;

		// update Hessian matrix.
		/************************计算H矩阵和b向量，存储在acc9**************************/
		//acc9.updateSSE使用了Intel的SSE指令集，一次操作取4个 float，完成4个不同数据、相同命令的float运算
		for(int i=0;i+3<patternNum;i+=4)
			//_mm_load_ps是取内存连续的4块地址中的数据
			//第一次_mm_load_ps能导入一个point的dp + 0~3 4个邻域点
			//第二次_mm_load_ps能导入一个point的dp + 4~7 剩下4个邻域点
			//updateSSE计算每个point的Hessian矩阵 并且遍历它的8-pattern部分的Hessian进行累加
			acc9.updateSSE(
					//转换到float类型操作
					_mm_load_ps(((float*)(&dp0))+i),	
					_mm_load_ps(((float*)(&dp1))+i),
					_mm_load_ps(((float*)(&dp2))+i),
					_mm_load_ps(((float*)(&dp3))+i),
					_mm_load_ps(((float*)(&dp4))+i),
					_mm_load_ps(((float*)(&dp5))+i),
					_mm_load_ps(((float*)(&dp6))+i),
					_mm_load_ps(((float*)(&dp7))+i),
					_mm_load_ps(((float*)(&r))+i));

		//若patternNum%4!=0,就把不满4的余数部分分别进行一次float运算
		for(int i=((patternNum>>2)<<2); i < patternNum; i++)
			acc9.updateSingle(
					(float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
					(float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
					(float)r[i]);
	}
	/*********************** 循环完了当前等级图像的point ******************************/

	
	E.finish();	
	//完成J*J^T的计算 针对一个point的J为1*8的Jacobian 和一个1*1的residual
	acc9.finish();      
	/************************acc9矩阵的构成************************/
	//x21是newFrame与firstFrame之间的变换参数 r21是两个point之间的光度误差
	//note:acc9里面并没有加入误差对逆深度的导数！！！！
	//acc9的优化更新量都是对帧而言的，而不带有描述landmark的参数(逆深度)
	//这也是为什么acc9可以把每个点的对应导数直接相加而不需要考虑分别更新每个点的参数(深度)
	//对应导数的累加代表了所有的point指导了x21的优化，而每个point逆深度的更新则交给了doStep函数
	//Jx21*Jx21 (8*8)    Jx21^T*r21 (8*1)
	//Jx21*r21  (1*8)     r21*r21   (1*1)

	// calculate alpha energy, and decide if we cap it.
	//11*11的矩阵？
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
		{
			
			E.updateSingle((float)(point->energy[1]));
		}
		else
		{
			//好点
			//E and EAlpha are mixed ???
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			E.updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();
	//only one thing can be confirmed that idepth bigger,alphaEnergy bigger and translation bigger,alphaEnergy bigger
	//（所有point的(idepth-1)^2累加 + 平移量*point个数） * 权重(alphaW = 150*150)
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


	// compute alpha opt.
	float alphaOpt;
	//alphaK = 2.5*2.5 这个变量有点像是alphaEnergy平均到每个point的一个阈值
	if(alphaEnergy > alphaK*npts)
	{
		//超出阈值
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW;
	}

	/************************计算HSC矩阵和bSC向量，存储在acc9SC**************************/
	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;
		//Jp * Jp
		point->lastHessian_new = JbBuffer_new[i][9];

		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		//平移过大
		if(alphaOpt==0)
		{
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}
		//分母+1是防止分母过小导致系统不稳定
		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		//
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


	//printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
	//形参引用将计算出的H、b、Hsc、bsc传给实参
	//Jx21^T * Jx21(8*8)
	H_out = acc9.H.topLeftCorner<8,8>();// / acc9.num;
	//Jx21^T * r21(8*1)
	b_out = acc9.H.topRightCorner<8,1>();// / acc9.num;
	//1/Jp*Jp^T *Jx21^T*Jp*Jp^T*Jx21 at(8*8)
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	//1/Jp*Jp^T *Jx21^T*Jp*Jp^T*r21 at(8*!)
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;

	//这些操作是为了干什么 控制位移更新量？
	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;

	//E.A = total energy 
	//alphaEnergy = measure translation and idepth
	//E.num = count of point
	return Vec3f(E.A, alphaEnergy ,E.num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}

void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	if(!snapped)
	{
		for(int i=0;i<npts;i++)
			ptsl[i].iR = 1;
		return;
	}


	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			idnn[nnn] = other->iR;
			nnn++;
		}

		if(nnn > 2)
		{
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}



void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1);
}

void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target

	int nptst= numPoints[srcLvl-1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl-1];

	for(int i=0;i<nptst;i++)
	{
		Pnt* point = ptst+i;
		Pnt* parent = ptss+point->parent;

		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		if(!point->isGood)
		{
			point->iR = point->idepth = point->idepth_new = parent->iR;
			point->isGood=true;
			point->lastHessian=0;
		}
		else
		{
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	optReg(srcLvl-1);
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}

/***********************对Fullsystem的第一个传入的图像选取immature point***************************/
//中间过程包括了在图像上依据梯度平方和提取像素点的过程
void CoarseInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian)
{
	//构造此时帧的相机内参矩阵（所有金子塔等级）
	makeK(HCalib);
	//当前帧赋给CoarseInitializer类中的firstFrame
	firstFrame = newFrameHessian;

	//像素点选择器  for the use of selecting immature points??
	//实例化对象时的parameter:  w[0] 第0层图像宽度   h[0] 第0层图像高度
	PixelSelector sel(w[0],h[0]);

	//0层图像选择像素点用的记录数组
	float* statusMap = new float[w[0]*h[0]];
	//1~5层图像选择像素点用的记录数组
	bool* statusMapB = new bool[w[0]*h[0]];


	//选取图像像素点的稠密程度  0.03体现的是Sparse
	//densities * w * h为该帧该层选取的像素数量
	float densities[] = {0.03,0.05,0.15,0.5,1};
	//遍历每层图像选择像素点
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{	
		//一开始设置pot大小为3
		sel.currentPotential = 3;
		//npts是选点数量
		int npts;
		//对第0层金字塔图像(原图像)进行像素点选择
		if(lvl == 0)
			//npts是选取的像素点数量 statusMap中保存了选取点的标志位信息
			//recursionsLeft = 1，递归两次
			//thFactor = 2 阈值乘数因子
			npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
		else
			//对1~5层金字塔图像进行像素点选择
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);

		if(points[lvl] != 0) delete[] points[lvl];
		//new出来npts数量的Pnt对象给points[lvl]
		//这里的points就是空间中的三维点了
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		//pl是points[lvl]的首地址 可以通过p1的数组索引找到当前金字塔图像等级的第n个point
		Pnt* pl = points[lvl];
		//pl数组的索引
		int nl = 0;

		/**************遍历当前图像等级所有被选出来的点 对points进行初始化**************/
		//被选出来的像素点都各自对应着三维空间中的一个point
		//这里就开始对三维空间点point进行初始化操作了
		//忽略了图像边缘的点
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
		for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
		{
			//if(x==2) printf("y=%d!\n",y);
			if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
			{
				//assert(patternNum==9);
				pl[nl].u = x+0.1;			//初始化point投影到像素坐标平面的u坐标值
				pl[nl].v = y+0.1;			//初始化point投影到像素坐标平面的v坐标值
				pl[nl].idepth = 1;			//初始化point在前端追踪中的逆深度为1
				pl[nl].iR = 1;				//初始化point的真实逆深度为1
				pl[nl].isGood=true;			//初始化point为好点
				pl[nl].energy.setZero();	//初始化point构成的残差能量项为0
				pl[nl].lastHessian=0;
				pl[nl].lastHessian_new=0;
				//如果是1~5层则标志位位1  如果是第0层 则标志位可能为1 2 4 代表了选择该点时的尺度
				pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

				//计算当前像素点的8邻域的梯度平方和 累积放入sumGrad2中
				Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
				float sumGrad2=0;
				for(int idx=0;idx<patternNum;idx++)
				{
					//patternP = staticstaticPattern[8] 论文中说的8-pattern
					int dx = patternP[idx][0];
					int dy = patternP[idx][1];
					float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
					sumGrad2 += absgrad;
				}

//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
//				pl[nl].outlierTH = patternNum*gth*gth;
//
				//外点阈值 超过该阈值就判断为外点？
				pl[nl].outlierTH = patternNum*setting_outlierTH;

				nl++;
				assert(nl <= npts);
			}
		}
		numPoints[lvl]=nl;
	}
	delete[] statusMap;
	delete[] statusMapB;

	makeNN();

	thisToNext=SE3();
	snapped = false;
	//帧的编号赋值为0了
	frameID = snappedAt = 0;

	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();
}

void CoarseInitializer::resetPoints(int lvl)
{
	//pts保存了lvl层的点(三维点信息)  points已经在seFirst和trackFrame中进行了初始化
	Pnt* pts = points[lvl];
	//npts保存了lvl层点的数量
	int npts = numPoints[lvl];
	//遍历所有的点
	for(int i=0;i<npts;i++)
	{	
		//能量项重新置为0
		pts[i].energy.setZero();
		//idepth和idepth_new初始化为一样的值
		pts[i].idepth_new = pts[i].idepth;

		//最高层的坏点
		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			//将这个坏点的真实逆深度iR赋值为其最近邻10个点的逆深度iR的平均值 
			//idepth = idepth_new = iR
			//并将这个坏点变为好点
			float snd=0, sn=0;
			for(int n = 0;n<10;n++)
			{
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if(sn > 0)
			{
				pts[i].isGood=true;
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}

/**************** 迭代优化中将求解出的增量更新到每个point的逆深度 *******************/
/* parameter:
		lvl 	——————> 当前金字塔等级
		lamda 	——————> 优化中的阻尼因子
		inc 	——————> 求解出的增量
*/
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;


		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}

void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}

void CoarseInitializer::makeNN()
{
	const float NNDistFactor=0.05;

	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		indexes[i]->buildIndex();
	}

	const int nn=10;

	// find NN & parents
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		int ret_index[nn];
		float ret_dist[nn];
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			resultSet.init(ret_index, ret_dist);
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			int myidx=0;
			float sumDF = 0;
			for(int k=0;k<nn;k++)
			{
				pts[i].neighbours[myidx]=ret_index[k];
				float df = expf(-ret_dist[k]*NNDistFactor);
				sumDF += df;
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;


			if(lvl < pyrLevelsUsed-1 )
			{
				resultSet1.init(ret_index, ret_dist);
				pt = pt*0.5f-Vec2f(0.25f,0.25f);
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				pts[i].parent = ret_index[0];
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.

	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}


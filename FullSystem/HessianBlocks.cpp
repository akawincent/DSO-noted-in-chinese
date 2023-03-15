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


 
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso
{


PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib)
{
	instanceCounter++;
	host = rawPoint->host;
	hasDepthPrior=false;

	idepth_hessian=0;
	maxRelBaseline=0;
	numGoodResiduals=0;

	// set static values & initialization.
	u = rawPoint->u;
	v = rawPoint->v;
	assert(std::isfinite(rawPoint->idepth_max));
	//idepth_init = rawPoint->idepth_GT;

	my_type = rawPoint->my_type;
	//计算idepth和idepth_scaled
	setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
	//设置点的状态PtStatus
	setPointStatus(PointHessian::INACTIVE);

	int n = patternNum;
	//把immature point初始化时计算出来的像素灰度 以及灰度权重wp都传给pointHessian
	memcpy(color, rawPoint->color, sizeof(float)*n);
	memcpy(weights, rawPoint->weights, sizeof(float)*n);
	energyTH = rawPoint->energyTH;
	//能量设为0
	efPoint=0;
}


void PointHessian::release()
{
	for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
	residuals.clear();
}

/******************* 计算零空间 ********************/
/* parameter：
		state_zero ——————————> FrameHessian中的state
*/
void FrameHessian::setStateZero(const Vec10 &state_zero)
{
	assert(state_zero.head<6>().squaredNorm() < 1e-20);
	

	//把state状态量固定线性化 用state_zero存一下
	this->state_zero = state_zero;


	for(int i=0;i<6;i++)
	{
		Vec6 eps; eps.setZero(); eps[i] = 1e-3;
		SE3 EepsP = Sophus::SE3::exp(eps);
		SE3 EepsM = Sophus::SE3::exp(-eps);
		SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT() * EepsP) * get_worldToCam_evalPT().inverse();
		SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT() * EepsM) * get_worldToCam_evalPT().inverse();
		nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
	}
	//nullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
	//nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

	// scale change
	SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_P_x0.translation() *= 1.00001;
	w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();
	SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
	w2c_leftEps_M_x0.translation() /= 1.00001;
	w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();
	nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);


	nullspaces_affine.setZero();
	nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
	assert(ab_exposure > 0);
	nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0().a)*ab_exposure);
};



void FrameHessian::release()
{
	// DELETE POINT
	// DELETE RESIDUAL
	for(unsigned int i=0;i<pointHessians.size();i++) delete pointHessians[i];
	for(unsigned int i=0;i<pointHessiansMarginalized.size();i++) delete pointHessiansMarginalized[i];
	for(unsigned int i=0;i<pointHessiansOut.size();i++) delete pointHessiansOut[i];
	for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];


	pointHessians.clear();
	pointHessiansMarginalized.clear();
	pointHessiansOut.clear();
	immaturePoints.clear();
}

/*************为当前帧建立图像金字塔 并计算每层每个像素的图像梯度以及辐射值***************/
/*parameter:
		color  ——————> 读入图像的灰度值  ImageAndExposure类中的成员变量image  是个指针
		HCalib ——————> Hcalib是Fullsystem中持有的CalibHessian类的对象  Hcalib中存有相机内参矩阵
*/
void FrameHessian::makeImages(float* color, CalibHessian* HCalib)
{

	for(int i=0;i<pyrLevelsUsed;i++)
	{
		// 该金字塔等级的dIp创建了该等级图像所有像素的 辐射度 x方向梯度 y方向梯度的存储空间
		dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];		//这里是把图像的行列变为了一维数组存储
		// 该金字塔等级的absSquaredGrad 创建了该等级图像素有像素的 梯度平方和的存储空间
		absSquaredGrad[i] = new float[wG[i]*hG[i]];		//这里是把图像的行列变为了一维数组存储
	}
	// dI存储了原图像的像素梯度及灰度信息
	dI = dIp[0];


	// make d0
	int w=wG[0];
	int h=hG[0];
	//从ImageAndExposure的color中把图像灰度信息传到FrameHessian中的dI
	for(int i=0;i<w*h;i++)
		dI[i][0] = color[i];

	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		// 当前金字塔等级图像的宽和高
		int wl = wG[lvl], hl = hG[lvl];
		// 当前金字塔等级图像的dIp像素信息
		Eigen::Vector3f* dI_l = dIp[lvl];
		// 当前金字塔等级图像的梯度平方和
		float* dabs_l = absSquaredGrad[lvl];

		if(lvl>0)
		{
			//上一级金字塔图像的信息
			int lvlm1 = lvl-1;
			int wlm1 = wG[lvlm1];
			Eigen::Vector3f* dI_lm = dIp[lvlm1];

			//以2倍大小构建下一级金字塔图像的灰度
			for(int y=0;y<hl;y++)
				for(int x=0;x<wl;x++)
				{
					//通过上一级金字塔图像的灰度 取相邻4个点灰度求平均作为下一级金字塔图像对应像素位置的灰度
					dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1][0] +
												dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
												dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
				}
		}

		// 求x方向和y方向的图像梯度以及梯度平方和
		// 注意这里忽略了图像边缘位置像素的梯度  没有计算
		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
			float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);


			if(!std::isfinite(dx)) dx=0;
			if(!std::isfinite(dy)) dy=0;

			dI_l[idx][1] = dx;
			dI_l[idx][2] = dy;


			dabs_l[idx] = dx*dx+dy*dy;

			//根据参数设置给梯度乘以一个权重
			if(setting_gammaWeightsPixelSelect==1 && HCalib!=0)
			{
				float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
				dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
			}
		}
	}
}

void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib )
{
	this->host = host;
	this->target = target;
	//host与target之间的姿态变换  固定线性化点的
	SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
	//旋转部分
	PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
	//平移部分
	PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();


	//带着优化时增量的的host与target之间的姿态变换  这里应该是LM迭代时要的 用于计算误差(这个东西是需要随着优化一直更新的)
	SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
	PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
	PRE_tTll = (leftToLeft.translation()).cast<float>();
	//计算两帧之间的距离
	distanceLL = leftToLeft.translation().norm();

	//拿到相机内参矩阵
	Mat33f K = Mat33f::Zero();
	K(0,0) = HCalib->fxl();
	K(1,1) = HCalib->fyl();
	K(0,2) = HCalib->cxl();
	K(1,2) = HCalib->cyl();
	K(2,2) = 1;
	//K*旋转*K^-1
	PRE_KRKiTll = K * PRE_RTll * K.inverse();
	//旋转*K^-1
	PRE_RKiTll = PRE_RTll * K.inverse();
	//K*平移
	PRE_KtTll = K * PRE_tTll;

	//拿到host与target两帧之间的相对光度变换参数
	PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
	//拿到线性化后的b参数？
	PRE_b0_mode = host->aff_g2l_0().b;
}

}


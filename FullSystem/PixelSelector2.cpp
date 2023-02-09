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


#include "FullSystem/PixelSelector2.h"
 
// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
	//randomPattern内是0~256的随机数
	randomPattern = new unsigned char[w*h];
	std::srand(3141592);	// want to be deterministic.
	for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;


	//选点范围
	currentPotential=3;

	//分配动态内存
	gradHist = new int[100*(1+w/32)*(1+h/32)];
	ths = new float[(w/32)*(h/32)+100];
	thsSmoothed = new float[(w/32)*(h/32)+100];

	allowFast=false;
	gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
	delete[] randomPattern;
	delete[] gradHist;
	delete[] ths;
	delete[] thsSmoothed;
}

/************ 选阈值的方式看不懂..... *********/
int computeHistQuantil(int* hist, float below)
{
	int th = hist[0]*below+0.5f;
	for(int i=0;i<90;i++)
	{
		th -= hist[i+1];
		if(th<0) return i;
	}
	return 90;
}

/******************* 对每个32*32的邻域创建梯度直方图 ******************/
void PixelSelector::makeHists(const FrameHessian* const fh)
{
	//此时PixelSelector类中的成员变量gradHistFrame就是传进来的Frame数据结构fh
	gradHistFrame = fh;
	//第0层图像所有像素的梯度平方和 
	//mapmax0存储的是absSquaredGrad[0]的首地址 也就是第一个像素位置梯度平方和的索引 
	//可以通过在mapmax0上做加法并解地址 便可以求出对应像素位置的梯度平方和
	float * mapmax0 = fh->absSquaredGrad[0];		//fh->absSquaredGrad是个指针数组

	//第0层金字塔原始图像的宽和高
	int w = wG[0];
	int h = hG[0];

	//第0层图像分成32*32的邻域
	int w32 = w/32;
	int h32 = h/32;
	//这个变量是在select函数中用到的  
	thsStep = w32;

	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			//32 * 32邻域的梯度平方和的索引  每个邻域开始的首地址
			float* map0 = mapmax0+32*x+32*y*w;
			//再来个整型指针保存gradHist
			int* hist0 = gradHist;// + 50*(x+y*w32);
			//给hist0后面又分配了50个整形数据大小的内存空间 初始化为0
			memset(hist0,0,sizeof(int)*50);

			//遍历邻域
			for(int j=0;j<32;j++) 
				for(int i=0;i<32;i++)
			{
				int it = i+32*x;
				int jt = j+32*y;
				if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;
				// 求出该像素位置梯度平方和的平方根
				int g = sqrtf(map0[i+j*w]);
				// 限制g最大为48
				if(g>48) g=48;
				// hist0[1~49]中存储着48个梯度值的数量
				hist0[g+1]++;
				// hist[0]中存储了所有梯度值的数量 32 * 32 = 1024
				hist0[0]++;
			}

			//计算得到所有32*32邻域的阈值 并填入ths数组 总共填入w/32*h/32个阈值
			//setting_minGradHistCut = 0.5;	setting_minGradHistAdd = 7;
			ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
		}

	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
	
			float sum=0,num=0;
			if(x>0)
			{
				if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
				num++; sum+=ths[x-1+(y)*w32];
			}

			if(x<w32-1)
			{
				if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
				num++; sum+=ths[x+1+(y)*w32];
			}

			if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
			num++; sum+=ths[x+y*w32];

			//对阈值矩阵ths做3*3的均值滤波得到平滑后的阈值矩阵thsSmoothed
			thsSmoothed[x+y*w32] = (sum/num) * (sum/num);
		}
}

/****************根据梯度平方和直方图的中位数阈值矩阵选择图像金字塔第0层的像素点***************/
/******************************** 递归函数动态调整pot大小 *********************************/
/* parameter:
		fh              ——————————> 帧Frame的数据结构
		map_out         ——————————> 选出的像素点
		density         ——————————> 每一层金字塔选点的稠密程度
		recursionLeft   ——————————> 最大递归次数
		thFactor        ——————————> 阈值因子

		note: 该函数是个递归 在内部调用自己 
*/
int PixelSelector::makeMaps(
		const FrameHessian* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{

	float numHave=0;
	//想要的点的个数  numWant = density[lvl] * w * h
	float numWant=density;
	float quotia;
	int idealPotential = currentPotential;


//	if(setting_pixelSelectionUseFast>0 && allowFast)
//	{
//		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
//		std::vector<cv::KeyPoint> pts;
//		cv::Mat img8u(hG[0],wG[0],CV_8U);
//		for(int i=0;i<wG[0]*hG[0];i++)
//		{
//			float v = fh->dI[i][0]*0.8;
//			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
//		}
//		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
//		for(unsigned int i=0;i<pts.size();i++)
//		{
//			int x = pts[i].pt.x+0.5;
//			int y = pts[i].pt.y+0.5;
//			map_out[x+y*wG[0]]=1;
//			numHave++;
//		}
//
//		printf("FAST selection: got %f / %f!\n", numHave, numWant);
//		quotia = numWant / numHave;
//	}
//	else
	{




		// the number of selected pixels behaves approximately as
		// K / (pot+1)^2, where K is a scene-dependent constant.
		// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.
		
		//创建梯度直方图保存并计算得到阈值矩阵 保存在PixelSelector中的ths和thsSmoothed
		if(fh != gradHistFrame) makeHists(fh);

		// select! 在当前帧上选点
		//尽管选点策略在使用三个尺度pot邻域时用到了第0~2层图像
		//然而最后选出的点仍然是基于第0层图像的
		Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor);

		// sub-select!
		numHave = n[0]+n[1]+n[2];		//提取到的像素点的数量(3个尺度)
		quotia = numWant / numHave;		//想要的点的数量/选择的点的数量

		// by default we want to over-sample by 40% just to be sure.

		// numHave:选择出点的数量
		// (currentPotential+1)^2:一个当前邻域的面积大小
		// K：因为一个邻域一个像素点 所以K就是所有选择出的像素点的邻域总的覆盖的面积
		// K可以在这个函数内认为是一个常数(随着递归 K也是动态变化的)
		float K = numHave * (currentPotential+1) * (currentPotential+1);
		// numWant:理想的选择像素点数量
		// idealPotential:为了达到numWant数量的像素点而所需的理想的邻域范围
		// 实现了pot根据选点数量而动态调整 pot越大 选点数量越少
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

		/*************************实际选的点少了****************************/
		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential)
				idealPotential = currentPotential-1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);

			//调整currentPotential 使其变小
			currentPotential = idealPotential;
			//重新选点
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
		}
		/************************实际选的点多了****************************/
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// re-sample to get less points!

			if(idealPotential<=currentPotential)
				idealPotential = currentPotential+1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);

			//调整currentPotential 使其变大 
			currentPotential = idealPotential;、
			//重新选点
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

		}
	}

	int numHaveSub = numHave;
	//选的点还是稍微有点多
	if(quotia < 0.95)
	{
		
		int wh=wG[0]*hG[0];
		int rn=0;
		unsigned char charTH = 255*quotia;
		//遍历第0层图像所有图像
		for(int i=0;i<wh;i++)
		{
			if(map_out[i] != 0)
			{	
				//随机去掉一些已经选择的点
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
//			100*numHave/(float)(wG[0]*hG[0]),
//			100*numWant/(float)(wG[0]*hG[0]),
//			currentPotential,
//			idealPotential,
//			100*numHaveSub/(float)(wG[0]*hG[0]));

	//即使没有出现以上的情况  也要将currentPotential更新
	currentPotential = idealPotential;

	//展示选点结果 
	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7;
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}
		IOWrap::displayImage("Selector Image", &img);

		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)			//小尺度选取的点 绿色
					img.setPixelCirc(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)	//中尺度选取的点 蓝色
					img.setPixelCirc(x,y,Vec3b(255,0,0));
				else if(map_out[i] == 4)	//中尺度选取的点 红色
					img.setPixelCirc(x,y,Vec3b(0,0,255));
			}
		IOWrap::displayImage("Selector Pixels", &img);
	}

	//返回的是选点数量
	return numHaveSub;
}


/**************************** 基于不同的pot和尺度上选点函数 *******************************/
/*parameter:
		fh 			——————————> 帧的数据结构
		map_out 	——————————> 选中的像素点
		pot 		——————————> 选点的范围(在一个pot范围内选一个点) 这里传入的是currentPotential
		thFactor 	——————————> 阈值尺度因子 用于放大经过阈值矩阵选择后的直方图
*/
Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
		float* map_out, int pot, float thFactor)
{
	//fh->dI是第0层图像
	//*右边的const 表示指针map0不可更改  *左边的const 表示指针map0指向的内容不可更改
	Eigen::Vector3f const * const map0 = fh->dI;

	float * mapmax0 = fh->absSquaredGrad[0];	//第0层图像的所有像素的梯度平方和
	float * mapmax1 = fh->absSquaredGrad[1];	//第1层图像的所有像素的梯度平方和
	float * mapmax2 = fh->absSquaredGrad[2];	//第2层图像的所有像素的梯度平方和

	//选第0层图像需要用到0 1 2层图像的梯度平方和，配合着不同的pot大小
	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

	//方向向量 模为1 以圆心为(0,0)，半径为1的圆形内16个方向
	const Vec2f directions[16] = {
	         Vec2f(0,    1.0000),
	         Vec2f(0.3827,    0.9239),
	         Vec2f(0.1951,    0.9808),
	         Vec2f(0.9239,    0.3827),
	         Vec2f(0.7071,    0.7071),
	         Vec2f(0.3827,   -0.9239),
	         Vec2f(0.8315,    0.5556),
	         Vec2f(0.8315,   -0.5556),
	         Vec2f(0.5556,   -0.8315),
	         Vec2f(0.9808,    0.1951),
	         Vec2f(0.9239,   -0.3827),
	         Vec2f(0.7071,   -0.7071),
	         Vec2f(0.5556,    0.8315),
	         Vec2f(0.9808,   -0.1951),
	         Vec2f(1.0000,    0.0000),
	         Vec2f(0.1951,   -0.9808)};

	//重新给map_out清零
	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));


	//setting_gradDownweightPerLevel = 0.75
	//金字塔层阈值的减小倍数
	float dw1 = setting_gradDownweightPerLevel;	//第二层阈值减小的倍数
	float dw2 = dw1*dw1;						//第三层阈值减小的倍数

	//在不同尺度下选取点的数量
	int n3=0, n2=0, n4=0;

	//第一层循环遍历第0层图像时是每隔着4*pot个点进行遍历的
	for(int y4=0;y4<h;y4+=(4*pot)) 
		for(int x4=0;x4<w;x4+=(4*pot))
	{
		//实际上my3和mx3永远是4*pot 其中pot传进来是3  
		//my3和mx3代表了此时该点处的邻域大小 12*12
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		int bestIdx4=-1; float bestVal4=0;

		//从directions中选择随机方向
		Vec2f dir4 = directions[randomPattern[n2] & 0xF];

		//在上面计算出的邻域内再进行遍历
		for(int y3=0;y3<my3;y3+=(2*pot)) 
			for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			//在第0层图像中的像素坐标
			int x34 = x3+x4;
			int y34 = y3+y4;

			//计算下一个尺度的邻域大小 6*6
			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			int bestIdx3=-1; float bestVal3=0;
			//随机方向
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];

			//在上一级尺度中遍历
			for(int y2=0;y2<my2;y2+=pot) 
				for(int x2=0;x2<mx2;x2+=pot)
			{
				//第0层原图像的像素坐标
				int x234 = x2+x34;
				int y234 = y2+y34;
				//计算邻域大小 3*3
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				int bestIdx2=-1; float bestVal2=0;
				//随机方向
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];

				//最后一个尺度 遍历时是一个一个像素无间隔的遍历
				for(int y1=0;y1<my1;y1+=1) 
					for(int x1=0;x1<mx1;x1+=1)
				{
					assert(x1+x234 < w);
					assert(y1+y234 < h);
					//idx是在第0层原图像中第几个像素的一维索引
					int idx = x1+x234 + w*(y1+y234);
					//xf和yf是第0层原图像的坐标
					int xf = x1+x234;
					int yf = y1+y234;

					//排除第0层原图像边缘的像素
					if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;

					//thsStep在makeHists函数中被赋值为w32
					//(xf>>5) + (yf>>5) * thsStep = xf/32 + yf/32 * w32 
					//上式可以确定当前遍历到的像素在哪个32*32的阈值邻域中
					//pixelTH0就得到了当前像素对应的第一级尺度下的阈值
					float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];
					//第二级尺度下的阈值大小 0.75 * threshold
					float pixelTH1 = pixelTH0 * dw1;
					//第三级尺度下的阈值大小 0.75 * 0.75 * threshold
					float pixelTH2 = pixelTH1 * dw2;

					//当前像素在第0层图像的梯度平方和
					float ag0 = mapmax0[idx];
					//满足大于阈值的条件 thFactor = 2
					if(ag0 > pixelTH0*thFactor)
					{
						//拿到当前像素的x y方向梯度
						Vec2f ag0d = map0[idx].tail<2>();
						//将梯度方向投影到随机方向dir2上  内积的大小可以体现两个向量的夹角大小
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));
						//setting_selectDirectionDistribution = true
						if(!setting_selectDirectionDistribution) dirNorm = ag0;
						//取内积最大的梯度
						if(dirNorm > bestVal2)
						{ bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
					}
					//在第0层选点了 则不在其他层选点 但还是会继续遍历当前层的pot邻域 找打最大的bestVal2
					if(bestIdx3==-2) continue;

					//第一层的梯度平方和  (int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1是第一层像素的index索引
					// * 0.5 代表了金字塔的2倍下采样
					// 一般的 满足不了第0层的阈值条件才会到这里
					float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();		//这里的梯度是第一层的
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
					}
					if(bestIdx4==-2) continue;

					//第二层选点 满足不了第一层的阈值条件
					float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();		//这里的梯度也是第一层的
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = idx; }
					}
				}

				//在每个3*3的pot中根据第0层的梯度满足阈值条件选出bestVal2并标记对应像素的索引
				//如果没有满足阈值条件的像素 则bestIdx2 = -1
				if(bestIdx2>0)
				{
					map_out[bestIdx2] = 1;		//最小的尺度 pixel-wise
					bestVal3 = 1e10;
					n2++;		//第0层找到点计数
				}
			}

			//第0层没找到 则根据第1层梯度阈值在6*6范围内选出一个点 bestIdx3
			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;		//中等尺度
				bestVal4 = 1e10;
				n3++;		//计数
			}
		}
		//第1层没找到 则根据第2层梯度阈值在12*12范围内选出一个点 bestIdx4
		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;		//最大的尺度
			n4++;
		}
	}

	//map_out中已经标记了哪些像素被选择了

	/* 返回一个向量  n2:根据第0层梯度阈值条件选出点的个数  每个3*3邻域内选出一个best
					n3:根据第1层梯度阈值条件选出点的个数  每个6*6邻域内选出一个best
					n3:根据第2层梯度阈值条件选出点的个数  每个12*12邻域内选出一个best
	*/
	return Eigen::Vector3i(n2,n3,n4);
}




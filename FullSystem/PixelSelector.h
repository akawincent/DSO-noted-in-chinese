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


 

namespace dso
{


const float minUseGrad_pixsel = 10;

//模板函数 pot为邻域大小 在一个pot内选择满足阈值条件的最大梯度的像素点
template<int pot>
inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, float THFac)
{

	memset(map_out, 0, sizeof(bool)*w*h);

	int numGood = 0;
	for(int y=1;y<h-pot;y+=pot)
	{
		for(int x=1;x<w-pot;x+=pot)
		{
			int bestXXID = -1;
			int bestYYID = -1;
			int bestXYID = -1;
			int bestYXID = -1;

			float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

			//每个邻域开始的首像素
			Eigen::Vector3f* grads0 = grads+x+y*w;
			//在邻域内遍历
			for(int dx=0;dx<pot;dx++)
				for(int dy=0;dy<pot;dy++)
				{	
					//当前像素在邻域内的索引
					int idx = dx+dy*w;
					//g为当前像素的三通道信息
					Eigen::Vector3f g=grads0[idx];
					//两个方向梯度的平方和
					float sqgd = g.tail<2>().squaredNorm();
					//minUseGrad_pixsel = 10
					//阈值 = (1 * 10 * 0.75)^2
					float TH = THFac*minUseGrad_pixsel * (0.75f);
					//满足阈值条件 记录下最大的agx agy gxpy gxmy以及对应的索引
					if(sqgd > TH*TH)
					{
						float agx = fabs((float)g[1]);
						if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

						float agy = fabs((float)g[2]);
						if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

						float gxpy = fabs((float)(g[1]-g[2]));
						if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

						float gxmy = fabs((float)(g[1]+g[2]));
						if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
					}
				}

			//map_out记录被选择的像素点
			bool* map0 = map_out+x+y*w;

			if(bestXXID>=0)
			{
				if(!map0[bestXXID])
					numGood++;
				map0[bestXXID] = true;

			}
			if(bestYYID>=0)
			{
				if(!map0[bestYYID])
					numGood++;
				map0[bestYYID] = true;

			}
			if(bestXYID>=0)
			{
				if(!map0[bestXYID])
					numGood++;
				map0[bestXYID] = true;

			}
			if(bestYXID>=0)
			{
				if(!map0[bestYXID])
					numGood++;
				map0[bestYXID] = true;

			}
		}
	}
	//返回的是选点数量
	return numGood;
}


inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, int pot, float THFac)
{

	memset(map_out, 0, sizeof(bool)*w*h);

	int numGood = 0;
	for(int y=1;y<h-pot;y+=pot)
	{
		for(int x=1;x<w-pot;x+=pot)
		{
			int bestXXID = -1;
			int bestYYID = -1;
			int bestXYID = -1;
			int bestYXID = -1;

			float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

			Eigen::Vector3f* grads0 = grads+x+y*w;
			for(int dx=0;dx<pot;dx++)
				for(int dy=0;dy<pot;dy++)
				{
					int idx = dx+dy*w;
					Eigen::Vector3f g=grads0[idx];
					float sqgd = g.tail<2>().squaredNorm();
					float TH = THFac*minUseGrad_pixsel * (0.75f);

					if(sqgd > TH*TH)
					{
						float agx = fabs((float)g[1]);
						if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

						float agy = fabs((float)g[2]);
						if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

						float gxpy = fabs((float)(g[1]-g[2]));
						if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

						float gxmy = fabs((float)(g[1]+g[2]));
						if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
					}
				}

			bool* map0 = map_out+x+y*w;

			if(bestXXID>=0)
			{
				if(!map0[bestXXID])
					numGood++;
				map0[bestXXID] = true;

			}
			if(bestYYID>=0)
			{
				if(!map0[bestYYID])
					numGood++;
				map0[bestYYID] = true;

			}
			if(bestXYID>=0)
			{
				if(!map0[bestXYID])
					numGood++;
				map0[bestXYID] = true;

			}
			if(bestYXID>=0)
			{
				if(!map0[bestYXID])
					numGood++;
				map0[bestYXID] = true;

			}
		}
	}

	return numGood;
}

/**************************图像第1~5层的选点函数**************************/
/* parameter:
		grads 			——————> 	lvl等级的图像所有像素信息 三通道:灰度值 x方向导数 y方向导数
		map 			——————> 	用于标记选择的像素点
		w 				——————> 	lvl等级图像的宽
		h 				——————> 	lvl等级图像的高
		desireDensity 	——————> 	想要的点的数量 density[lvl] * w[0] * h[0]
		recsLeft 		——————> 	递归次数 默认5次
		THFac 			——————> 	阈值乘数因子
*/
inline int makePixelStatus(Eigen::Vector3f* grads, bool* map, int w, int h, float desiredDensity, int recsLeft=5, float THFac = 1)
{
	//settings.h中定义初始时sparsityFactor = 5
	//sparsityFactor就是pot的大小
	if(sparsityFactor < 1) sparsityFactor = 1;
	//选出点的数量
	int numGoodPoints;
	//邻域为1 实际上就是pixel-wise范围内选一个点
	if(sparsityFactor==1) numGoodPoints = gridMaxSelection<1>(grads, map, w, h, THFac);
	else if(sparsityFactor==2) numGoodPoints = gridMaxSelection<2>(grads, map, w, h, THFac);
	else if(sparsityFactor==3) numGoodPoints = gridMaxSelection<3>(grads, map, w, h, THFac);
	else if(sparsityFactor==4) numGoodPoints = gridMaxSelection<4>(grads, map, w, h, THFac);
	else if(sparsityFactor==5) numGoodPoints = gridMaxSelection<5>(grads, map, w, h, THFac);
	else if(sparsityFactor==6) numGoodPoints = gridMaxSelection<6>(grads, map, w, h, THFac);
	else if(sparsityFactor==7) numGoodPoints = gridMaxSelection<7>(grads, map, w, h, THFac);
	else if(sparsityFactor==8) numGoodPoints = gridMaxSelection<8>(grads, map, w, h, THFac);
	else if(sparsityFactor==9) numGoodPoints = gridMaxSelection<9>(grads, map, w, h, THFac);
	else if(sparsityFactor==10) numGoodPoints = gridMaxSelection<10>(grads, map, w, h, THFac);
	else if(sparsityFactor==11) numGoodPoints = gridMaxSelection<11>(grads, map, w, h, THFac);
	else numGoodPoints = gridMaxSelection(grads, map, w, h, sparsityFactor, THFac);


	/*
	 * #points is approximately proportional to sparsityFactor^2.
	 */
	//比例
	float quotia = numGoodPoints / (float)(desiredDensity);
	//计算新的pot
	int newSparsity = (sparsityFactor * sqrtf(quotia))+0.7f;
	if(newSparsity < 1) newSparsity=1;

	float oldTHFac = THFac;

	//这代表上一次递归想要选的点较多 这一次递归也想要选更多的点 这时候要放低阈值条件
	if(newSparsity==1 && sparsityFactor==1) THFac = 0.5;

	//满足该条件表明选点数量符合要求 则直接返回numGoodPoints
	//若不满足 则递归重新找点
	if((abs(newSparsity-sparsityFactor) < 1 && THFac==oldTHFac) ||
			( quotia > 0.8 &&  1.0f / quotia > 0.8) ||
			recsLeft == 0)
	{

//		printf(" \n");
		//all good
		sparsityFactor = newSparsity;
		return numGoodPoints;
	}
	else
	{
//		printf(" -> re-evaluate! \n");
		// re-evaluate.
		sparsityFactor = newSparsity;
		return makePixelStatus(grads, map, w,h, desiredDensity, recsLeft-1, THFac);
	}
}

}


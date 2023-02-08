/*
 * @Author: error: git config user.name && git config user.email & please set dead value or install git
 * @Date: 2022-09-01 16:55:29
 * @LastEditors: error: git config user.name && git config user.email & please set dead value or install git
 * @LastEditTime: 2023-01-06 15:27:42
 * @FilePath: \dso-master\dso-master\src\util\ImageAndExposure.h
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
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
#include <cstring>
#include <iostream>


namespace dso
{


class ImageAndExposure
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	//保存灰度值
	float* image;			// irradiance. between 0 and 256
	//图像的宽和高
	int w,h;				// width and height;
	//时间戳
	double timestamp;
	//曝光时间
	float exposure_time;	// exposure time in ms.
	//构造函数
	inline ImageAndExposure(int w_, int h_, double timestamp_=0) : w(w_), h(h_), timestamp(timestamp_)
	{
		image = new float[w*h];	//new了一个w*h大小的浮点型指针作为image，注意数组是一维的，不是图像直观上的二维
		exposure_time=1;		//初始化曝光时间为1ms
	}
	//析构函数 对于new出来的对象 要专门delete 防止内存泄漏
	inline ~ImageAndExposure()
	{
		delete[] image;
	}
	//复制曝光时间 将this->exposure_time传给other->exposure_time
	inline void copyMetaTo(ImageAndExposure &other)
	{
		other.exposure_time = exposure_time;
	}

	//深拷贝一个ImageAndExposure对象
	inline ImageAndExposure* getDeepCopy()
	{
		ImageAndExposure* img = new ImageAndExposure(w,h,timestamp);
		img->exposure_time = exposure_time;
		memcpy(img->image, image, w*h*sizeof(float));
		return img;
	}
};


}

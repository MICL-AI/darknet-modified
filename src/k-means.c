#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define N 11
#define K 3
#define KM_EPSILON 0.1f

typedef struct Point
{
	float x;
	float y;
	struct Point *next;
} Point;

int center[N]; ///  判断每个点属于哪个簇

Point point[N] = {
	{2.0, 10.0},
	{2.0, 5.0},
	{8.0, 4.0},
	{5.0, 8.0},
	{7.0, 5.0},
	{6.0, 4.0},
	{1.0, 2.0},
	{4.0, 9.0},
	{7.0, 3.0},
	{1.0, 3.0},
	{3.0, 9.0}};

Point mean[K]; ///  保存每个簇的中心点

float getDistance(Point point1, Point point2)
{
	float d;
	d = sqrt((point1.x - point2.x) * (point1.x - point2.x) + (point1.y - point2.y) * (point1.y - point2.y));
	return d;
}

/// 计算每个簇的中心点
void getMean(int center[N])
{
	Point tep;
	int i, j, count = 0;
	for (i = 0; i < K; ++i)
	{
		count = 0;
		tep.x = 0.0; /// 每算出一个簇的中心点值后清0
		tep.y = 0.0;
		for (j = 0; j < N; ++j)
		{
			if (i == center[j])
			{
				count++;
				tep.x += point[j].x;
				tep.y += point[j].y;
			}
		}
		tep.x /= count;
		tep.y /= count;
		mean[i] = tep;
	}
	for (i = 0; i < K; ++i)
	{
		printf("The new center point of %d is : \t( %f, %f )\n", i + 1, mean[i].x, mean[i].y);
	}
}

/// 计算平方误差函数
float getE()
{
	int i, j;
	float cnt = 0.0, sum = 0.0;
	for (i = 0; i < K; ++i)
	{
		for (j = 0; j < N; ++j)
		{
			if (i == center[j])
			{
				cnt = (point[j].x - mean[i].x) * (point[j].x - mean[i].x) + (point[j].y - mean[i].y) * (point[j].y - mean[i].y);
				sum += cnt;
			}
		}
	}
	return sum;
}

/// 把N个点聚类
void cluster()
{
	int i, j, q;
	float min;
	float distance[N][K];
	for (i = 0; i < N; ++i)
	{
		min = 999999.0;
		for (j = 0; j < K; ++j)
		{
			distance[i][j] = getDistance(point[i], mean[j]);

			/// printf("%f\n", distance[i][j]);  /// 可以用来测试对于每个点与3个中心点之间的距离
		}
		for (q = 0; q < K; ++q)
		{
			if (distance[i][q] < min)
			{
				min = distance[i][q];
				center[i] = q;
			}
		}
		printf("( %.0f, %.0f )\t in cluster-%d\n", point[i].x, point[i].y, center[i] + 1);
	}
	printf("-----------------------------\n");
}

Point *fmap_point(float *fmap, int channel, int width, int height)
{
	Point point[100];
	int k = 0;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
			if (fabs(fmap[channel * height * width + i * width + j]) > KM_EPSILON)
			{
				// printf("point%d={%d,%d}\n", k, i, j);
				point[k].x = i;
				point[k].y = j;
				k++;
			}
	}
	return &point;
}
int main()
{
	int i, j, n = 0;
	float temp1;
	float temp2, t;

	float fmap[25] = {
		1, 3, 2, 0, 0,
		1, 3, 3, 0, 0,
		0, 0, 0, 0, 0,
		0, 0, 0, 0, 1,
		0, 0, 0, 0, 1};
	Point *p = fmap_point(&fmap, 0, 5, 5);

	printf("----------Data sets----------\n");
	for (i = 0; i < 25; ++i)
	{
		printf("\t( %.0f, %.0f )\n", &p[i].x, &p[i].y);
	}
	printf("-----------------------------\n");

	/*
    可以选择当前时间为随机数
    srand((unsigned int)time(NULL));
    for(i = 0; i < K; ++i)
    {
    	j = rand() % K;
    	mean[i].x = point[j].x;
    	mean[i].y = point[j].y;
    }
*/
	// mean[0].x = point[0].x; /// 初始化k个中心点
	// mean[0].y = point[0].y;

	// mean[1].x = point[3].x;
	// mean[1].y = point[3].y;

	// mean[2].x = point[6].x;
	// mean[2].y = point[6].y;

	// cluster();		/// 第一次根据预设的k个点进行聚类
	// temp1 = getE(); ///  第一次平方误差
	// n++;			///  n计算形成最终的簇用了多少次

	// printf("The E1 is: %f\n\n", temp1);

	// getMean(center);
	// cluster();
	// temp2 = getE(); ///  根据簇形成新的中心点，并计算出平方误差
	// n++;

	// printf("The E2 is: %f\n\n", temp2);

	// while (fabs(temp2 - temp1) != 0) ///  比较两次平方误差 判断是否相等，不相等继续迭代
	// {
	// 	temp1 = temp2;
	// 	getMean(center);
	// 	cluster();
	// 	temp2 = getE();
	// 	n++;
	// 	printf("The E%d is: %f\n", n, temp2);
	// }

	// printf("The total number of cluster is: %d\n\n", n); /// 统计出迭代次数
	return 0;
}

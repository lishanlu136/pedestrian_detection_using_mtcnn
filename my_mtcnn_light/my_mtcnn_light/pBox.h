#ifndef PBOX_H
#define PBOX_H
#include <stdlib.h>
#include <iostream>

using namespace std;
#define mydataFmt float
//#define modelFromFile
#define modelFromHFile

struct pBox     //����������ͼ
{
	mydataFmt *pdata;
	int width;
	int height;
	int channel;
};

struct pRelu
{
    mydataFmt *pdata;
    int width;
};

struct Weight   //����Ĳ�����Ȩ�ؼ�ƫ�ã�
{
	mydataFmt *pdata;
    mydataFmt *pbias;
    int lastChannel;
    int selfChannel;
	int kernelWidth;
	int kernelHeight;
    int strideW;
	int strideH;
	int padW;
	int padH;
};

struct Bbox     //��ѡ���ο�
{
    float score;
	float x1;
	float y1;
	float x2;
	float y2;
    float area;
    bool exist;
    mydataFmt regreCoord[4];
};

struct orderScore     //���о��ο�
{
    mydataFmt score;
    int oriOrder;
};

void freepBox(struct pBox *pbox);
void freeWeight(struct Weight *weight);
void freepRelu(struct pRelu *prelu);
void pBoxShow(const struct pBox *pbox);
void pBoxShowE(const struct pBox *pbox,int channel, int row);
void weightShow(const struct Weight *weight);
void pReluShow(const struct pRelu *prelu);
#endif
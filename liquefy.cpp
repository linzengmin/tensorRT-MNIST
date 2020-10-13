#include "liquefy.h"
#include <iostream>
void mapping(warp *w,int x,int y, double *u,double *v)
{
	double fu,fv;
	fu = x;
	fv = y;
	double dx = fu - w->fOldPointX;
	double dy = fv - w->fOldPointY;
	double rsq = dx*dx+dy*dy;
	if(dx>0.0-w->radius&& dx<w->radius&& dy>0.0-w->radius&& dy<w->radius&& rsq<  w->radius*w->radius)
	{
		double cmx = w->fNewPointX - w->fOldPointX;
		double cmy = w->fNewPointY - w->fOldPointY;
		//double msq = (dx-cmx)*(dx-cmx) +(dy-cmy)*(dy-cmy);
		double msq = cmx*cmx + cmy*cmy;
		double edge_dist = w->radius*w->radius -rsq;
		double a   = edge_dist/(edge_dist+msq);
		a *= a;
		fu -= a*cmx;
		fv -= a*cmy;
		*u = fu;
		*v = fv;
	}else
	{
		*u = x;
		*v = y;
	}
}
void scanning(warp *w,unsigned char *pOldColorData,unsigned char *pNewColorData)
{
    
	for(int i=0;i<w->nHeight;i++)
	{
		for(int j=0;j<w->nWidth;j++)
		{
			double dx = j-w->fNewPointX;
			double dy = i-w->fNewPointY;
			if(dx> (0.0-w->radius)&& dx<(w->radius)&& dy>(0.0-w->radius)&& dy<w->radius&& (dx*dx+dy*dy) < w->radius *w->radius)
			{
				
				double u,v;
				mapping(w,j,i,&u,&v);//寻找映射关系
				if(u>w->nWidth)
				{
                    u = w->nWidth;
                }
				if(v>w->nHeight)
				{
                    v = w->nHeight;
                }
				if(u<0)
                {
					u =0;
                }
				if(v<0)
                {
					v = 0;
                }
				int iu= (int)u;
				int iv = (int)v;
				pNewColorData[i*(w->nWidthStep)+3*j+0] = pOldColorData[iv*(w->nWidthStep)+3*iu+0];
				pNewColorData[i*(w->nWidthStep)+3*j+1] = pOldColorData[iv*(w->nWidthStep)+3*iu+1];
				pNewColorData[i*(w->nWidthStep)+3*j+2] = pOldColorData[iv*(w->nWidthStep)+3*iu+2];

			}//if
			else
			{
				pNewColorData[i*(w->nWidthStep)+3*j+0] = pOldColorData[i*(w->nWidthStep)+3*j+0];
				pNewColorData[i*(w->nWidthStep)+3*j+1] = pOldColorData[i*(w->nWidthStep)+3*j+1];
				pNewColorData[i*(w->nWidthStep)+3*j+2] = pOldColorData[i*(w->nWidthStep)+3*j+2];
			}
		}//width
	}//height 
}
void liquefyExecution(float fOldPointX,float fOldPointY,float fNewPointX,float fNewPointY,int nWidth,int nHeight,unsigned char *pOldColorData,unsigned char *pNewColorData)
{
    int radius=100;
    warp *w = new warp;
	w->radius = radius;
    w->fOldPointX = fOldPointX;
    w->fOldPointY = fOldPointY;
    w->fNewPointX = fNewPointX;
    w->fNewPointY = fNewPointY;
    w->nWidth=nWidth;
    w->nHeight=nHeight;
    w->nWidthStep = (((nWidth*24)+31)/32*4);
    scanning(w,pOldColorData,pNewColorData);//扫描原图，返回液化后对图。
}

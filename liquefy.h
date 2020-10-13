//-------------------------------------------------------------------------------------------------
/// @brief 液化算法
/// @author Meitu
/// @date 2020/10/12
/// @note
/// @version 1.0.0
///-------------------------------------------------------------------------------------------------
//#ifndef  _H_LIQUEFIED_H_              
//#define  _H_LIQUEFIED_H_      
///-------------------------------------------------------------------------------------------------
///
/// @brief	算法入口
///
/// @author	Meitu
/// @date	2020/10/12
///
/// @param fOldPointX	         The x of old point.
/// @param fOldPointY	         The y of old point.
/// @param fNewPointX	         The x of new point.
/// @param fNewPointY	         The y of new point.
/// @param nWidth	             The width.
/// @param nHeight               The length.
/// @param [in]  pOldColorData   If non-null, destination landmark.
/// @param [out] pNewColorData   If non-null, destination landmark.
///
/// @return	NULL
///-------------------------------------------------------------------------------------------------
void liquefyExecution(float fOldPointX,float fOldPointY,float fNewPointX,float fNewPointY,int nWidth,int nHeight,unsigned char *pOldColorData,unsigned char *pNewColorData);
///-------------------------------------------------------------------------------------------------
///
/// @brief	液化函数对结构体，用于保存点击事件的值以及图像大小、圆的半径和位图的实际宽度
///
/// @author	Meitu
/// @date	2020/10/12
///
/// @param fOldPointX	         The x of old point.
/// @param fOldPointY	         The y of old point.
/// @param fNewPointX	         The x of new point.
/// @param fNewPointY	         The y of new point.
/// @param radius	             The radius.
/// @param nWidth	             The width.
/// @param nHeight	             The length.
/// @param nWidthStep	         The nWidthStep.
///
///-------------------------------------------------------------------------------------------------
typedef struct LiquefyWarp
{
    float fOldPointX;
	float fOldPointY;
	float fNewPointX;
	float fNewPointY;
	float radius;
    int nWidth;
    int nHeight;
    int nWidthStep;
}warp;
//-------------------------------------------------------------------------------------------------
///
/// @brief	映射的函数
///
/// @author	Meitu
/// @date	2020/10/12
///
/// @param [in]x                     The x of old point.
/// @param [in]y	                 The y of old point.
/// @param [out]u                    The mapping of x.
/// @param [out]v	                 The mapping of y.
/// @param w                         The sturct of w
/// @return	NULL
///
///-------------------------------------------------------------------------------------------------
void mapping(warp *w,int x,int y, double *u,double *v);
//-------------------------------------------------------------------------------------------------
///
/// @brief	扫描图像的函数
///
/// @author	Meitu
/// @date	2020/10/12
///
/// @param w                     The sturct of w
/// @param [in]  pOldColorData   If non-null, destination landmark.
/// @param [out] pNewColorData   If non-null, destination landmark.
///
/// @return	NULL
///-------------------------------------------------------------------------------------------------
void scanning(warp *w,unsigned char *pOldColorData,unsigned char *pNewColorData);
//#endif

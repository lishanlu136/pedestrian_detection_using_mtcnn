#pragma once

#include <opencv2/core/version.hpp>
//����꣬��֤��debugģʽ�£�����opencv_xxxd.lib��releaseģʽ�µ���opencv_xxx.lib
#ifdef _DEBUG
#   define CC_CVLIB(name) "opencv_" name CC_CVVERSION_ID "d.lib"
#   define CC_CVLIB_2(name) "opencv_" name CC_CVVERSION_ID2 "d.lib"
#   define CC_LIB(name) name "d.lib"
#else
#   define CC_CVLIB(name) "opencv_" name CC_CVVERSION_ID ".lib"
#   define CC_CVLIB_2(name) "opencv_" name CC_CVVERSION_ID2 ".lib"
#   define CC_LIB(name)   name ".lib"
#endif

#ifndef OPENCV_DLLIB
//���ھ�̬�⣬���뵼��������Щ��
#pragma comment(lib, "kernel32.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "Vfw32.lib")
#pragma comment(lib, "winspool.lib")
#pragma comment(lib, "comdlg32.lib")
#pragma comment(lib, "advapi32.lib")
#pragma comment(lib, "shell32.lib")
#pragma comment(lib, "ole32.lib")
#pragma comment(lib, "oleaut32.lib")
#pragma comment(lib, "uuid.lib")
#pragma comment(lib, "odbc32.lib")
#pragma comment(lib, "odbccp32.lib")
#pragma comment(lib, "Comctl32.lib")
#endif

//���Ϊ3�棬���뷽ʽ��ͬ
#if CV_MAJOR_VERSION==3
//����cv�Ŀ����ƣ�2410��ģ�����������汾�����޸�Ϊָ���ľͺ���
#define CC_CVVERSION_ID       "2410"
#define CC_CVVERSION_ID2       "300"

#ifndef OPENCV_DLLIB
//���뾲̬������
#pragma comment(lib, CC_LIB("IlmImf"))
#pragma comment(lib, CC_LIB("libjasper"))
#pragma comment(lib, CC_LIB("libjpeg"))
#pragma comment(lib, CC_LIB("libpng"))
#pragma comment(lib, CC_LIB("libtiff"))
#pragma comment(lib, CC_LIB("libwebp"))
#pragma comment(lib, "ippicvmt.lib")
#endif

#pragma comment( lib, CC_CVLIB("calib3d") )
#pragma comment( lib, CC_CVLIB_2("calib3d") )
#pragma comment( lib, CC_CVLIB("contrib") )
//#pragma comment( lib, CC_CVLIB("core") )
#pragma comment( lib, CC_CVLIB_2("core") )
//#pragma comment( lib, CC_CVLIB("core") )
#pragma comment( lib, CC_CVLIB_2("features2d") )
#pragma comment( lib, CC_CVLIB("features2d") )
#pragma comment( lib, CC_CVLIB_2("flann") )
#pragma comment( lib, CC_CVLIB("flann") )
#pragma comment( lib, CC_CVLIB("gpu") )
#pragma comment( lib, CC_CVLIB_2("highgui") )
#pragma comment( lib, CC_CVLIB("highgui") )
#pragma comment( lib, CC_CVLIB_2("imgcodecs") )
#pragma comment( lib, CC_CVLIB_2("imgproc") )
#pragma comment( lib, CC_CVLIB("imgproc") )
#pragma comment( lib, CC_CVLIB("legacy") )
#pragma comment( lib, CC_CVLIB_2("ml") )
#pragma comment( lib, CC_CVLIB("ml") )
#pragma comment( lib, CC_CVLIB("ocl") )
#pragma comment( lib, CC_CVLIB("nonfree") )
#pragma comment( lib, CC_CVLIB_2("objdetect") )
#pragma comment( lib, CC_CVLIB("objdetect") )
#pragma comment( lib, CC_CVLIB_2("photo") )
#pragma comment( lib, CC_CVLIB("photo") )
#pragma comment( lib, CC_CVLIB_2("shape") )
#pragma comment( lib, CC_CVLIB_2("stitching") )
#pragma comment( lib, CC_CVLIB("stitching") )
#pragma comment( lib, CC_CVLIB_2("superres") )
#pragma comment( lib, CC_CVLIB("superres") )
#pragma comment( lib, CC_CVLIB_2("ts") )
#pragma comment( lib, CC_CVLIB("ts") )
#pragma comment( lib, CC_CVLIB_2("video") )
#pragma comment( lib, CC_CVLIB("video") )
#pragma comment( lib, CC_CVLIB_2("videoio") )
#pragma comment( lib, CC_CVLIB_2("videostab") )
#pragma comment( lib, CC_CVLIB("videostab") )

#ifndef OPENCV_DLLIB
#pragma comment(lib, CC_LIB("zlib"))
#endif
#else
#define CC_CVVERSION_ID CVAUX_STR(CV_VERSION_EPOCH) CVAUX_STR(CV_VERSION_MAJOR) CVAUX_STR(CV_VERSION_MINOR)

#ifndef OPENCV_DLLIB
//���뾲̬������
#pragma comment(lib, CC_LIB("IlmImf"))
#pragma comment(lib, CC_LIB("libjasper"))
#pragma comment(lib, CC_LIB("libjpeg"))
#pragma comment(lib, CC_LIB("libpng"))
#pragma comment(lib, CC_LIB("libtiff"))
#endif

#pragma comment( lib, CC_CVLIB("calib3d") )
#pragma comment( lib, CC_CVLIB("contrib") )
#pragma comment( lib, CC_CVLIB("core") )
#pragma comment( lib, CC_CVLIB("features2d") )
#pragma comment( lib, CC_CVLIB("flann") )
#pragma comment( lib, CC_CVLIB("gpu") )
#pragma comment( lib, CC_CVLIB("highgui") )
#pragma comment( lib, CC_CVLIB("imgproc") )
#pragma comment( lib, CC_CVLIB("legacy") )
#pragma comment( lib, CC_CVLIB("ml") )
#pragma comment( lib, CC_CVLIB("ocl") )
#pragma comment( lib, CC_CVLIB("nonfree") )
#pragma comment( lib, CC_CVLIB("objdetect") )
#pragma comment( lib, CC_CVLIB("photo") )
#pragma comment( lib, CC_CVLIB("stitching") )
#pragma comment( lib, CC_CVLIB("superres") )
#pragma comment( lib, CC_CVLIB("ts") )
#pragma comment( lib, CC_CVLIB("video") )
#pragma comment( lib, CC_CVLIB("videostab") )

#ifndef OPENCV_DLLIB
#pragma comment(lib, CC_LIB("zlib"))
#endif
#endif //CV_MAJOR_VERSION

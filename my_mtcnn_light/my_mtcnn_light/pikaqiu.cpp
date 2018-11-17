#include "network.h"
#include "mtcnn.h"
#include <time.h>
#include <io.h>
#include <fstream>
#include <string>

//#pragma comment(lib, "libopenblas.dll.a")

void getAllFiles(string path, vector<string> &files)
{
	long hFile = 0;
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("\\*").c_str(), &fileinfo)) != -1)
	{
		do
		{
			if (fileinfo.attrib & _A_SUBDIR)
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					files.push_back(p.assign(path).append("\\").append(fileinfo.name));
					getAllFiles(p.assign(path).append("\\").append(fileinfo.name), files);
				}
			}
			else
			{
				files.push_back(p.assign(path).append("\\").append(fileinfo.name));
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
}

int main()
{
	char *picPath = "D:\\MTCNN\\mtcnn-master-caffe\\train\\Train\\pos";
	vector<string> files;
	getAllFiles(picPath, files);
	int size = files.size();
	cout << "test pic num is " << size << endl;
	for (int i = 0; i < size; ++i)
	{
		Mat im = imread(files[i]);
		cout << "this is " << i << "th pic" << endl;
		mtcnn find(im.rows, im.cols);
		vector<Rect> objs = find.detectObject(im);
		for (int j = 0; j < objs.size(); ++j)
			rectangle(im, objs[j], Scalar(0, 0, 255), 2);

		imshow("demo", im);
		waitKey(10);
	}
    return 0;
}
#include "mtcnn.h"

#ifdef modelFromHFile
#include "mtcnn_models.h"
#endif

Pnet::Pnet(){
	Pthreshold = 0.6;
    nms_threshold = 0.5;
    firstFlag = true;
    this->rgb = new pBox;

    this->conv1_matrix = new pBox;
    this->conv1 = new pBox;

    this->conv2_matrix = new pBox;
    this->conv2 = new pBox;

    this->conv3_matrix = new pBox;
    this->conv3 = new pBox;

	this->conv4_matrix = new pBox;
	this->conv4 = new pBox;

    this->score_matrix = new pBox;
    this->score_ = new pBox;

    this->location_matrix = new pBox;
    this->location_ = new pBox;

    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;
	this->conv4_wb = new Weight;
	this->prelu_gmma4 = new pRelu;
    this->conv5c1_wb = new Weight;
    this->conv5c2_wb = new Weight;
    //                                 w       sc lc k_w k_h s_w s_h p_w p_h
    long conv1 = initConvAndFc(this->conv1_wb, 10, 3, 3,  5,  1,  1,  0,  0);
    initpRelu(this->prelu_gmma1, 10);
    long conv2 = initConvAndFc(this->conv2_wb, 16, 10, 3, 5, 2, 2, 1, 0);
    initpRelu(this->prelu_gmma2, 16);
    long conv3 = initConvAndFc(this->conv3_wb, 32, 16, 3, 5, 2, 2, 1, 0);
    initpRelu(this->prelu_gmma3, 32);
	long conv4 = initConvAndFc(this->conv4_wb, 32, 32, 3, 5, 1, 1, 0, 0);
	initpRelu(this->prelu_gmma4, 32);
    long conv5c1 = initConvAndFc(this->conv5c1_wb, 2, 32, 1, 1, 1, 1, 0, 0);
    long conv5c2 = initConvAndFc(this->conv5c2_wb, 4, 32, 1, 1, 1, 1, 0, 0);
	long dataNumber[16] = { conv1, 10, 10, conv2, 16, 16, conv3, 32, 32, conv4, 32, 32, conv5c1, 2, conv5c2, 4 };
    mydataFmt *pointTeam[16] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
							    this->conv4_wb->pdata, this->conv4_wb->pbias, this->prelu_gmma4->pdata, \
                                this->conv5c1_wb->pdata, this->conv5c1_wb->pbias, \
                                this->conv5c2_wb->pdata, this->conv5c2_wb->pbias \
                               };
    //string filename = "Pnet.txt";
    //readData(filename, dataNumber, pointTeam);

#ifdef modelFromHFile
	readData(model_weights_PNet_, sizeof(model_weights_PNet_) / sizeof(model_weights_PNet_[0]), 
		dataNumber, pointTeam, sizeof(dataNumber) / sizeof(dataNumber[0]));
#else
	readData("Pnet.txt", dataNumber, pointTeam);
#endif
}

Pnet::~Pnet(){
    freepBox(this->rgb);
    freepBox(this->conv1);
    freepBox(this->conv2);
    freepBox(this->conv3);
	freepBox(this->conv4);
    freepBox(this->score_);
    freepBox(this->location_);

    freepBox(this->conv1_matrix);
    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);
    freepBox(this->conv2_matrix);
    freeWeight(this->conv2_wb);
	freepRelu(this->prelu_gmma2);
    freepBox(this->conv3_matrix);
    freeWeight(this->conv3_wb);
	freepRelu(this->prelu_gmma3);
	freepBox(this->conv4_matrix);
	freeWeight(this->conv4_wb);
	freepRelu(this->prelu_gmma4);
    freepBox(this->score_matrix);
    freeWeight(this->conv5c1_wb);
    freepBox(this->location_matrix);
    freeWeight(this->conv5c2_wb);
}

void Pnet::run(Mat &image, float scale){
    if(firstFlag){
        image2MatrixInit(image, this->rgb);

        feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
        convolutionInit(this->conv1_wb, this->rgb, this->conv1, this->conv1_matrix);

        feature2MatrixInit(this->conv1, this->conv2_matrix, this->conv2_wb);
        convolutionInit(this->conv2_wb, this->conv1, this->conv2, this->conv2_matrix);
        
        feature2MatrixInit(this->conv2, this->conv3_matrix, this->conv3_wb);
        convolutionInit(this->conv3_wb, this->conv2, this->conv3, this->conv3_matrix);

		feature2MatrixInit(this->conv3, this->conv4_matrix, this->conv4_wb);
		convolutionInit(this->conv4_wb, this->conv3, this->conv4, this->conv4_matrix);

        feature2MatrixInit(this->conv4, this->score_matrix, this->conv5c1_wb);
        convolutionInit(this->conv5c1_wb, this->conv4, this->score_, this->score_matrix);

        feature2MatrixInit(this->conv4, this->location_matrix, this->conv5c2_wb);
        convolutionInit(this->conv5c2_wb, this->conv4, this->location_, this->location_matrix);
        firstFlag = false;
    }

    image2Matrix(image, this->rgb);
	//conv1
    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1, this->conv1_matrix);
    prelu(this->conv1, this->conv1_wb->pbias, this->prelu_gmma1->pdata);
    //conv2
    feature2Matrix(this->conv1, this->conv2_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->conv1, this->conv2, this->conv2_matrix);
    prelu(this->conv2, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
    //conv3 
    feature2Matrix(this->conv2, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->conv2, this->conv3, this->conv3_matrix);
    prelu(this->conv3, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
	//conv4
	feature2Matrix(this->conv3, this->conv4_matrix, this->conv4_wb);
	convolution(this->conv4_wb, this->conv3, this->conv4, this->conv4_matrix);
	prelu(this->conv4, this->conv4_wb->pbias, this->prelu_gmma4->pdata);
    //conv5c1   score
    feature2Matrix(this->conv4, this->score_matrix, this->conv5c1_wb);
    convolution(this->conv5c1_wb, this->conv4, this->score_, this->score_matrix);
    addbias(this->score_, this->conv5c1_wb->pbias);
    softmax(this->score_);
    // pBoxShow(this->score_);

    //conv5c2   location
    feature2Matrix(this->conv4, this->location_matrix, this->conv5c2_wb);
    convolution(this->conv5c2_wb, this->conv4, this->location_, this->location_matrix);
    addbias(this->location_, this->conv5c2_wb->pbias);
    //softmax layer
    generateBbox(this->score_, this->location_, scale);
}
void Pnet::generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale){
    //for pooling 
    int stride = 2;
	int pading = 1;
    int cellsize = 12;
    int count = 0;
    //score p
    mydataFmt *p = score->pdata + score->width*score->height;
    mydataFmt *plocal = location->pdata;

	Mat cls_map(score->height, score->width, CV_32F, p);
	Mat loc(location->height, location->width, CV_32F, plocal);

    struct Bbox bbox;
    struct orderScore order;
	vector<pair<Point, float>> pts;
    for(int row=0;row<score->height;row++){
        for(int col=0;col<score->width;col++){
            if(*p>Pthreshold){
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
				if (col == 0)
				{
					bbox.x1 = 0.0;
					bbox.x2 = (stride*cellsize - pading) / scale;
				}
				else
				{
					bbox.x1 = ((stride*(stride*col - pading) - pading) / scale);
					//bbox.x2 = (((stride*(stride*col - pading + cellsize) - pading)) / scale);
					bbox.x2 = (((stride*(stride*col - pading) - pading)+cellsize) / scale);
				}
                bbox.y1 = ((stride*(stride*row))/scale);
                //bbox.y2 = ((stride*(stride*row+3*cellsize))/scale);
				bbox.y2 = ((stride*(stride*row) + 3 * cellsize) / scale);
                bbox.exist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for(int channel=0;channel<4;channel++)
                    bbox.regreCoord[channel]=*(plocal+channel*location->width*location->height);
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;

				pts.push_back(make_pair(Point(col, row), *p));
            }
            p++;
            plocal++;
        }
    }

#if 0
	std::sort(pts.begin(), pts.end(), [](const pair<Point, float>& a, const pair<Point, float>& b){
		return a.second > b.second;
	});
#endif
}

//Pnet::Pnet(){
//	Pthreshold = 0.6;
//	nms_threshold = 0.5;
//	firstFlag = true;
//	this->rgb = new pBox;
//
//	this->conv1_matrix = new pBox;
//	this->conv1 = new pBox;
//	this->maxPooling1 = new pBox;
//
//	this->maxPooling_matrix = new pBox;
//	this->conv2 = new pBox;
//
//	this->conv3_matrix = new pBox;
//	this->conv3 = new pBox;
//
//	this->score_matrix = new pBox;
//	this->score_ = new pBox;
//
//	this->location_matrix = new pBox;
//	this->location_ = new pBox;
//
//	this->conv1_wb = new Weight;
//	this->prelu_gmma1 = new pRelu;
//	this->conv2_wb = new Weight;
//	this->prelu_gmma2 = new pRelu;
//	this->conv3_wb = new Weight;
//	this->prelu_gmma3 = new pRelu;
//	this->conv4c1_wb = new Weight;
//	this->conv4c2_wb = new Weight;
//	//                                 w       sc  lc ks    s     p
//	long conv1 = initConvAndFc(this->conv1_wb, 10, 3, 3, 3, 1, 1, 0, 0);
//	initpRelu(this->prelu_gmma1, 10);
//	long conv2 = initConvAndFc(this->conv2_wb, 16, 10, 3, 3, 1, 1, 0, 0);
//	initpRelu(this->prelu_gmma2, 16);
//	long conv3 = initConvAndFc(this->conv3_wb, 32, 16, 3, 3, 1, 1, 0, 0);
//	initpRelu(this->prelu_gmma3, 32);
//	long conv4c1 = initConvAndFc(this->conv4c1_wb, 2, 32, 1, 1, 1, 1, 0, 0);
//	long conv4c2 = initConvAndFc(this->conv4c2_wb, 4, 32, 1, 1, 1, 1, 0, 0);
//	long dataNumber[13] = { conv1, 10, 10, conv2, 16, 16, conv3, 32, 32, conv4c1, 2, conv4c2, 4 };
//	mydataFmt *pointTeam[13] = { this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
//		this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
//		this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
//		this->conv4c1_wb->pdata, this->conv4c1_wb->pbias, \
//		this->conv4c2_wb->pdata, this->conv4c2_wb->pbias \
//	};
//	//string filename = "Pnet.txt";
//	//readData(filename, dataNumber, pointTeam);
//
//#ifdef modelFromHFile
//	readData(model_weights_PNet_, sizeof(model_weights_PNet_) / sizeof(model_weights_PNet_[0]),
//		dataNumber, pointTeam, sizeof(dataNumber) / sizeof(dataNumber[0]));
//#else
//	readData("Pnet.txt", dataNumber, pointTeam);
//#endif
//}
//
//Pnet::~Pnet(){
//	freepBox(this->rgb);
//	freepBox(this->conv1);
//	freepBox(this->maxPooling1);
//	freepBox(this->conv2);
//	freepBox(this->conv3);
//	freepBox(this->score_);
//	freepBox(this->location_);
//
//	freepBox(this->conv1_matrix);
//	freeWeight(this->conv1_wb);
//	freepRelu(this->prelu_gmma1);
//	freepBox(this->maxPooling_matrix);
//	freeWeight(this->conv2_wb);
//	freepBox(this->conv3_matrix);
//	freepRelu(this->prelu_gmma2);
//	freeWeight(this->conv3_wb);
//	freepBox(this->score_matrix);
//	freepRelu(this->prelu_gmma3);
//	freeWeight(this->conv4c1_wb);
//	freepBox(this->location_matrix);
//	freeWeight(this->conv4c2_wb);
//}
//
//void Pnet::run(Mat &image, float scale){
//	if (firstFlag){
//		image2MatrixInit(image, this->rgb);
//
//		feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
//		convolutionInit(this->conv1_wb, this->rgb, this->conv1, this->conv1_matrix);
//
//		maxPoolingInit(this->conv1, this->maxPooling1, 2, 2);
//		feature2MatrixInit(this->maxPooling1, this->maxPooling_matrix, this->conv2_wb);
//		convolutionInit(this->conv2_wb, this->maxPooling1, this->conv2, this->maxPooling_matrix);
//
//		feature2MatrixInit(this->conv2, this->conv3_matrix, this->conv3_wb);
//		convolutionInit(this->conv3_wb, this->conv2, this->conv3, this->conv3_matrix);
//
//		feature2MatrixInit(this->conv3, this->score_matrix, this->conv4c1_wb);
//		convolutionInit(this->conv4c1_wb, this->conv3, this->score_, this->score_matrix);
//
//		feature2MatrixInit(this->conv3, this->location_matrix, this->conv4c2_wb);
//		convolutionInit(this->conv4c2_wb, this->conv3, this->location_, this->location_matrix);
//		firstFlag = false;
//	}
//
//	image2Matrix(image, this->rgb);
//
//	feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
//	convolution(this->conv1_wb, this->rgb, this->conv1, this->conv1_matrix);
//	prelu(this->conv1, this->conv1_wb->pbias, this->prelu_gmma1->pdata);
//	//Pooling layer
//	maxPooling(this->conv1, this->maxPooling1, 2, 2);
//
//	feature2Matrix(this->maxPooling1, this->maxPooling_matrix, this->conv2_wb);
//	convolution(this->conv2_wb, this->maxPooling1, this->conv2, this->maxPooling_matrix);
//	prelu(this->conv2, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
//	//conv3 
//	feature2Matrix(this->conv2, this->conv3_matrix, this->conv3_wb);
//	convolution(this->conv3_wb, this->conv2, this->conv3, this->conv3_matrix);
//	prelu(this->conv3, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
//	//conv4c1   score
//	feature2Matrix(this->conv3, this->score_matrix, this->conv4c1_wb);
//	convolution(this->conv4c1_wb, this->conv3, this->score_, this->score_matrix);
//	addbias(this->score_, this->conv4c1_wb->pbias);
//	softmax(this->score_);
//	// pBoxShow(this->score_);
//
//	//conv4c2   location
//	feature2Matrix(this->conv3, this->location_matrix, this->conv4c2_wb);
//	convolution(this->conv4c2_wb, this->conv3, this->location_, this->location_matrix);
//	addbias(this->location_, this->conv4c2_wb->pbias);
//	//softmax layer
//	generateBbox(this->score_, this->location_, scale);
//}
//void Pnet::generateBbox(const struct pBox *score, const struct pBox *location, mydataFmt scale){
//	//for pooling 
//	int stride = 2;
//	int cellsize = 12;
//	int count = 0;
//	//score p
//	mydataFmt *p = score->pdata + score->width*score->height;
//	mydataFmt *plocal = location->pdata;
//
//	Mat cls_map(score->height, score->width, CV_32F, p);
//	Mat loc(location->height, location->width, CV_32F, plocal);
//
//	struct Bbox bbox;
//	struct orderScore order;
//	vector<pair<Point, float>> pts;
//	for (int row = 0; row<score->height; row++){
//		for (int col = 0; col<score->width; col++){
//			if (*p>Pthreshold){
//				bbox.score = *p;
//				order.score = *p;
//				order.oriOrder = count;
//				bbox.x1 = ((stride*col) / scale);
//				bbox.y1 = ((stride*row) / scale);
//				bbox.x2 = ((stride*col + cellsize) / scale);
//				bbox.y2 = ((stride*row + cellsize) / scale);
//				bbox.exist = true;
//				bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
//				for (int channel = 0; channel<4; channel++)
//					bbox.regreCoord[channel] = *(plocal + channel*location->width*location->height);
//				boundingBox_.push_back(bbox);
//				bboxScore_.push_back(order);
//				count++;
//
//				pts.push_back(make_pair(Point(col, row), *p));
//			}
//			p++;
//			plocal++;
//		}
//	}
//
//#if 0
//	std::sort(pts.begin(), pts.end(), [](const pair<Point, float>& a, const pair<Point, float>& b){
//		return a.second > b.second;
//	});
//#endif
//}


Rnet::Rnet(){
	Rthreshold = 0.6;

    this->rgb = new pBox;
    this->conv1_matrix = new pBox;
    this->conv1_out = new pBox;

    this->conv2_matrix = new pBox;
    this->conv2_out = new pBox;

    this->conv3_matrix = new pBox;
    this->conv3_out = new pBox;

	this->conv4_matrix = new pBox;
	this->conv4_out = new pBox;

	this->conv5_out = new pBox;

    this->score_ = new pBox;
    this->location_ = new pBox;

    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;
	this->conv4_wb = new Weight;
	this->prelu_gmma4 = new pRelu;
	this->conv5_wb = new Weight;
	this->prelu_gmma5 = new pRelu;
    this->score_wb = new Weight;
    this->location_wb = new Weight;
    // //                             w        sc  lc k_w k_h s_w s_h p_w p_h
    long conv1 = initConvAndFc(this->conv1_wb, 16, 3,  3,  5,  1,  1,  0,  0);
    initpRelu(this->prelu_gmma1, 16);
    long conv2 = initConvAndFc(this->conv2_wb, 32, 16, 3, 5, 2, 2, 1, 0);
    initpRelu(this->prelu_gmma2, 32);
    long conv3 = initConvAndFc(this->conv3_wb, 64, 32, 3, 5, 2, 2, 1, 0);
    initpRelu(this->prelu_gmma3, 64);
	long conv4 = initConvAndFc(this->conv4_wb, 64, 64, 3, 5, 2, 2, 1, 0);
	initpRelu(this->prelu_gmma4, 64);
	long conv5 = initConvAndFc(this->conv5_wb, 128, 960, 1, 1, 1, 1, 0, 0);
	initpRelu(this->prelu_gmma5, 128);
    long score = initConvAndFc(this->score_wb, 2, 128, 1, 1, 1, 1, 0, 0);
    long location = initConvAndFc(this->location_wb, 4, 128, 1, 1, 1, 1, 0, 0);
	long dataNumber[19] = { conv1, 16, 16, conv2, 32, 32, conv3, 64, 64, conv4, 64, 64, conv5, 128, 128, score, 2, location, 4 };
    mydataFmt *pointTeam[19] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
								this->conv4_wb->pdata, this->conv4_wb->pbias, this->prelu_gmma4->pdata, \
								this->conv5_wb->pdata, this->conv5_wb->pbias, this->prelu_gmma5->pdata, \
                                this->score_wb->pdata, this->score_wb->pbias, \
                                this->location_wb->pdata, this->location_wb->pbias \
                                };
    //string filename = "Rnet.txt";
    //readData(filename, dataNumber, pointTeam);

#ifdef modelFromHFile
	readData(model_weights_RNet_, sizeof(model_weights_RNet_) / sizeof(model_weights_RNet_[0]),
		dataNumber, pointTeam, sizeof(dataNumber) / sizeof(dataNumber[0]));
#else
	readData("Rnet.txt", dataNumber, pointTeam);
#endif

    //Init the network
    RnetImage2MatrixInit(rgb);
    feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolutionInit(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    feature2MatrixInit(this->conv1_out, this->conv2_matrix, this->conv2_wb);
    convolutionInit(this->conv2_wb, this->conv1_out, this->conv2_out, this->conv2_matrix);
    feature2MatrixInit(this->conv2_out, this->conv3_matrix, this->conv3_wb);
    convolutionInit(this->conv3_wb, this->conv2_out, this->conv3_out, this->conv3_matrix);
	feature2MatrixInit(this->conv3_out, this->conv4_matrix, this->conv4_wb);
	convolutionInit(this->conv4_wb, this->conv3_out, this->conv4_out, this->conv4_matrix);
	fullconnectInit(this->conv5_wb, this->conv5_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
}
Rnet::~Rnet(){
    freepBox(this->rgb);
    freepBox(this->conv1_matrix);
    freepBox(this->conv1_out);
    freepBox(this->conv2_matrix);
    freepBox(this->conv2_out);
    freepBox(this->conv3_matrix);
    freepBox(this->conv3_out);
	freepBox(this->conv4_matrix);
	freepBox(this->conv4_out);
	freepBox(this->conv5_out);
    freepBox(this->score_);
    freepBox(this->location_);

    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);
    freeWeight(this->conv2_wb);
    freepRelu(this->prelu_gmma2);
    freeWeight(this->conv3_wb);
    freepRelu(this->prelu_gmma3);
	freeWeight(this->conv4_wb);
	freepRelu(this->prelu_gmma4);
	freeWeight(this->conv5_wb);
	freepRelu(this->prelu_gmma5);
    freeWeight(this->score_wb);
    freeWeight(this->location_wb);
}
void Rnet::RnetImage2MatrixInit(struct pBox *pbox){
    pbox->channel = 3;
    pbox->height = 72;
    pbox->width = 24;
    
    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL)cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}
void Rnet::run(Mat &image){
    image2Matrix(image, this->rgb);
	//conv1
    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    prelu(this->conv1_out, this->conv1_wb->pbias, this->prelu_gmma1->pdata);
	//conv2
    feature2Matrix(this->conv1_out, this->conv2_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->conv1_out, this->conv2_out, this->conv2_matrix);
    prelu(this->conv2_out, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
    //conv3 
    feature2Matrix(this->conv2_out, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->conv2_out, this->conv3_out, this->conv3_matrix);
    prelu(this->conv3_out, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
	//conv4 
	feature2Matrix(this->conv3_out, this->conv4_matrix, this->conv4_wb);
	convolution(this->conv4_wb, this->conv3_out, this->conv4_out, this->conv4_matrix);
	prelu(this->conv4_out, this->conv4_wb->pbias, this->prelu_gmma4->pdata);
	//conv5 / flatten
    fullconnect(this->conv5_wb, this->conv4_out, this->conv5_out);
    prelu(this->conv5_out, this->conv5_wb->pbias, this->prelu_gmma5->pdata);

    //conv6_1   score
	fullconnect(this->score_wb, this->conv5_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);

    //conv6_2   location
	fullconnect(this->location_wb, this->conv5_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
    // pBoxShow(location_);
}

Onet::Onet(){
    Othreshold = 0.8;
    this->rgb = new pBox;

    this->conv1_matrix = new pBox;
    this->conv1_out = new pBox;

    this->conv2_matrix = new pBox;
    this->conv2_out = new pBox;

    this->conv3_matrix = new pBox;
    this->conv3_out = new pBox;

    this->conv4_matrix = new pBox;
    this->conv4_out = new pBox;

	this->conv5_matrix = new pBox;
	this->conv5_out = new pBox;

    this->fc6_out = new pBox;

    this->score_ = new pBox;
    this->location_ = new pBox;

    this->conv1_wb = new Weight;
    this->prelu_gmma1 = new pRelu;
    this->conv2_wb = new Weight;
    this->prelu_gmma2 = new pRelu;
    this->conv3_wb = new Weight;
    this->prelu_gmma3 = new pRelu;
    this->conv4_wb = new Weight;
    this->prelu_gmma4 = new pRelu;
	this->conv5_wb = new Weight;
	this->prelu_gmma5 = new pRelu;
    this->fc6_wb = new Weight;
    this->prelu_gmma6 = new pRelu;
    this->score_wb = new Weight;
    this->location_wb = new Weight;

    // //                             w        sc lc k_w k_h s_w s_h p_w p_h
    long conv1 = initConvAndFc(this->conv1_wb, 24, 3, 3,  5,  1,  1,  0,  0);
    initpRelu(this->prelu_gmma1, 24);
    long conv2 = initConvAndFc(this->conv2_wb, 48, 24, 3, 5, 2, 2, 1, 0);
    initpRelu(this->prelu_gmma2, 48);
    long conv3 = initConvAndFc(this->conv3_wb, 64, 48, 3, 5, 2, 2, 1, 0);
    initpRelu(this->prelu_gmma3, 64);
    long conv4 = initConvAndFc(this->conv4_wb, 128, 64, 3, 5, 2, 2, 1, 0);
    initpRelu(this->prelu_gmma4, 128);
	long conv5 = initConvAndFc(this->conv5_wb, 128, 128, 3, 5, 2, 2, 1, 0);
	initpRelu(this->prelu_gmma5, 128);
    long fc6 = initConvAndFc(this->fc6_wb, 256, 1920, 1, 1, 1, 1, 0, 0);
    initpRelu(this->prelu_gmma6, 256);
    long score = initConvAndFc(this->score_wb, 2, 256, 1, 1, 1, 1, 0, 0);
    long location = initConvAndFc(this->location_wb, 4, 256, 1, 1, 1, 1, 0, 0);
	long dataNumber[22] = { conv1, 24, 24, conv2, 48, 48, conv3, 64, 64, conv4, 128, 128, conv5, 128, 128, fc6, 256, 256, score, 2, location, 4 };
    mydataFmt *pointTeam[22] = {this->conv1_wb->pdata, this->conv1_wb->pbias, this->prelu_gmma1->pdata, \
                                this->conv2_wb->pdata, this->conv2_wb->pbias, this->prelu_gmma2->pdata, \
                                this->conv3_wb->pdata, this->conv3_wb->pbias, this->prelu_gmma3->pdata, \
                                this->conv4_wb->pdata, this->conv4_wb->pbias, this->prelu_gmma4->pdata, \
								this->conv5_wb->pdata, this->conv5_wb->pbias, this->prelu_gmma5->pdata, \
                                this->fc6_wb->pdata, this->fc6_wb->pbias, this->prelu_gmma6->pdata, \
                                this->score_wb->pdata, this->score_wb->pbias, \
                                this->location_wb->pdata, this->location_wb->pbias, \
                                };
    //string filename = "Onet.txt";
    //readData(filename, dataNumber, pointTeam);
#ifdef modelFromHFile
	readData(model_weights_ONet_, sizeof(model_weights_ONet_) / sizeof(model_weights_ONet_[0]),
		dataNumber, pointTeam, sizeof(dataNumber) / sizeof(dataNumber[0]));
#else
	readData("Onet.txt", dataNumber, pointTeam);
#endif

    //Init the network
    OnetImage2MatrixInit(rgb);

    feature2MatrixInit(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolutionInit(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);

    feature2MatrixInit(this->conv1_out, this->conv2_matrix, this->conv2_wb);
    convolutionInit(this->conv2_wb, this->conv1_out, this->conv2_out, this->conv2_matrix);

    feature2MatrixInit(this->conv2_out, this->conv3_matrix, this->conv3_wb);
    convolutionInit(this->conv3_wb, this->conv2_out, this->conv3_out, this->conv3_matrix);

    feature2MatrixInit(this->conv3_out, this->conv4_matrix, this->conv4_wb);
    convolutionInit(this->conv4_wb, this->conv3_out, this->conv4_out, this->conv4_matrix);

	feature2MatrixInit(this->conv4_out, this->conv5_matrix, this->conv5_wb);
	convolutionInit(this->conv5_wb, this->conv4_out, this->conv5_out, this->conv5_matrix);

    fullconnectInit(this->fc6_wb, this->fc6_out);
    fullconnectInit(this->score_wb, this->score_);
    fullconnectInit(this->location_wb, this->location_);
}
Onet::~Onet(){
    freepBox(this->rgb);
    freepBox(this->conv1_matrix);
    freepBox(this->conv1_out);
    freepBox(this->conv2_matrix);
    freepBox(this->conv2_out);
    freepBox(this->conv3_matrix);
    freepBox(this->conv3_out);
    freepBox(this->conv4_matrix);
    freepBox(this->conv4_out);
	freepBox(this->conv5_matrix);
	freepBox(this->conv5_out);
    freepBox(this->fc6_out);
    freepBox(this->score_);
    freepBox(this->location_);

    freeWeight(this->conv1_wb);
    freepRelu(this->prelu_gmma1);
    freeWeight(this->conv2_wb);
    freepRelu(this->prelu_gmma2);
    freeWeight(this->conv3_wb);
    freepRelu(this->prelu_gmma3);
    freeWeight(this->conv4_wb);
    freepRelu(this->prelu_gmma4);
	freeWeight(this->conv5_wb);
	freepRelu(this->prelu_gmma5);
    freeWeight(this->fc6_wb);
    freepRelu(this->prelu_gmma6);
    freeWeight(this->score_wb);
    freeWeight(this->location_wb);
}

void Onet::OnetImage2MatrixInit(struct pBox *pbox){
    pbox->channel = 3;
    pbox->height = 144;
    pbox->width = 48;
    
    pbox->pdata = (mydataFmt *)malloc(pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
    if(pbox->pdata==NULL)cout<<"the image2MatrixInit is failed!!"<<endl;
    memset(pbox->pdata, 0, pbox->channel*pbox->height*pbox->width*sizeof(mydataFmt));
}

void Onet::run(Mat &image){
    image2Matrix(image, this->rgb);
	//conv1
    feature2Matrix(this->rgb, this->conv1_matrix, this->conv1_wb);
    convolution(this->conv1_wb, this->rgb, this->conv1_out, this->conv1_matrix);
    prelu(this->conv1_out, this->conv1_wb->pbias, this->prelu_gmma1->pdata);
	//conv2
    feature2Matrix(this->conv1_out, this->conv2_matrix, this->conv2_wb);
    convolution(this->conv2_wb, this->conv1_out, this->conv2_out, this->conv2_matrix);
    prelu(this->conv2_out, this->conv2_wb->pbias, this->prelu_gmma2->pdata);
    //conv3 
    feature2Matrix(this->conv2_out, this->conv3_matrix, this->conv3_wb);
    convolution(this->conv3_wb, this->conv2_out, this->conv3_out, this->conv3_matrix);
    prelu(this->conv3_out, this->conv3_wb->pbias, this->prelu_gmma3->pdata);
    //conv4
    feature2Matrix(this->conv3_out, this->conv4_matrix, this->conv4_wb);
    convolution(this->conv4_wb, this->conv3_out, this->conv4_out, this->conv4_matrix);
    prelu(this->conv4_out, this->conv4_wb->pbias, this->prelu_gmma4->pdata);
	//conv5
	feature2Matrix(this->conv4_out, this->conv5_matrix, this->conv5_wb);
	convolution(this->conv5_wb, this->conv4_out, this->conv5_out, this->conv5_matrix);
	prelu(this->conv5_out, this->conv5_wb->pbias, this->prelu_gmma5->pdata);
	//fc6(conv6)
    fullconnect(this->fc6_wb, this->conv5_out, this->fc6_out);
    prelu(this->fc6_out, this->fc6_wb->pbias, this->prelu_gmma6->pdata);

    //conv7_1   score
    fullconnect(this->score_wb, this->fc6_out, this->score_);
    addbias(this->score_, this->score_wb->pbias);
    softmax(this->score_);
    // pBoxShow(this->score_);

    //conv7_2   location
    fullconnect(this->location_wb, this->fc6_out, this->location_);
    addbias(this->location_, this->location_wb->pbias);
    // pBoxShow(location_);
}


mtcnn::mtcnn(int row, int col, int minsize){
    nms_threshold[0] = 0.7;
    nms_threshold[1] = 0.5;
    nms_threshold[2] = 0.3;

    float minl = row<col?row:col;
    int MIN_DET_SIZE = 36;
    float m = (float)MIN_DET_SIZE/minsize;
    float factor = 0.709;
	//float factor = 0.5;
    int factor_count = 0;

    while(minl*m>MIN_DET_SIZE){
        scales_.push_back(m);
        m *= factor;
    }
    simpleFace_ = new Pnet[scales_.size()];
}

mtcnn::~mtcnn(){
    delete []simpleFace_;
}

vector<Rect> mtcnn::detectObject(Mat &image){

	vector<Rect> objs;
	struct orderScore order;
	int count = 0;
	for (size_t i = 0; i < scales_.size(); i++) {
		int changedH = (int)ceil(image.rows*scales_.at(i));
		int changedW = (int)ceil(image.cols*scales_.at(i));
		resize(image, reImage, Size(changedW, changedH), 0, 0, cv::INTER_LINEAR);
		simpleFace_[i].run(reImage, scales_.at(i));
		nms(simpleFace_[i].boundingBox_, simpleFace_[i].bboxScore_, simpleFace_[i].nms_threshold);

		for (vector<struct Bbox>::iterator it = simpleFace_[i].boundingBox_.begin(); it != simpleFace_[i].boundingBox_.end(); it++){
			if ((*it).exist){
				firstBbox_.push_back(*it);
				order.score = (*it).score;
				order.oriOrder = count;
				firstOrderScore_.push_back(order);
				count++;
			}
		}
		simpleFace_[i].bboxScore_.clear();
		simpleFace_[i].boundingBox_.clear();
	}
	//the first stage's nms
	if (count<1)return objs;
	nms(firstBbox_, firstOrderScore_, nms_threshold[0]);
	//refineAndSquareBbox(firstBbox_, image.rows, image.cols, true);
	refineAndSquareBbox(firstBbox_, image.rows, image.cols, false);
	//画出第一个网络检测到的矩形框
	for (vector<struct Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++){
		if ((*it).exist){
			Rect temp((*it).x1, (*it).y1, (*it).x2 - (*it).x1, (*it).y2 - (*it).y1);
			if (temp.width < 36 || temp.height < 72){
				(*it).exist = false;
				continue;
			}
			//rectangle(image, temp, Scalar(0, 0, 255));
		}
	}
	//imshow("result_1", image);
	//cvWaitKey(0);

	//second stage
	count = 0;
	for (vector<struct Bbox>::iterator it = firstBbox_.begin(); it != firstBbox_.end(); it++){
		if ((*it).exist){
			Rect temp((*it).x1, (*it).y1, (*it).x2 - (*it).x1, (*it).y2 - (*it).y1);
			//if (temp.width < 24 || temp.height < 72){
			//	(*it).exist = false;
			//	continue;
			//}

			Mat secImage;
			resize(image(temp), secImage, Size(24,72), 0, 0, cv::INTER_LINEAR);
			refineNet.run(secImage);
			//printf("RNet: %f / %f\n", *(refineNet.score_->pdata + 1), refineNet.Rthreshold);
			if (*(refineNet.score_->pdata + 1)>refineNet.Rthreshold){
				memcpy(it->regreCoord, refineNet.location_->pdata, 4 * sizeof(mydataFmt));
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = *(refineNet.score_->pdata + 1);
				secondBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				secondBboxScore_.push_back(order);
				//rectangle(image, Rect((*it).x1, (*it).y1, (*it).x2 - (*it).x1, (*it).y2 - (*it).y1), Scalar(0, 0, 255));
			}
			else{
				(*it).exist = false;
			}
		}
	}
	if (count<1)return objs;
	nms(secondBbox_, secondBboxScore_, nms_threshold[1]);
	refineAndSquareBbox(secondBbox_, image.rows, image.cols, false);
	//imshow("result_2", image);
	//cvWaitKey(0);

	//third stage 
	count = 0;
	for (vector<struct Bbox>::iterator it = secondBbox_.begin(); it != secondBbox_.end(); it++){
		if ((*it).exist){
			Rect temp((*it).x1, (*it).y1, (*it).x2 - (*it).x1, (*it).y2 - (*it).y1);
			//if (temp.width < 24 || temp.height < 72){
			//	(*it).exist = false;
			//	continue;
			//}
			

			Mat thirdImage;
			resize(image(temp), thirdImage, Size(48, 144), 0, 0, cv::INTER_LINEAR);
			outNet.run(thirdImage);
			mydataFmt *pp = NULL;

			//printf("ONet: %f / %f\n", *(outNet.score_->pdata + 1), outNet.Othreshold);
			if (*(outNet.score_->pdata + 1)>outNet.Othreshold){
				memcpy(it->regreCoord, outNet.location_->pdata, 4 * sizeof(mydataFmt));
				it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
				it->score = *(outNet.score_->pdata + 1);
				thirdBbox_.push_back(*it);
				order.score = it->score;
				order.oriOrder = count++;
				thirdBboxScore_.push_back(order);
			}
			else{
				it->exist = false;
			}
		}
	}

	if (count<1)return objs;
	refineAndSquareBbox(thirdBbox_, image.rows, image.cols, false);
	nms(thirdBbox_, thirdBboxScore_, nms_threshold[2], "Min");

	for (vector<struct Bbox>::iterator it = thirdBbox_.begin(); it != thirdBbox_.end(); it++){
		if ((*it).exist){
			objs.push_back(Rect(it->x1, it->y1, it->x2 - it->x1 + 1, it->y2 - it->y1 + 1));
		}
	}

	firstBbox_.clear();
	firstOrderScore_.clear();
	secondBbox_.clear();
	secondBboxScore_.clear();
	thirdBbox_.clear();
	thirdBboxScore_.clear();
	return objs;
}
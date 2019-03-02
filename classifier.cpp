#if _WIN32
#include <vx_ext_amd.h>
#include <vx_ext_winml.h>
#include <vx_winml.h>
#else
#include <vx_ext_amd.h>
#include <vx_amd_nn.h>
#endif
#include <iostream>
#include <sstream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <string>
#include <inttypes.h>
#include <chrono>
#if _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <math.h>
#include <immintrin.h>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#define CVUI_IMPLEMENTATION
#include "cvui.h"


#ifdef _MSC_VER 
//not #if defined(_WIN32) || defined(_WIN64) because we have strncasecmp in mingw
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

using namespace cv;
using namespace std;


#if ENABLE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#define CVUI_IMPLEMENTATION
#include "cvui.h"
using namespace cv;
#endif

#define MIVisionX_LEGEND "MIVisionX Image Classification"


unsigned char colors[20][3] = {
	{ 0,255,0 },
{ 0, 0,255 },
{ 255, 0,0 },
{ 250, 150, 70 },
{ 102,102,156 },
{ 190,153,153 },
{ 0,  0,   0 },
{ 250,170, 30 },
{ 220,220,  0 },
{ 0, 255, 0 },
{ 152,251,152 },
{ 135,206,250 },
{ 220, 20, 60 },
{ 255,  0,  0 },
{ 0,  0,255 },
{ 0,  0, 70 },
{ 0, 60,100 },
{ 0, 80,100 },
{ 0,  0,230 },
{ 119, 11, 32 }
};

std::string classificationModels[20] = {
	"InceptionV2",
	"Resnet50",
	"VGG19",
	"Shufflenet",
	"Squeezenet",
	"Densenet121",
	"Zfnet512",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified",
	"Unclassified"
};

// probability track bar
const int threshold_slider_max = 100;
int threshold_slider;
double thresholdValue = 0.5;
void threshold_on_trackbar(int, void*) {
	thresholdValue = (double)threshold_slider / threshold_slider_max;
	return;
}

bool runInception, runResnet50, runVgg19, runShufflenet, runSqueezenet, runDensenet121, runZfnet512;
float inceptionV2Time_g, resnet50Time_g, vgg19Time_g, shufflenetTime_g, squeezenetTime_g, densenet121Time_g, zfnet512Time_g;

void createLegendImage()
{
	// create display legend image
	int fontFace = CV_FONT_HERSHEY_DUPLEX;
	double fontScale = 0.75;
	int thickness = 1.3;
	cv::Size legendGeometry = cv::Size(625, (10 * 40) + 40);
	Mat legend = Mat::zeros(legendGeometry, CV_8UC3);
	Rect roi = Rect(0, 0, 625, (10 * 40) + 40);
	legend(roi).setTo(cv::Scalar(128, 128, 128));
	int l = 0, model = 0;
	int red, green, blue;
	std::string className;
	std::string bufferName;
	char buffer[50];

	// add headers
	bufferName = "MIVisionX Image Classification";
	putText(legend, bufferName, Point(20, (l * 40) + 30), fontFace, 1.2, cv::Scalar(66, 13, 9), thickness, 5);
	l++;
	bufferName = "Model";
	putText(legend, bufferName, Point(100, (l * 40) + 30), fontFace, 1, Scalar::all(0), thickness, 5);
	bufferName = "ms/frame";
	putText(legend, bufferName, Point(300, (l * 40) + 30), fontFace, 1, Scalar::all(0), thickness, 5);
	bufferName = "Color";
	putText(legend, bufferName, Point(525, (l * 40) + 30), fontFace, 1, Scalar::all(0), thickness, 5);
	l++;

	// add legend items
	thickness = 1;
	red = (colors[model][2]); green = (colors[model][1]); blue = (colors[model][0]);
	className = classificationModels[model];
	sprintf_s(buffer, " %.2f ", inceptionV2Time_g);
	cvui::checkbox(legend, 30, (l * 40) + 15, "", &runInception);
	putText(legend, className, Point(80, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	putText(legend, buffer, Point(320, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	rectangle(legend, Point(550, (l * 40)), Point(575, (l * 40) + 40), Scalar(red, green, blue), -1);
	l++; model++;
	red = (colors[model][2]); green = (colors[model][1]); blue = (colors[model][0]);
	className = classificationModels[model];
	sprintf_s(buffer, " %.2f ", resnet50Time_g);
	cvui::checkbox(legend, 30, (l * 40) + 15, "", &runResnet50);
	putText(legend, className, Point(80, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	putText(legend, buffer, Point(320, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	rectangle(legend, Point(550, (l * 40)), Point(575, (l * 40) + 40), Scalar(red, green, blue), -1);
	l++; model++;
	red = (colors[model][2]); green = (colors[model][1]); blue = (colors[model][0]);
	className = classificationModels[model];
	sprintf_s(buffer, " %.2f ", vgg19Time_g);
	cvui::checkbox(legend, 30, (l * 40) + 15, "", &runVgg19);
	putText(legend, className, Point(80, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	putText(legend, buffer, Point(320, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	rectangle(legend, Point(550, (l * 40)), Point(575, (l * 40) + 40), Scalar(red, green, blue), -1);
	l++; model++;
	red = (colors[model][2]); green = (colors[model][1]); blue = (colors[model][0]);
	className = classificationModels[model];
	sprintf_s(buffer, " %.2f ", shufflenetTime_g);
	cvui::checkbox(legend, 30, (l * 40) + 15, "", &runShufflenet);
	putText(legend, className, Point(80, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	putText(legend, buffer, Point(320, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	rectangle(legend, Point(550, (l * 40)), Point(575, (l * 40) + 40), Scalar(red, green, blue), -1);
	l++; model++;
	red = (colors[model][2]); green = (colors[model][1]); blue = (colors[model][0]);
	className = classificationModels[model];
	sprintf_s(buffer, " %.2f ", squeezenetTime_g);
	cvui::checkbox(legend, 30, (l * 40) + 15, "", &runSqueezenet);
	putText(legend, className, Point(80, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	putText(legend, buffer, Point(320, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	rectangle(legend, Point(550, (l * 40)), Point(575, (l * 40) + 40), Scalar(red, green, blue), -1);
	l++; model++;
	red = (colors[model][2]); green = (colors[model][1]); blue = (colors[model][0]);
	className = classificationModels[model];
	sprintf_s(buffer, " %.2f ", densenet121Time_g);
	cvui::checkbox(legend, 30, (l * 40) + 15, "", &runDensenet121);
	putText(legend, className, Point(80, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	putText(legend, buffer, Point(320, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	rectangle(legend, Point(550, (l * 40)), Point(575, (l * 40) + 40), Scalar(red, green, blue), -1);
	l++; model++;
	red = (colors[model][2]); green = (colors[model][1]); blue = (colors[model][0]);
	className = classificationModels[model];
	sprintf_s(buffer, " %.2f ", zfnet512Time_g);
	cvui::checkbox(legend, 30, (l * 40) + 15, "", &runZfnet512);
	putText(legend, className, Point(80, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	putText(legend, buffer, Point(320, (l * 40) + 30), fontFace, fontScale, Scalar::all(0), thickness, 3);
	rectangle(legend, Point(550, (l * 40)), Point(575, (l * 40) + 40), Scalar(red, green, blue), -1);
	l++; model++;

	cvui::trackbar(legend, 100, (l * 40) + 10, 450, &threshold_slider, 0, 100);

	cvui::update();
	cv::imshow(MIVisionX_LEGEND, legend);

	thresholdValue = (double)threshold_slider / threshold_slider_max;
}

#define ERROR_CHECK_OBJECT(obj) { vx_status status = vxGetStatus((vx_reference)(obj)); if(status != VX_SUCCESS) { vxAddLogEntry((vx_reference)context, status     , "ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return status; } }
#define ERROR_CHECK_STATUS(call) { vx_status status = (call); if(status != VX_SUCCESS) { printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status, __LINE__); return -1; } }

static void VX_CALLBACK log_callback(vx_context context, vx_reference ref, vx_status status, const vx_char string[])
{
	size_t len = strlen(string);
	if (len > 0) {
		printf("%s", string);
		if (string[len - 1] != '\n')
			printf("\n");
		fflush(stdout);
	}
}

inline int64_t clockCounter()
{
	return std::chrono::high_resolution_clock::now().time_since_epoch().count();
}

inline int64_t clockFrequency()
{
	return std::chrono::high_resolution_clock::period::den / std::chrono::high_resolution_clock::period::num;
}

int main(int argc, const char ** argv)
{
	// check command-line usage
	if (argc < 8) {
		printf(
			"\n"
			"Usage: ./classifier <inceptionV2 model.onnx> <resnet50 model.onnx> <vgg19 model.onnx> <shufflenet model.onnx>"
			"<squeezenet model.onnx> <densenet121 model.onnx> <zfnet512 model.onnx> [ --label <label text> --video <video file>/<--capture 0> ] \n"
			"\n"
		);
		return -1;
	}

	const char * onnxModel_inception = argv[1];
	const char * onnxModel_resnet = argv[2];
	const char * onnxModel_vgg = argv[3];
	const char * onnxModel_shufflenet = argv[4];
	const char * onnxModel_squeezenet = argv[5];
	const char * onnxModel_densenet121 = argv[6];
	const char * onnxModel_zfnet512 = argv[7];

	argc -= 8;
	argv += 8;
	
	std::string videoFile = "empty";
	std::string labelFileName = "empty";
	std::string labelText[1000];
	int captureID = -1;

	bool captureFromVideo = false;
	
	if (argc && !strcasecmp(*argv, "--label"))
	{
		argc--;
		argv++;
		labelFileName = *argv;
		std::string line;
		std::ifstream out(labelFileName);
		int lineNum = 0;
		while (getline(out, line)) {
			labelText[lineNum] = line;
			lineNum++;
		}
		out.close();
		argc--;
		argv++;
	}
	if (argc && !strcasecmp(*argv, "--video"))
	{
		argv++;
		videoFile = *argv;
		captureFromVideo = true;
	}
	else if (argc && !strcasecmp(*argv, "--capture"))
	{
		argv++;
		captureID = atoi(*argv);
	}

	// create context, input, output, and graph
	vxRegisterLogCallback(NULL, log_callback, vx_false_e);
	vx_context context = vxCreateContext();
	vx_status status = vxGetStatus((vx_reference)context);
	if (status) {
		printf("ERROR: vxCreateContext() failed\n");
		return -1;
	}
	vxRegisterLogCallback(context, log_callback, vx_false_e);

	// load vx_nn kernels
	ERROR_CHECK_STATUS(vxLoadKernels(context, "vx_winml"));

	//input tensor dimensions
	vx_size dims_data_224X224[4] = { 224, 224, 3, 1 };

	// output tensor prob dimensions
	vx_size dims_prob[2] = { 1000, 1 };
	vx_size dims_prob_2[4] = { 1, 1, 1000, 1 };

	char labelBuf[2048];
	//labelFileName = "C:\\Users\\lakshmi\\work\\MIVisionX_winmlFix\\MIVisionX_winmlFix\\amd_openvx_extensions\\amd_winml\\samples\\data\\Labels.txt";
	int n = sprintf_s(labelBuf, "%s", labelFileName.c_str());
	vx_scalar labelDir = vxCreateScalar(context, VX_TYPE_STRING_AMD, &labelFileName);
	ERROR_CHECK_STATUS(vxWriteScalarValue(labelDir, labelBuf));

	vx_float32 num_a = 1.0, num_b = 0.0;
	vx_float32 incpeption_a = 0.007843, inception_b = -1.0;
	vx_bool reverse = 0;
	vx_int32 device = 0;
	vx_scalar a = vxCreateScalar(context, VX_TYPE_FLOAT32, &num_a);
	vx_scalar b = vxCreateScalar(context, VX_TYPE_FLOAT32, &num_b);
	vx_scalar a_inception = vxCreateScalar(context, VX_TYPE_FLOAT32, &incpeption_a);
	vx_scalar b_inception = vxCreateScalar(context, VX_TYPE_FLOAT32, &inception_b);
	vx_scalar rev = vxCreateScalar(context, VX_TYPE_BOOL, &reverse);
	vx_scalar deviceKind = vxCreateScalar(context, VX_TYPE_INT32, &device);


	//top k (=1) answers
	vx_scalar top1_vgg19 = vxCreateScalar(context, VX_TYPE_STRING_AMD, &top1_vgg19);
	vx_scalar top1_squeezenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &top1_squeezenet);
	vx_scalar top1_shufflenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &top1_shufflenet);
	vx_scalar top1_densenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &top1_densenet);
	vx_scalar top1_resnet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &top1_resnet);
	vx_scalar top1_inception = vxCreateScalar(context, VX_TYPE_STRING_AMD, &top1_inception);
	vx_scalar top1_zfnet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &top1_zfnet);

	//model Location, input and output tensor names for all models
	//vgg19
	char model_vgg19Buf[2048];
	n = sprintf_s(model_vgg19Buf, "%s", onnxModel_vgg);
	string inputTensor_vgg19 = "data_0";
	char inputTensor_vgg19Buf[2048];
	n = sprintf_s(inputTensor_vgg19Buf, "%s", inputTensor_vgg19.c_str());
	string outputTensor_vgg19 = "prob_1";
	char outputTensor_vgg19Buf[2048];
	n = sprintf_s(outputTensor_vgg19Buf, "%s", outputTensor_vgg19.c_str());

	//squeezenet
	char model_squeezenetBuf[2048];
	n = sprintf_s(model_squeezenetBuf, "%s", onnxModel_squeezenet);
	string inputTensor_squeezenet = "data_0";
	char inputTensor_squeezenetBuf[2048];
	n = sprintf_s(inputTensor_squeezenetBuf, "%s", inputTensor_squeezenet.c_str());
	string outputTensor_squeezenet = "softmaxout_1";
	char outputTensor_squeezenetBuf[2048];
	n = sprintf_s(outputTensor_squeezenetBuf, "%s", outputTensor_squeezenet.c_str());

	//densenet
	char model_densenetBuf[2048];
	n = sprintf_s(model_densenetBuf, "%s", onnxModel_densenet121);
	string inputTensor_densenet = "data_0";
	char inputTensor_densenetBuf[2048];
	n = sprintf_s(inputTensor_densenetBuf, "%s", inputTensor_densenet.c_str());
	string outputTensor_densenet = "fc6_1";
	char outputTensor_densenetBuf[2048];
	n = sprintf_s(outputTensor_densenetBuf, "%s", outputTensor_densenet.c_str());

	//inception
	char model_inceptionBuf[2048];
	n = sprintf_s(model_inceptionBuf, "%s", onnxModel_inception);
	string inputTensor_inception = "data_0";
	char inputTensor_inceptionBuf[2048];
	n = sprintf_s(inputTensor_inceptionBuf, "%s", inputTensor_inception.c_str());
	string outputTensor_inception = "prob_1";
	char outputTensor_inceptionBuf[2048];
	n = sprintf_s(outputTensor_inceptionBuf, "%s", outputTensor_inception.c_str());

	//resnet
	char model_resnetBuf[2048];
	n = sprintf_s(model_resnetBuf, "%s", onnxModel_resnet);
	string inputTensor_resnet = "gpu_0/data_0";
	char inputTensor_resnetBuf[2048];
	n = sprintf_s(inputTensor_resnetBuf, "%s", inputTensor_resnet.c_str());
	string outputTensor_resnet = "gpu_0/softmax_1";
	char outputTensor_resnetBuf[2048];
	n = sprintf_s(outputTensor_resnetBuf, "%s", outputTensor_resnet.c_str());

	//shufflenet
	char model_shufflenetBuf[2048];
	n = sprintf_s(model_shufflenetBuf, "%s", onnxModel_shufflenet);
	string inputTensor_shufflenet = "gpu_0/data_0";
	char inputTensor_shufflenetBuf[2048];
	n = sprintf_s(inputTensor_shufflenetBuf, "%s", inputTensor_shufflenet.c_str());
	string outputTensor_shufflenet = "gpu_0/softmax_1";
	char outputTensor_shufflenetBuf[2048];
	n = sprintf_s(outputTensor_shufflenetBuf, "%s", outputTensor_shufflenet.c_str());

	//zfnet
	char model_zfnetBuf[2048];
	n = sprintf_s(model_zfnetBuf, "%s", onnxModel_zfnet512);
	string inputTensor_zfnet = "gpu_0/data_0";
	char inputTensor_zfnetBuf[2048];
	n = sprintf_s(inputTensor_zfnetBuf, "%s", inputTensor_zfnet.c_str());
	string outputTensor_zfnet = "gpu_0/softmax_1";
	char outputTensor_zfnetBuf[2048];
	n = sprintf_s(outputTensor_zfnetBuf, "%s", outputTensor_zfnet.c_str());

	//create openvx input image
	vx_image input_image = vxCreateImage(context, 224, 224, VX_DF_IMAGE_RGB);
	if (vxGetStatus((vx_reference)input_image)) {
		printf("ERROR: createImage() at input failed (%d)\n", status);
		return -1;
	}

	//creating input tensors
	vx_tensor data_224x224_vgg19 = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_vgg19);
	vx_tensor data_224x224_squeezenet = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_squeezenet);
	vx_tensor data_224x224_densenet121 = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_densenet121); 
	vx_tensor data_224x224_inception = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_inception);
	vx_tensor data_224x224_resnet = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_resnet);
	vx_tensor data_224x224_shufflenet = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_shufflenet);
	vx_tensor data_224x224_zfnet512 = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_zfnet512);


	//creating output tensors
	vx_tensor prob_vgg19 = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_vgg19);
	vx_tensor prob_squeezenet = vxCreateTensor(context, 4, dims_prob_2, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_squeezenet);
	vx_tensor prob_densenet121 = vxCreateTensor(context, 4, dims_prob_2, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_densenet121);
	vx_tensor prob_inception = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_inception);
	vx_tensor prob_resnet = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_resnet);
	vx_tensor prob_shufflenet = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_shufflenet);
	vx_tensor prob_zfnet512 = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_zfnet512);


	//vgg19 scalars
	vx_scalar modelLocation_vgg19 = vxCreateScalar(context, VX_TYPE_STRING_AMD, &onnxModel_vgg);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelLocation_vgg19, model_vgg19Buf));
	vx_scalar modelInputName_vgg19 = vxCreateScalar(context, VX_TYPE_STRING_AMD, &inputTensor_vgg19);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelInputName_vgg19, inputTensor_vgg19Buf));
	vx_scalar modelOutputName_vgg19 = vxCreateScalar(context, VX_TYPE_STRING_AMD, &outputTensor_vgg19);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelOutputName_vgg19, outputTensor_vgg19Buf));

	//squeezenet scalars
	
	vx_scalar modelLocation_squeezenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &onnxModel_squeezenet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelLocation_squeezenet, model_squeezenetBuf));
	vx_scalar modelInputName_squeezenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &inputTensor_squeezenet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelInputName_squeezenet, inputTensor_squeezenetBuf));
	vx_scalar modelOutputName_squeezenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &outputTensor_squeezenet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelOutputName_squeezenet, outputTensor_squeezenetBuf));

	//densenet scalars
	
	vx_scalar modelLocation_densenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &onnxModel_densenet121);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelLocation_densenet, model_densenetBuf));
	vx_scalar modelInputName_densenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &inputTensor_densenet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelInputName_densenet, inputTensor_densenetBuf));
	vx_scalar modelOutputName_densenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &outputTensor_densenet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelOutputName_densenet, outputTensor_densenetBuf));

	//resnet scalars
	vx_scalar modelLocation_resnet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &onnxModel_resnet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelLocation_resnet, model_resnetBuf));
	vx_scalar modelInputName_resnet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &inputTensor_resnet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelInputName_resnet, inputTensor_resnetBuf));
	vx_scalar modelOutputName_resnet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &outputTensor_resnet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelOutputName_resnet, outputTensor_resnetBuf));


	//inception scalars
	vx_scalar modelLocation_inception = vxCreateScalar(context, VX_TYPE_STRING_AMD, &onnxModel_resnet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelLocation_inception, model_inceptionBuf));
	vx_scalar modelInputName_inception = vxCreateScalar(context, VX_TYPE_STRING_AMD, &inputTensor_resnet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelInputName_inception, inputTensor_inceptionBuf));
	vx_scalar modelOutputName_inception = vxCreateScalar(context, VX_TYPE_STRING_AMD, &outputTensor_resnet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelOutputName_inception, outputTensor_inceptionBuf));

	//shufflenet scalars
	vx_scalar modelLocation_shufflenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &onnxModel_shufflenet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelLocation_shufflenet, model_shufflenetBuf));
	vx_scalar modelInputName_shufflenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &inputTensor_shufflenet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelInputName_shufflenet, inputTensor_shufflenetBuf));
	vx_scalar modelOutputName_shufflenet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &outputTensor_shufflenet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelOutputName_shufflenet, outputTensor_shufflenetBuf));

	//zfnet512 scalars

	vx_scalar modelLocation_zfnet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &onnxModel_zfnet512);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelLocation_zfnet, model_zfnetBuf));
	vx_scalar modelInputName_zfnet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &inputTensor_zfnet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelInputName_zfnet, inputTensor_zfnetBuf));
	vx_scalar modelOutputName_zfnet = vxCreateScalar(context, VX_TYPE_STRING_AMD, &outputTensor_zfnet);
	ERROR_CHECK_STATUS(vxWriteScalarValue(modelOutputName_zfnet, outputTensor_zfnetBuf));

	runInception = true; runResnet50 = true; runVgg19 = true;
	runShufflenet = true; runSqueezenet = true;
	runDensenet121 = true; runZfnet512 = true;


	std::string inceptionText = "Unclassified", resnetText = "Unclassified", vggText = "Unclassified", shufflenetText = "Unclassified";
	std::string squeezenetText = "Unclassified", densenetText = "Unclassified", zfnetText = "Unclassified";

	int64_t freq = clockFrequency(), t0, t1;
	t0 = clockCounter();

	/***** OPENCV Additions *****/

	// create display windows
	cv::namedWindow(MIVisionX_LEGEND);
	cvui::init(MIVisionX_LEGEND);
	cv::namedWindow("MIVisionX Image Classification - LIVE", cv::WINDOW_GUI_EXPANDED);

	//create a probability track bar
	threshold_slider = 50;
	//cv::createTrackbar("Probability Threshold", MIVisionX_LEGEND, &threshold_slider, threshold_slider_max, threshold_on_trackbar);

	// create display legend image
	createLegendImage();

	// define variables for run
	cv::Mat frame;
	int total_size = 1000;
	int outputImgWidth = 1080, outputImgHeight = 720;
	float threshold = 0.01;
	cv::Size output_geometry = cv::Size(outputImgWidth, outputImgHeight);
	Mat inputDisplay, outputDisplay;

	cv::Mat inputFrame_224x224;
	int fontFace = CV_FONT_HERSHEY_DUPLEX;
	double fontScale = 1;
	int thickness = 1.5;

	int loopSeg = 1;

	while (argc && loopSeg)
	{
		VideoCapture cap;
		if (captureFromVideo) {
			cap.open(videoFile);
			if (!cap.isOpened()) {
				std::cout << "Unable to open the video: " << videoFile << std::endl;
				return 0;
			}
		}
		else {
			cap.open(captureID);
			if (!cap.isOpened()) {
				std::cout << "Unable to open the camera feed: " << captureID << std::endl;
				return 0;
			}
		}

		int frameCount = 0;
		float msFrame = 0, fpsAvg = 0, frameMsecs = 0;
		for (;;)
		{
			msFrame = 0;
			// capture image frame
			t0 = clockCounter();
			cap >> frame;
			if (frame.empty()) break; // end of video stream
			t1 = clockCounter();
			msFrame += (float)(t1 - t0)*1000.0f / (float)freq;
			//printf("\n\nLIVE: OpenCV Frame Capture Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

			// preprocess image frame
			t0 = clockCounter();
			cv::resize(frame, inputFrame_224x224, cv::Size(224, 224));
			t1 = clockCounter();
			msFrame += (float)(t1 - t0)*1000.0f / (float)freq;
			//printf("LIVE: OpenCV Frame Resize Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

				if (frame.empty()) {
				printf("Image not found\n");
			}
			cv::resize(frame, frame, Size(224, 224));

			vx_rectangle_t cv_image_region;
			cv_image_region.start_x = 0;
			cv_image_region.start_y = 0;
			cv_image_region.end_x = 224;
			cv_image_region.end_y = 224;
			vx_imagepatch_addressing_t cv_image_layout;
			cv_image_layout.stride_x = 3;
			cv_image_layout.stride_y = frame.step;
			vx_uint8 * cv_image_buffer = frame.data;

			ERROR_CHECK_STATUS(vxCopyImagePatch(input_image, &cv_image_region, 0,
				&cv_image_layout, cv_image_buffer,
				VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
			
		
			if (runVgg19)
			{
				vx_graph graph_vgg19 = vxCreateGraph(context);
				ERROR_CHECK_OBJECT(graph_vgg19);

				vx_node nodes[] =
				{
					vxExtWinMLNode_convertImageToTensor(graph_vgg19, input_image, data_224x224_vgg19, a, b, rev),
					vxExtWinMLNode_OnnxToMivisionX(graph_vgg19, modelLocation_vgg19, modelInputName_vgg19, modelOutputName_vgg19, data_224x224_vgg19, prob_vgg19, deviceKind),
					vxExtWinMLNode_getTopKLabels(graph_vgg19, prob_vgg19, labelDir, top1_vgg19, NULL, NULL, NULL, NULL)
				};

				for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
				{
					ERROR_CHECK_OBJECT(nodes[i]);
					ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
				}
				ERROR_CHECK_STATUS(vxVerifyGraph(graph_vgg19));
				t0 = clockCounter();
				ERROR_CHECK_STATUS(vxProcessGraph(graph_vgg19)); 
				t1 = clockCounter();
				vgg19Time_g = (float)(t1 - t0)*1000.0f / (float)freq;
				//printf("LIVE: Process vgg19 Classification Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);
				char top1_result[1024];
				ERROR_CHECK_STATUS(vxReadScalarValue(top1_vgg19, top1_result));
				vggText = top1_result;

				ERROR_CHECK_STATUS(vxReleaseGraph(&graph_vgg19));
								
			}
			
			if (runSqueezenet)
			{
				vx_graph graph_squeezenet = vxCreateGraph(context);
				ERROR_CHECK_OBJECT(graph_squeezenet);
				vx_node nodes[] =
				{
						vxExtWinMLNode_convertImageToTensor(graph_squeezenet, input_image, data_224x224_squeezenet, a, b, rev),
						vxExtWinMLNode_OnnxToMivisionX(graph_squeezenet, modelLocation_squeezenet, modelInputName_squeezenet, modelOutputName_squeezenet, data_224x224_squeezenet, prob_squeezenet, deviceKind),
						vxExtWinMLNode_getTopKLabels(graph_squeezenet, prob_squeezenet, labelDir, top1_squeezenet, NULL, NULL, NULL, NULL)
				};

				for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
				{
					ERROR_CHECK_OBJECT(nodes[i]);
					ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
				}

				ERROR_CHECK_STATUS(vxVerifyGraph(graph_squeezenet));
				t0 = clockCounter();
				ERROR_CHECK_STATUS(vxProcessGraph(graph_squeezenet));
				t1 = clockCounter();
				squeezenetTime_g = (float)(t1 - t0)*1000.0f / (float)freq;
				//printf("LIVE: Process squeezenet Classification Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

				char top1_result[1024];
				ERROR_CHECK_STATUS(vxReadScalarValue(top1_squeezenet, top1_result));
				squeezenetText = (top1_result);

				ERROR_CHECK_STATUS(vxReleaseGraph(&graph_squeezenet));
			
			}

			if (runDensenet121)
			{
				vx_graph graph_densenet121 = vxCreateGraph(context);
				ERROR_CHECK_OBJECT(graph_densenet121);

				vx_node nodes[] =
				{
					   vxExtWinMLNode_convertImageToTensor(graph_densenet121, input_image, data_224x224_densenet121, a, b, rev),
					   vxExtWinMLNode_OnnxToMivisionX(graph_densenet121, modelLocation_densenet, modelInputName_densenet, modelOutputName_densenet, data_224x224_densenet121, prob_densenet121, deviceKind),
					   vxExtWinMLNode_getTopKLabels(graph_densenet121, prob_densenet121, labelDir, top1_densenet, NULL, NULL, NULL, NULL)
				};

				for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
				{
					ERROR_CHECK_OBJECT(nodes[i]);
					ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
				}

				ERROR_CHECK_STATUS(vxVerifyGraph(graph_densenet121));
				t0 = clockCounter();
				ERROR_CHECK_STATUS(vxProcessGraph(graph_densenet121));
				t1 = clockCounter();
				densenet121Time_g = (float)(t1 - t0)*1000.0f / (float)freq;
				//printf("LIVE: Process densenet121 Classification Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

				char top1_result[1024];
				ERROR_CHECK_STATUS(vxReadScalarValue(top1_densenet, top1_result));
				densenetText =  top1_result;

				ERROR_CHECK_STATUS(vxReleaseGraph(&graph_densenet121));


			}
			
			if (runInception)
			{

				vx_graph graph_inception = vxCreateGraph(context);
				ERROR_CHECK_OBJECT(graph_inception);

				vx_node nodes[] =
				{
					   vxExtWinMLNode_convertImageToTensor(graph_inception, input_image, data_224x224_inception, a_inception, b_inception, rev),
					   vxExtWinMLNode_OnnxToMivisionX(graph_inception, modelLocation_inception, modelInputName_inception, modelOutputName_inception, data_224x224_inception, prob_inception, deviceKind),
					   vxExtWinMLNode_getTopKLabels(graph_inception, prob_inception, labelDir, top1_inception, NULL, NULL, NULL, NULL)
				};
				
				for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
				{
					ERROR_CHECK_OBJECT(nodes[i]);
					ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
				}

				ERROR_CHECK_STATUS(vxVerifyGraph(graph_inception));
				t0 = clockCounter();
				ERROR_CHECK_STATUS(vxProcessGraph(graph_inception));
				t1 = clockCounter();
				inceptionV2Time_g = (float)(t1 - t0)*1000.0f / (float)freq;
				//printf("LIVE: Process inceptionV2 Classification Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

				char top1_result[1024];
				ERROR_CHECK_STATUS(vxReadScalarValue(top1_inception, top1_result));
				inceptionText = top1_result;

				ERROR_CHECK_STATUS(vxReleaseGraph(&graph_inception));
				

			}
			
			if (runResnet50)
			{
				vx_graph graph_resnet = vxCreateGraph(context);
				ERROR_CHECK_OBJECT(graph_resnet);

				vx_node nodes[] =
				{
					   vxExtWinMLNode_convertImageToTensor(graph_resnet, input_image, data_224x224_resnet, a, b, rev),
					   vxExtWinMLNode_OnnxToMivisionX(graph_resnet, modelLocation_resnet, modelInputName_resnet, modelOutputName_resnet, data_224x224_resnet, prob_resnet, deviceKind),
					   vxExtWinMLNode_getTopKLabels(graph_resnet, prob_resnet, labelDir, top1_resnet, NULL, NULL, NULL, NULL)
				};
				
				for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
				{
					ERROR_CHECK_OBJECT(nodes[i]);
					ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
				}


				ERROR_CHECK_STATUS(vxVerifyGraph(graph_resnet));
				t0 = clockCounter();
				ERROR_CHECK_STATUS(vxProcessGraph(graph_resnet));
				t1 = clockCounter();
				resnet50Time_g = (float)(t1 - t0)*1000.0f / (float)freq;
				//printf("LIVE: Process resnet50 Classification Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

				char top1_result[1024];
				ERROR_CHECK_STATUS(vxReadScalarValue(top1_resnet, top1_result));
				resnetText = top1_result;

				ERROR_CHECK_STATUS(vxReleaseGraph(&graph_resnet));
			}
			
			if (runShufflenet)
			{
				vx_graph graph_shufflenet = vxCreateGraph(context);
				ERROR_CHECK_OBJECT(graph_shufflenet);


				vx_node nodes[] =
				{
					   vxExtWinMLNode_convertImageToTensor(graph_shufflenet, input_image, data_224x224_shufflenet, a, b, rev),
					   vxExtWinMLNode_OnnxToMivisionX(graph_shufflenet, modelLocation_shufflenet, modelInputName_shufflenet, modelOutputName_shufflenet, data_224x224_shufflenet, prob_shufflenet, deviceKind),
					   vxExtWinMLNode_getTopKLabels(graph_shufflenet, prob_shufflenet, labelDir, top1_shufflenet, NULL, NULL, NULL, NULL)
				};
				
				for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
				{
					ERROR_CHECK_OBJECT(nodes[i]);
					ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
				}

				ERROR_CHECK_STATUS(vxVerifyGraph(graph_shufflenet));
				t0 = clockCounter();
				ERROR_CHECK_STATUS(vxProcessGraph(graph_shufflenet));
				t1 = clockCounter();
				shufflenetTime_g = (float)(t1 - t0)*1000.0f / (float)freq;
				//printf("LIVE: Process shufflenet Classification Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

				char top1_result[1024];
				ERROR_CHECK_STATUS(vxReadScalarValue(top1_shufflenet, top1_result));
				shufflenetText = top1_result;


				ERROR_CHECK_STATUS(vxReleaseGraph(&graph_shufflenet));
			}
			
			if (runZfnet512)
			{
				
				vx_graph graph_zfnet512 = vxCreateGraph(context);
				ERROR_CHECK_OBJECT(graph_zfnet512);


				vx_node nodes[] =
				{
					   vxExtWinMLNode_convertImageToTensor(graph_zfnet512, input_image, data_224x224_zfnet512, a, b, rev),
					   vxExtWinMLNode_OnnxToMivisionX(graph_zfnet512, modelLocation_zfnet, modelInputName_zfnet, modelOutputName_zfnet, data_224x224_zfnet512, prob_zfnet512, deviceKind),
					   vxExtWinMLNode_getTopKLabels(graph_zfnet512, prob_zfnet512, labelDir, top1_zfnet, NULL, NULL, NULL, NULL)
				};
				for (vx_size i = 0; i < sizeof(nodes) / sizeof(nodes[0]); i++)
				{
					ERROR_CHECK_OBJECT(nodes[i]);
					ERROR_CHECK_STATUS(vxReleaseNode(&nodes[i]));
				}


				ERROR_CHECK_STATUS(vxVerifyGraph(graph_zfnet512));
				t0 = clockCounter();
				ERROR_CHECK_STATUS(vxProcessGraph(graph_zfnet512));
				t1 = clockCounter();
				zfnet512Time_g = (float)(t1 - t0)*1000.0f / (float)freq;
				//printf("LIVE: Process zfnet Classification Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

				char top1_result[1024];
				ERROR_CHECK_STATUS(vxReadScalarValue(top1_zfnet, top1_result));
				zfnetText = top1_result;


				ERROR_CHECK_STATUS(vxReleaseGraph(&graph_zfnet512));
			}
			
			t1 = clockCounter();
			msFrame += (float)(t1 - t0)*1000.0f / (float)freq;
			//printf("LIVE: Convert Image to Tensor Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

			
			// Write Output on Image
			t0 = clockCounter();
			cv::resize(frame, outputDisplay, cv::Size(outputImgWidth, outputImgHeight));
			int l = 1;
			std::string modelName1 = "InceptionV4 - ";
			std::string modelName2 = "Resnet50 - ";
			std::string modelName3 = "VGG19 - ";
			std::string modelName4 = "ShuffleNet - ";
			std::string modelName5 = "Squeezenet - ";
			std::string modelName6 = "Densenet121 - ";
			std::string modelName7 = "Zfnet512 - ";

			modelName1 = modelName1 + inceptionText;
			modelName2 = modelName2 + resnetText;
			modelName3 = modelName3 + vggText;
			modelName4 = modelName4 + shufflenetText;
			modelName5 = modelName5 + squeezenetText;
			modelName6 = modelName6 + densenetText;
			modelName7 = modelName7 + zfnetText;
			int red, green, blue;
			if (runInception)
			{
				red = (colors[0][2]); green = (colors[0][1]); blue = (colors[0][0]);
				putText(outputDisplay, modelName1, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar(red, green, blue), thickness, 8);
				l++;
			}
			if (runResnet50)
			{
				red = (colors[1][2]); green = (colors[1][1]); blue = (colors[1][0]);
				putText(outputDisplay, modelName2, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar(red, green, blue), thickness, 8);
				l++;
			}
			if (runVgg19)
			{
				red = (colors[2][2]); green = (colors[2][1]); blue = (colors[2][0]);
				putText(outputDisplay, modelName3, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar(red, green, blue), thickness, 8);
				l++;
			}
			if (runShufflenet)
			{
				red = (colors[3][2]); green = (colors[3][1]); blue = (colors[3][0]);
				putText(outputDisplay, modelName4, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar(red, green, blue), thickness, 8);
				l++;
			}
			if (runSqueezenet)
			{
				red = (colors[4][2]); green = (colors[4][1]); blue = (colors[4][0]);
				putText(outputDisplay, modelName5, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar(red, green, blue), thickness, 8);
				l++;
			}
			if (runDensenet121)
			{
				red = (colors[5][2]); green = (colors[5][1]); blue = (colors[5][0]);
				putText(outputDisplay, modelName6, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar(red, green, blue), thickness, 8);
				l++;
			}
			if (runZfnet512)
			{
				red = (colors[6][2]); green = (colors[6][1]); blue = (colors[6][0]);
				putText(outputDisplay, modelName7, Point(20, (l * 40) + 30), fontFace, fontScale, Scalar(red, green, blue), thickness, 8);
				l++;
			}
			t1 = clockCounter();
			msFrame += (float)(t1 - t0)*1000.0f / (float)freq;
			//printf("LIVE: Resize and write on Output Image Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

			// display img time
			t0 = clockCounter();
			cv::imshow("MIVisionX Image Classification - LIVE", outputDisplay);
			createLegendImage();
			t1 = clockCounter();
			msFrame += (float)(t1 - t0)*1000.0f / (float)freq;
			//printf("LIVE: Output Image Display Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

			// calculate FPS
			//printf("LIVE: msec for frame -- %.3f msec\n", (float)msFrame);
			frameMsecs += msFrame;
			if (frameCount && frameCount % 10 == 0) {
				printf("FPS LIVE: Avg FPS -- %d\n", (int)((ceil)(1000 / (frameMsecs / 10))));
				frameMsecs = 0;
			}
			// wait to close live inference application
			if (waitKey(2) == 27) { loopSeg = 0; break; } // stop capturing by pressing ESC
			else if (waitKey(2) == 82) { break; } // for restart pressing R

			frameCount++;
		}
	}

	
	// release input data
	ERROR_CHECK_STATUS(vxReleaseTensor(&data_224x224_inception));
	ERROR_CHECK_STATUS(vxReleaseTensor(&data_224x224_resnet));
	ERROR_CHECK_STATUS(vxReleaseTensor(&data_224x224_vgg19));
	ERROR_CHECK_STATUS(vxReleaseTensor(&data_224x224_squeezenet));
	ERROR_CHECK_STATUS(vxReleaseTensor(&data_224x224_shufflenet));
	ERROR_CHECK_STATUS(vxReleaseTensor(&data_224x224_densenet121));
	ERROR_CHECK_STATUS(vxReleaseTensor(&data_224x224_zfnet512));

	// release output data
	ERROR_CHECK_STATUS(vxReleaseTensor(&prob_inception));
	ERROR_CHECK_STATUS(vxReleaseTensor(&prob_resnet));
	ERROR_CHECK_STATUS(vxReleaseTensor(&prob_vgg19));
	ERROR_CHECK_STATUS(vxReleaseTensor(&prob_squeezenet));
	ERROR_CHECK_STATUS(vxReleaseTensor(&prob_shufflenet));
	ERROR_CHECK_STATUS(vxReleaseTensor(&prob_densenet121));
	ERROR_CHECK_STATUS(vxReleaseTensor(&prob_zfnet512));
	
	printf("OK: successful\n");

	return 0;
}

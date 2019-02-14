#if _WIN32
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
	"GoogleNet",
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

bool runInception, runResnet50, runVgg19, runGooglenet, runSqueezenet, runDensenet121, runZfnet512;
float inceptionV2Time_g, resnet50Time_g, vgg19Time_g, googlenetTime_g, squeezenetTime_g, densenet121Time_g, zfnet512Time_g;

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
	sprintf_s(buffer, " %.2f ", googlenetTime_g);
	cvui::checkbox(legend, 30, (l * 40) + 15, "", &runGooglenet);
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
			"Usage: ./classifier <inceptionV2 model.onnx> <resnet50 model.onnx> <vgg19 model.onnx> <googlenet model.onnx>"
			"<squeezenet model.onnx> <densenet121 model.onnx> <zfnet512 model.onnx> [ --label <label text> --video <video file>/<--capture 0> ] \n"
			"\n"
		);
		return -1;
	}
	const char * onnxModel_inception = argv[1];
	const char * onnxModel_resnet = argv[2];
	const char * onnxModel_vgg = argv[3];
	const char * onnxModel_googlenet = argv[4];
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

	// creation of graphs
	vx_graph graph_inception = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph_inception);

	vx_graph graph_resnet = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph_resnet);

	vx_graph graph_vgg19 = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph_vgg19);

	vx_graph graph_googlenet = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph_googlenet);

	vx_graph graph_squeezenet = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph_squeezenet);

	vx_graph graph_densenet121 = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph_densenet121);

	vx_graph graph_zfnet512 = vxCreateGraph(context);
	ERROR_CHECK_OBJECT(graph_zfnet512);


	// create and initialize input tensor data
	vx_size dims_data_224X224[4] = { 224, 224, 3, 1 };

	// create data for different sizes
	vx_tensor data_224x224_inception = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_inception);

	vx_tensor data_224x224_resnet = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_resnet);

	vx_tensor data_224x224_vgg19 = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_vgg19);

	vx_tensor data_224x224_googlenet = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_googlenet);

	vx_tensor data_224x224_squeezenet = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_squeezenet);

	vx_tensor data_224x224_densenet121 = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_densenet121);

	vx_tensor data_224x224_zfnet512 = vxCreateTensor(context, 4, dims_data_224X224, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(data_224x224_zfnet512);



	// create output tensor prob
	vx_size dims_prob[2] = { 1, 1000 };
	vx_size dims_prob_2[4] = { 1, 1000, 1, 1 };

	vx_tensor prob_inception = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_inception);

	vx_tensor prob_resnet = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_resnet);

	vx_tensor prob_vgg19 = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_vgg19);

	vx_tensor prob_googlenet = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_googlenet);

	vx_tensor prob_squeezenet = vxCreateTensor(context, 4, dims_prob_2, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_squeezenet);

	vx_tensor prob_densenet121 = vxCreateTensor(context, 4, dims_prob_2, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_densenet121);

	vx_tensor prob_zfnet512 = vxCreateTensor(context, 2, dims_prob, VX_TYPE_FLOAT32, 0);
	ERROR_CHECK_OBJECT(prob_zfnet512);


	const char * inceptionV2_inputTensorName = "data_0";
	const char * inceptionV2_outputTensorName = "prob_1";
	const char * resnet50_inputTensorName = "gpu_0/data_0";
	const char * resnet50_outputTensorName = "gpu_0/softmax_1";
	const char * vgg19_inputTensorName = "data_0";
	const char * vgg19_outputTensorName = "prob_1";
	const char * googlenet_inputTensorName = "data_0";
	const char * googlenet_outputTensorName = "softmaxout_1";
	const char * squeezenet_inputTensorName = "data_0";
	const char * squeezenet_outputTensorName = "softmaxout_1";
	const char * densenet121_inputTensorName = "data_0";
	const char * densenet121_outputTensorName = "fc6_1";
	const char * zfnet512_inputTensorName = "gpu_0/data_0";
	const char * zfnet512_outputTensorName = "gpu_0/softmax_1";

	//convert onnx to MIVisionX using winml
	int64_t freq = clockFrequency(), t0, t1;	
	t0 = clockCounter();

	vx_node MIVisionX_inception = vxExtWinMLNode_OnnxToMivisionX(graph_inception, (vx_scalar)onnxModel_inception, (vx_scalar)inceptionV2_inputTensorName, (vx_scalar)inceptionV2_outputTensorName, data_224x224_inception, prob_inception, NULL);
	if (vxGetStatus((vx_reference)MIVisionX_inception)) {
		printf("ERROR: inception OnnxToMivisionX() failed (%d)\n", status);
		return -1;
	}
	vx_node MIVisionX_resnet = vxExtWinMLNode_OnnxToMivisionX(graph_resnet, (vx_scalar)onnxModel_resnet, (vx_scalar)resnet50_inputTensorName, (vx_scalar)resnet50_outputTensorName, data_224x224_resnet, prob_resnet, NULL);
	if (vxGetStatus((vx_reference)MIVisionX_resnet)) {
		printf("ERROR: resnet50 OnnxToMivisionX() failed (%d)\n", status);
		return -1;
	}
	vx_node MIVisionX_vgg19 = vxExtWinMLNode_OnnxToMivisionX(graph_vgg19, (vx_scalar)onnxModel_vgg, (vx_scalar)vgg19_inputTensorName, (vx_scalar)vgg19_outputTensorName, data_224x224_vgg19, prob_vgg19, NULL);
	if (vxGetStatus((vx_reference)MIVisionX_vgg19)) {
		printf("ERROR: vgg19 OnnxToMivisionX() failed (%d)\n", status);
		return -1;
	}
	vx_node MIVisionX_googlenet = vxExtWinMLNode_OnnxToMivisionX(graph_googlenet, (vx_scalar)onnxModel_googlenet, (vx_scalar)googlenet_inputTensorName, (vx_scalar)googlenet_outputTensorName, data_224x224_googlenet, prob_googlenet, NULL);
	if (vxGetStatus((vx_reference)MIVisionX_googlenet)) {
		printf("ERROR: googlenet OnnxToMivisionX() failed (%d)\n", status);
		return -1;
	}
	vx_node MIVisionX_squeezenet = vxExtWinMLNode_OnnxToMivisionX(graph_squeezenet, (vx_scalar)onnxModel_squeezenet, (vx_scalar)squeezenet_inputTensorName, (vx_scalar)squeezenet_outputTensorName, data_224x224_squeezenet, prob_squeezenet , NULL);
	if (vxGetStatus((vx_reference)MIVisionX_squeezenet)) {
		printf("ERROR: squeezenet OnnxToMivisionX() failed (%d)\n", status);
		return -1;
	}
	vx_node MIVisionX_densenet121 = vxExtWinMLNode_OnnxToMivisionX(graph_densenet121, (vx_scalar)onnxModel_densenet121, (vx_scalar)densenet121_inputTensorName, (vx_scalar)densenet121_outputTensorName, data_224x224_densenet121, prob_densenet121, NULL);
	if (vxGetStatus((vx_reference)MIVisionX_densenet121)) {
		printf("ERROR: densenet121 OnnxToMivisionX() failed (%d)\n", status);
		return -1;
	}
	vx_node MIVisionX_zfnet512 = vxExtWinMLNode_OnnxToMivisionX(graph_zfnet512, (vx_scalar)onnxModel_zfnet512, (vx_scalar)zfnet512_inputTensorName, (vx_scalar)zfnet512_outputTensorName , data_224x224_zfnet512, prob_zfnet512, NULL);
	if (vxGetStatus((vx_reference)MIVisionX_zfnet512)) {
		printf("ERROR: zfnet512 OnnxToMivisionX() failed (%d)\n", status);
		return -1;
	}

	t1 = clockCounter();
	printf("OK:node initialization with vxExtWinMLNode_OnnxToMivisionX() took %.3f msec (1st iteration)\n", (float)(t1 - t0)*1000.0f / (float)freq);


	t0 = clockCounter();
	status = vxVerifyGraph(graph_inception);
    if(status) {
        printf("ERROR: inception vxVerifyGraph(...) failed (%d)\n", status);
        return -1;
    }

    status = vxVerifyGraph(graph_resnet);
    if(status) {
        printf("ERROR: resnet50 vxVerifyGraph(...) failed (%d)\n", status);
        return -1;
    }

    status = vxVerifyGraph(graph_vgg19);
    if(status) {
        printf("ERROR: vgg19 vxVerifyGraph(...) failed (%d)\n", status);
        return -1;
    }

    status = vxVerifyGraph(graph_googlenet);
    if(status) {
        printf("ERROR: googlenet vxVerifyGraph(...) failed (%d)\n", status);
        return -1;
    }

    status = vxVerifyGraph(graph_squeezenet);
    if(status) {
        printf("ERROR: squeezenet vxVerifyGraph(...) failed (%d)\n", status);
        return -1;
    }

    status = vxVerifyGraph(graph_densenet121);
    if(status) {
        printf("ERROR: densenet121 vxVerifyGraph(...) failed (%d)\n", status);
        return -1;
    }

    status = vxVerifyGraph(graph_zfnet512);
    if(status) {
        printf("ERROR: zfnet512 vxVerifyGraph(...) failed (%d)\n", status);
        return -1;
    }

    t1 = clockCounter();
    printf("OK: vxVerifyGraph() took %.3f msec (1st iteration)\n", (float)(t1-t0)*1000.0f/(float)freq);

	t0 = clockCounter();
    status = vxProcessGraph(graph_inception);
    if(status != VX_SUCCESS) {
        printf("ERROR: inceptionv2 vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    status = vxProcessGraph(graph_resnet);
    if(status != VX_SUCCESS) {
        printf("ERROR: resnet50 vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    status = vxProcessGraph(graph_vgg19);
    if(status != VX_SUCCESS) {
        printf("ERROR: vgg19 vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    status = vxProcessGraph(graph_googlenet);
    if(status != VX_SUCCESS) {
        printf("ERROR: googlenet vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    status = vxProcessGraph(graph_squeezenet);
    if(status != VX_SUCCESS) {
        printf("ERROR: squeezenet vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    status = vxProcessGraph(graph_densenet121);
    if(status != VX_SUCCESS) {
        printf("ERROR: densenet121 vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    status = vxProcessGraph(graph_zfnet512);
    if(status != VX_SUCCESS) {
        printf("ERROR: zfnet512 vxProcessGraph() failed (%d)\n", status);
        return -1;
    }
    t1 = clockCounter();
    printf("OK: vxProcessGraph() took %.3f msec (1st iteration)\n", (float)(t1-t0)*1000.0f/(float)freq);

	int N = 100;
	float inceptionV2Time, resnet50Time, vgg19Time, googlenetTime, densenet121Time, squeezenetTime, zfnet512Time;
	t0 = clockCounter();
	for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph_inception);
        if(status != VX_SUCCESS)
            break;
    }
	t1 = clockCounter();
	inceptionV2Time = (float)(t1 - t0)*1000.0f / (float)freq / (float)N;
	printf("OK: inceptionV2 took %.3f msec (average over %d iterations)\n", (float)(t1 - t0)*1000.0f / (float)freq / (float)N, N);
	t0 = clockCounter();
	for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph_resnet);
        if(status != VX_SUCCESS)
            break;
    }
	t1 = clockCounter();
	resnet50Time = (float)(t1 - t0)*1000.0f / (float)freq / (float)N;
	printf("OK: resnet50 took %.3f msec (average over %d iterations)\n", (float)(t1 - t0)*1000.0f / (float)freq / (float)N, N);
	t0 = clockCounter();
	for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph_vgg19);
        if(status != VX_SUCCESS)
            break;
    }
	t1 = clockCounter();
	vgg19Time = (float)(t1 - t0)*1000.0f / (float)freq / (float)N;
	printf("OK: vgg19 took %.3f msec (average over %d iterations)\n", (float)(t1 - t0)*1000.0f / (float)freq / (float)N, N);
	t0 = clockCounter();
	for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph_googlenet);
        if(status != VX_SUCCESS)
            break;
    }
	t1 = clockCounter();
	googlenetTime = (float)(t1 - t0)*1000.0f / (float)freq / (float)N;
	printf("OK: googlenet took %.3f msec (average over %d iterations)\n", (float)(t1 - t0)*1000.0f / (float)freq / (float)N, N);
	t0 = clockCounter();
	for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph_squeezenet);
        if(status != VX_SUCCESS)
            break;
    }
	t1 = clockCounter();
	squeezenetTime = (float)(t1 - t0)*1000.0f / (float)freq / (float)N;
	printf("OK: squeezenet took %.3f msec (average over %d iterations)\n", (float)(t1 - t0)*1000.0f / (float)freq / (float)N, N);
	t0 = clockCounter();
	for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph_densenet121);
        if(status != VX_SUCCESS)
            break;
    }
	t1 = clockCounter();
	densenet121Time = (float)(t1 - t0)*1000.0f / (float)freq / (float)N;
	printf("OK: densenet121 took %.3f msec (average over %d iterations)\n", (float)(t1 - t0)*1000.0f / (float)freq / (float)N, N);
	t0 = clockCounter();
	for(int i = 0; i < N; i++) {
        status = vxProcessGraph(graph_zfnet512);
        if(status != VX_SUCCESS)
            break;
    }
	t1 = clockCounter();
	zfnet512Time = (float)(t1 - t0)*1000.0f / (float)freq / (float)N;
	printf("OK: zfnet512 took %.3f msec (average over %d iterations)\n", (float)(t1 - t0)*1000.0f / (float)freq / (float)N, N);

	/***** OPENCV Additions *****/

	// create display windows
	cv::namedWindow(MIVisionX_LEGEND);
	cvui::init(MIVisionX_LEGEND);
	cv::namedWindow("MIVisionX Image Classification - LIVE", cv::WINDOW_GUI_EXPANDED);

	//create a probability track bar
	threshold_slider = 50;
	//cv::createTrackbar("Probability Threshold", MIVisionX_LEGEND, &threshold_slider, threshold_slider_max, threshold_on_trackbar);

	// create display legend image
	runInception = true; runResnet50 = true; runVgg19 = true;
	runGooglenet = true; runSqueezenet = true;
	runDensenet121 = true; runZfnet512 = true;
	inceptionV2Time_g = inceptionV2Time; resnet50Time_g = resnet50Time;
	vgg19Time_g = vgg19Time; densenet121Time_g = densenet121Time;
	googlenetTime_g = googlenetTime; zfnet512Time_g = zfnet512Time; squeezenetTime_g = squeezenetTime;
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
	float *outputBuffer[7];
	for (int models = 0; models < 7; models++) {
		outputBuffer[models] = new float[total_size];
	}

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

			// Copy Image frame into the input tensor
			t0 = clockCounter();
			vx_enum usage = VX_WRITE_ONLY;
			vx_enum data_type = VX_TYPE_FLOAT32;
			vx_size num_of_dims = 4, dims[4] = { 1, 1, 1, 1 }, stride[4];
			vx_map_id map_id;
			float * ptr;
			vx_size count;
			if (runInception)
			{
				vxQueryTensor(data_224x224_inception, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
				vxQueryTensor(data_224x224_inception, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
				vxQueryTensor(data_224x224_inception, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
				if (data_type != VX_TYPE_FLOAT32) {
					std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for " << std::endl;
					return -1;
				}
				count = dims[0] * dims[1] * dims[2] * dims[3];
				vx_status status = vxMapTensorPatch(data_224x224_inception, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
				if (status) {
					std::cerr << "ERROR: vxMapTensorPatch() failed for " << std::endl;
					return -1;
				}
				Mat srcImg;
				for (size_t n = 0; n < dims[3]; n++) {
					srcImg = inputFrame_224x224;
					for (vx_size y = 0; y < dims[1]; y++) {
						unsigned char * src = srcImg.data + y * dims[0] * 3;
						float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
						float * dstG = dstR + (stride[2] >> 2);
						float * dstB = dstG + (stride[2] >> 2);
						for (vx_size x = 0; x < dims[0]; x++, src += 3) {
							*dstR++ = (src[2] * 0.007843137) - 1;
							*dstG++ = (src[1] * 0.007843137) - 1;
							*dstB++ = (src[0] * 0.007843137) - 1;
						}
					}
				}
				status = vxUnmapTensorPatch(data_224x224_inception, map_id);
				if (status) {
					std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << std::endl;
					return -1;
				}
			}
			if (runResnet50)
			{
				vxQueryTensor(data_224x224_resnet, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
				vxQueryTensor(data_224x224_resnet, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
				vxQueryTensor(data_224x224_resnet, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
				if (data_type != VX_TYPE_FLOAT32) {
					std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for " << std::endl;
					return -1;
				}
				count = dims[0] * dims[1] * dims[2] * dims[3];
				vx_status status = vxMapTensorPatch(data_224x224_resnet, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
				if (status) {
					std::cerr << "ERROR: vxMapTensorPatch() failed for " << std::endl;
					return -1;
				}
				Mat srcImg;
				for (size_t n = 0; n < dims[3]; n++) {
					srcImg = inputFrame_224x224;
					for (vx_size y = 0; y < dims[1]; y++) {
						unsigned char * src = srcImg.data + y * dims[0] * 3;
						float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
						float * dstG = dstR + (stride[2] >> 2);
						float * dstB = dstG + (stride[2] >> 2);
						for (vx_size x = 0; x < dims[0]; x++, src += 3) {
							*dstR++ = src[2];
							*dstG++ = src[1];
							*dstB++ = src[0];
						}
					}
				}
				status = vxUnmapTensorPatch(data_224x224_resnet, map_id);
				if (status) {
					std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << std::endl;
					return -1;
				}
			}
			if (runVgg19)
			{
				vxQueryTensor(data_224x224_vgg19, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
				vxQueryTensor(data_224x224_vgg19, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
				vxQueryTensor(data_224x224_vgg19, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
				if (data_type != VX_TYPE_FLOAT32) {
					std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for " << std::endl;
					return -1;
				}
				count = dims[0] * dims[1] * dims[2] * dims[3];
				vx_status status = vxMapTensorPatch(data_224x224_vgg19, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
				if (status) {
					std::cerr << "ERROR: vxMapTensorPatch() failed for " << std::endl;
					return -1;
				}
				Mat srcImg;
				for (size_t n = 0; n < dims[3]; n++) {
					srcImg = inputFrame_224x224;
					for (vx_size y = 0; y < dims[1]; y++) {
						unsigned char * src = srcImg.data + y * dims[0] * 3;
						float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
						float * dstG = dstR + (stride[2] >> 2);
						float * dstB = dstG + (stride[2] >> 2);
						for (vx_size x = 0; x < dims[0]; x++, src += 3) {
							*dstR++ = src[2];
							*dstG++ = src[1];
							*dstB++ = src[0];
						}
					}
				}
				status = vxUnmapTensorPatch(data_224x224_vgg19, map_id);
				if (status) {
					std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << std::endl;
					return -1;
				}
			}
			if (runGooglenet)
			{
				vxQueryTensor(data_224x224_googlenet, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
				vxQueryTensor(data_224x224_googlenet, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
				vxQueryTensor(data_224x224_googlenet, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
				if (data_type != VX_TYPE_FLOAT32) {
					std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for " << std::endl;
					return -1;
				}
				count = dims[0] * dims[1] * dims[2] * dims[3];
				vx_status status = vxMapTensorPatch(data_224x224_googlenet, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
				if (status) {
					std::cerr << "ERROR: vxMapTensorPatch() failed for " << std::endl;
					return -1;
				}
				Mat srcImg;
				for (size_t n = 0; n < dims[3]; n++) {
					srcImg = inputFrame_224x224;
					for (vx_size y = 0; y < dims[1]; y++) {
						unsigned char * src = srcImg.data + y * dims[0] * 3;
						float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
						float * dstG = dstR + (stride[2] >> 2);
						float * dstB = dstG + (stride[2] >> 2);
						for (vx_size x = 0; x < dims[0]; x++, src += 3) {
							*dstR++ = src[2];
							*dstG++ = src[1];
							*dstB++ = src[0];
						}
					}
				}
				status = vxUnmapTensorPatch(data_224x224_googlenet, map_id);
				if (status) {
					std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << std::endl;
					return -1;
				}
			}
			if (runSqueezenet)
			{
				vxQueryTensor(data_224x224_squeezenet, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
				vxQueryTensor(data_224x224_squeezenet, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
				vxQueryTensor(data_224x224_squeezenet, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
				if (data_type != VX_TYPE_FLOAT32) {
					std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for " << std::endl;
					return -1;
				}
				count = dims[0] * dims[1] * dims[2] * dims[3];
				vx_status status = vxMapTensorPatch(data_224x224_squeezenet, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
				if (status) {
					std::cerr << "ERROR: vxMapTensorPatch() failed for " << std::endl;
					return -1;
				}
				Mat srcImg;
				for (size_t n = 0; n < dims[3]; n++) {
					srcImg = inputFrame_224x224;
					for (vx_size y = 0; y < dims[1]; y++) {
						unsigned char * src = srcImg.data + y * dims[0] * 3;
						float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
						float * dstG = dstR + (stride[2] >> 2);
						float * dstB = dstG + (stride[2] >> 2);
						for (vx_size x = 0; x < dims[0]; x++, src += 3) {
							*dstR++ = src[2];
							*dstG++ = src[1];
							*dstB++ = src[0];
						}
					}
				}
				status = vxUnmapTensorPatch(data_224x224_squeezenet, map_id);
				if (status) {
					std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << std::endl;
					return -1;
				}
			}
			if (runDensenet121)
			{
				vxQueryTensor(data_224x224_densenet121, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
				vxQueryTensor(data_224x224_densenet121, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
				vxQueryTensor(data_224x224_densenet121, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
				if (data_type != VX_TYPE_FLOAT32) {
					std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for " << std::endl;
					return -1;
				}
				count = dims[0] * dims[1] * dims[2] * dims[3];
				vx_status status = vxMapTensorPatch(data_224x224_densenet121, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
				if (status) {
					std::cerr << "ERROR: vxMapTensorPatch() failed for " << std::endl;
					return -1;
				}
				Mat srcImg;
				for (size_t n = 0; n < dims[3]; n++) {
					srcImg = inputFrame_224x224;
					for (vx_size y = 0; y < dims[1]; y++) {
						unsigned char * src = srcImg.data + y * dims[0] * 3;
						float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
						float * dstG = dstR + (stride[2] >> 2);
						float * dstB = dstG + (stride[2] >> 2);
						for (vx_size x = 0; x < dims[0]; x++, src += 3) {
							*dstR++ = src[2];
							*dstG++ = src[1];
							*dstB++ = src[0];
						}
					}
				}
				status = vxUnmapTensorPatch(data_224x224_densenet121, map_id);
				if (status) {
					std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << std::endl;
					return -1;
				}
			}
			if (runZfnet512)
			{
				vxQueryTensor(data_224x224_zfnet512, VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type));
				vxQueryTensor(data_224x224_zfnet512, VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims));
				vxQueryTensor(data_224x224_zfnet512, VX_TENSOR_DIMS, &dims, sizeof(dims[0])*num_of_dims);
				if (data_type != VX_TYPE_FLOAT32) {
					std::cerr << "ERROR: copyTensor() supports only VX_TYPE_FLOAT32: invalid for " << std::endl;
					return -1;
				}
				count = dims[0] * dims[1] * dims[2] * dims[3];
				vx_status status = vxMapTensorPatch(data_224x224_zfnet512, num_of_dims, nullptr, nullptr, &map_id, stride, (void **)&ptr, usage, VX_MEMORY_TYPE_HOST, 0);
				if (status) {
					std::cerr << "ERROR: vxMapTensorPatch() failed for " << std::endl;
					return -1;
				}
				Mat srcImg;
				for (size_t n = 0; n < dims[3]; n++) {
					srcImg = inputFrame_224x224;
					for (vx_size y = 0; y < dims[1]; y++) {
						unsigned char * src = srcImg.data + y * dims[0] * 3;
						float * dstR = ptr + ((n * stride[3] + y * stride[1]) >> 2);
						float * dstG = dstR + (stride[2] >> 2);
						float * dstB = dstG + (stride[2] >> 2);
						for (vx_size x = 0; x < dims[0]; x++, src += 3) {
							*dstR++ = src[2];
							*dstG++ = src[1];
							*dstB++ = src[0];
						}
					}
				}
				status = vxUnmapTensorPatch(data_224x224_zfnet512, map_id);
				if (status) {
					std::cerr << "ERROR: vxUnmapTensorPatch() failed for " << std::endl;
					return -1;
				}
			}

			t1 = clockCounter();
			msFrame += (float)(t1 - t0)*1000.0f / (float)freq;
			//printf("LIVE: Convert Image to Tensor Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

			// process probabilty
			t0 = clockCounter();
			threshold = (float)thresholdValue;
			const int N = 1000;
			int inceptionID, resnetID, vggID, googlenetID, squeezenetID, densenetID, zfnetID;
			if (runInception)
			{
				inceptionID = std::distance(outputBuffer[0], std::max_element(outputBuffer[0], outputBuffer[0] + N));
			}
			if (runResnet50)
			{
				resnetID = std::distance(outputBuffer[1], std::max_element(outputBuffer[1], outputBuffer[1] + N));
			}
			if (runVgg19)
			{
				vggID = std::distance(outputBuffer[2], std::max_element(outputBuffer[2], outputBuffer[2] + N));
			}
			if (runGooglenet)
			{
				googlenetID = std::distance(outputBuffer[3], std::max_element(outputBuffer[3], outputBuffer[3] + N));
			}
			if (runSqueezenet)
			{
				squeezenetID = std::distance(outputBuffer[4], std::max_element(outputBuffer[4], outputBuffer[4] + N));
			}
			if (runDensenet121)
			{
				densenetID = std::distance(outputBuffer[5], std::max_element(outputBuffer[5], outputBuffer[5] + N));
			}
			if (runZfnet512)
			{
				zfnetID = std::distance(outputBuffer[5], std::max_element(outputBuffer[6], outputBuffer[6] + N));
			}
			t1 = clockCounter();
			msFrame += (float)(t1 - t0)*1000.0f / (float)freq;
			//printf("LIVE: Get Classification ID Time -- %.3f msec\n", (float)(t1-t0)*1000.0f/(float)freq);

			// Write Output on Image
			t0 = clockCounter();
			cv::resize(frame, outputDisplay, cv::Size(outputImgWidth, outputImgHeight));
			int l = 1;
			std::string modelName1 = "InceptionV4 - ";
			std::string modelName2 = "Resnet50 - ";
			std::string modelName3 = "VGG19 - ";
			std::string modelName4 = "GoogleNet - ";
			std::string modelName5 = "Squeezenet - ";
			std::string modelName6 = "Densenet121 - ";
			std::string modelName7 = "Zfnet512 - ";
			std::string inceptionText = "Unclassified", resnetText = "Unclassified", vggText = "Unclassified", googlenetText = "Unclassified";
			std::string squeezenetText = "Unclassified", densenetText = "Unclassified", zfnetText = "Unclassified";
			if (outputBuffer[0][inceptionID] >= threshold) { inceptionText = labelText[inceptionID]; }
			if (outputBuffer[1][resnetID] >= threshold) { resnetText = labelText[resnetID]; }
			if (outputBuffer[2][vggID] >= threshold) { vggText = labelText[vggID]; }
			if (outputBuffer[3][googlenetID] >= threshold) { googlenetText = labelText[googlenetID]; }
			if (outputBuffer[4][squeezenetID] >= threshold) { squeezenetText = labelText[squeezenetID]; }
			if (outputBuffer[5][densenetID] >= threshold) { densenetText = labelText[densenetID]; }
			if (outputBuffer[6][zfnetID] >= threshold) { zfnetText = labelText[zfnetID]; }
			modelName1 = modelName1 + inceptionText;
			modelName2 = modelName2 + resnetText;
			modelName3 = modelName3 + vggText;
			modelName4 = modelName4 + googlenetText;
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
			if (runGooglenet)
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

	// release resources
	for (int models = 0; models < 7; models++) {
		delete outputBuffer[models];
	}


	printf("OK: successful\n");

	return 0;
}

import re
import numpy as np

def Calculate_IoU(ground_truth, predict):
	#Initilize the return value
	#Initialize the Ground truth object width and height
	#Since the value must be positive here, I will not use abs()
	ground_truth_width = ground_truth[1] - ground_truth[0]
	ground_truth_height = ground_truth[3] - ground_truth[2]
	#Initialize the predict object width and height
	predict_width = predict[1] - predict[0]
	predict_height = predict[3] - predict[2]

	#Initalize the parameter we need when calculating IoU.
	#I will use formula below to calculate the width and height of overlapped area:
	#overlapped_width = min(x1,x2,x3,x4)+width1+width2-max(x1,x2,x3,x4)
	#overlapped_height =  min(y1,y2,y3,y4)+height1+height2-max(y1,y2,y3,y4)

	#listing min/max(x1,x2,x3,x4) min/max(y1,y2,y3,y4)
	pesudo_xmin = min(ground_truth[0],ground_truth[1],predict[0],predict[1])
	pesudo_xmax = max(ground_truth[0],ground_truth[1],predict[0],predict[1])
	pesudo_ymin = min(ground_truth[2],ground_truth[3],predict[2],predict[3])
	pesudo_ymax = max(ground_truth[2],ground_truth[3],predict[2],predict[3])

	#Calculate width of overlapping box
	overlapped_width = pesudo_xmin + ground_truth_width + predict_width - pesudo_xmax
	overlapped_height = pesudo_ymin + ground_truth_height + predict_height - pesudo_ymax

	#Check whether there is an overlapping box
	if ((overlapped_width <=0) or (overlapped_height <= 0)):
		IoU = 0
	else:
		overlapped_area = overlapped_width * overlapped_height
		ground_truth_area = ground_truth_width * ground_truth_height
		predict_area = predict_width * predict_height
		union_area = ground_truth_area + predict_area - overlapped_area
		IoU = (overlapped_area/union_area)
	return IoU

def Average_IoU(ground_truth_txt,predict_txt):

	IoU = []
	ground_truth_name = []
	ground_truth_box = []
	predict_name = []
	predict_box = []

	ground_truth_txt = open(ground_truth_txt).readlines()
	predict_txt = open(predict_txt).readlines()

	for line in predict_txt:
	    name = line[:line.find('.jpg')]
	    line = re.sub(str(name)+'.jpg', '', line)
	    line = line[line.find('[')+1:line.find(']')].split(',')
	    box = list(map(int,line))
	    predict_name.append(name)
	    predict_box.append(box)

	for line in ground_truth_txt:
	    name = line[:line.find('.jpg')]
	    line = re.sub(str(name)+'.jpg', '', line)
	    line = line[line.find('[')+1:line.find(']')].split(',')     
	    box = list(map(int,line))
	    ground_truth_name.append(name)
	    ground_truth_box.append(box)
	    
	for i in range(len(predict_name)):
		if (predict_name[i]!=ground_truth_name[i]):
			print('predicted and ground mismatch at line ' + str(i))
			pass
		IoU.append(Calculate_IoU(ground_truth_box[i],predict_box[i]))

	IoU = np.array(IoU)
	Average_IoU = np.mean(IoU)
	print('Average_IoU: '+ str(Average_IoU))
	return Average_IoU

IoU = Average_IoU('./result/ground_truth.txt', './result/result.txt')

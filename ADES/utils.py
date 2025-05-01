import cv2 as cv
import scipy.io as sio
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing.image import img_to_array
from imutils.object_detection import non_max_suppression
import numpy as np
import argparse
import cv2
from keras.models import load_model
from sklearn import preprocessing
from matplotlib import pyplot
import PIL
from PIL import Image
import torch
from torchvision.ops import nms
from scipy.spatial import distance as dist
from collections import OrderedDict
from skimage import io
from ADES.tracker import CentroidTracker

def pad_video(video_array,pad_dimension):
  
  padded_video = np.pad(video_array, ((pad_dimension, pad_dimension),(0,0), (0,0),(0,0)), 'reflect')

  return padded_video

def selective_search(image, method="fast"):
	ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
	ss.setBaseImage(image)
 
	if method == "fast":
		ss.switchToSelectiveSearchFast()
	else:
		ss.switchToSelectiveSearchQuality()
	rects = ss.process()
	return rects

def process_frame(padded_video,time,time_window,spatial_window,model,method,conf = 0.9, Intensity = 40, overlap=0.3):
  image = padded_video[time,:,:,:]

  map = np.zeros(image.shape)
  
  filter = None

  method = method

  labelFilters = filter
  if labelFilters is not None:
	  labelFilters = labelFilters.lower().split(",")
 

  (H, W) = image.shape[:2]
  rects = selective_search(image, method=method)
  proposals = []
  boxes = []
  temporal_batch = []
  raw_boxes = []
  nms_boxes = []
  proba = []
  
  for (x, y, w, h) in rects:
	  crop = padded_video[time,y:y+h,x:x+w,:]
	  intensity = np.mean(crop)

	  if intensity < Intensity or w > spatial_window or h > spatial_window:
		  continue

	  roi = image[y:y + h, x:x + w] #change this line of code
    

	  roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	  roi = cv2.resize(roi, (59, 59))
	  roi = img_to_array(roi)
	  roi = preprocess_input(roi)
 	
	  roi[:,:,0] = preprocessing.normalize(roi[:,:,0])
	  proposals.append(roi)
	  boxes.append((x, y, w, h))
	 
	  if time_window == 5:
		  sequence_crop = padded_video[time-3:time+2, y:y + h, x:x + w]
	  else:
		  startT = int(np.floor(time_window/2))
		  endT = int(np.ceil(time_window/2))
		  sequence_crop = padded_video[time-startT:time+endT, y:y + h, x:x + w]
		  idx = np.linspace(0, time_window-1, 5,dtype = int)
		  sequence_crop = sequence_crop[idx,:,:]


	  len_seq = sequence_crop.shape[0]

	  sequence = np.zeros((len_seq,59,59,3))

	  for nn in range(0,len_seq):
	   roi = sequence_crop[nn,:,:,:]
	   roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
	   roi = cv2.resize(roi, (59, 59))
	   roi = img_to_array(roi)
	   sequence[nn,:,:,:] = roi
          
	  temporal_batch.append(sequence)

  proposals = np.array(proposals)
  temporal_batch = np.array(temporal_batch)

  print("[INFO] proposal shape: {}".format(proposals.shape))
  print("[INFO] classifying proposals...")
  print(temporal_batch.shape)
  X = temporal_batch[:,:,:,:,0].reshape(temporal_batch.shape[0],5,59,59,1)
  Y = model.predict(X)

  predictions = np.zeros((proposals.shape[0],1000))
  predictions[:,0:1] = Y[:,1:2]

  preds = imagenet_utils.decode_predictions(predictions, top=1)

  labels = {}
  clone = image.copy()

  for (i, p) in enumerate(preds):

	  (imagenetID, label, prob) = p[0]
	 
	  if labelFilters is not None and label not in labelFilters:
		  continue

	  if prob >= conf:

		  (x, y, w, h) = boxes[i]
		  box = (x, y, x + w, y + h)

		  L = labels.get(label, [])
		  L.append((box, prob))
		  labels[label] = L

		  clone = image.copy()
		  raw_boxes = np.array([p[0] for p in labels[label]])
		  proba = np.array([p[1] for p in labels[label]])
		  proba = proba.reshape((proba.shape[0],1))
		  listBoxes = nms(boxes = torch.tensor(raw_boxes.copy(),dtype = torch.float32), scores = torch.tensor(proba.flatten(),dtype = torch.float32), iou_threshold=overlap)
		  nms_boxes = raw_boxes[listBoxes]
		  idx = 0

		  for (startX, startY, endX, endY) in raw_boxes:
		    map[startY:endY,startX:endX,0] = proba[idx]
		    idx +=1	  
		    cv2.rectangle(clone, (startX, startY), (endX, endY),
			  (0, 255, 0), 2)
		    y = startY - 10 if startY - 10 > 10 else startY + 10
    
    
  return clone,map,nms_boxes,raw_boxes,proba#,X,Y


def getTracks(bb_detections,tracer):
  n = len(bb_detections)
  bb_copy = bb_detections.copy()
  tracks = []

  for i in range(0,n):
    if bb_copy[i] != []:
      if (np.mean((bb_copy[i]).flatten() == bb_copy[i])) == 1:
        bb_copy[i] = bb_copy[i].reshape([1,4])
    
    obj = tracer.update(bb_copy[i])
    keys = np.array(list(obj.keys()))
    values = np.array(list(obj.values()))

    if len(keys) > 0:
      
      for ii in range(0,len(keys)):
        track = np.zeros((1,8))
        track[0,0]= i
        track[0,1] = keys[ii]
        track[0,2:8] = values[ii]
        tracks.append(track)

  return np.array(tracks,dtype = int).reshape((len(tracks),8))

def process_tracks(tracks,video_len,gap):
  ids = np.unique(tracks[:,1])

  filtered = np.zeros((tracks[0:1,:].shape),dtype=int)

  time_points = np.array(range(0,video_len))
  growth_curve = np.zeros((video_len,1))

  tracks_clone = tracks.copy()

  count = 0

  for id in ids:
    temp = tracks_clone[tracks_clone[:,1]==id]
    max_time = np.max(temp[:,0])
    min_time = np.min(temp[:,0])

    if max_time != video_len-1:
      temp = temp[0:len(temp)-gap,:]

    if len(temp) > 2:
      temp[temp[:,1] == id,1] = count
      filtered = np.concatenate((filtered,temp))
      growth_curve[time_points > min_time] += 1
      count +=1
  
  filtered = np.delete(filtered,0,axis = 0)

  count = np.unique(filtered[:,1])

  return filtered,count,growth_curve

def display(video,filtered_tracks): 
  processed_video = video.copy()
  tracks_clone = filtered_tracks.copy()
  count = 0
  for (frame,id,X,Y,startX, startY, endX, endY) in tracks_clone:

    clone = processed_video[int(frame),:,:,:]
    cv2.rectangle(clone, (startX, startY), (endX, endY),(0, 255, 0), 2)
    y = startY - 10 if startY - 10 > 10 else startY + 10
	
    text = "Apo ID {}".format(id)
    cv2.putText(clone, text, (X - 10, Y - 20),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    #cv2.circle(clone, (X, Y), 4, (0, 255, 0), -1)
    
    processed_video[frame,:,:,:] = clone
    
  return processed_video


def resize_video(video,um_window, pixel_size):
  t,xi,yi,ch = video.shape
  scaling_factor = pixel_size/0.5
  xii = np.int(np.ceil(xi*scaling_factor))
  yii = np.int(np.ceil(yi*scaling_factor))
  rescaled_video = np.zeros((t,xii,yii,ch),dtype = 'uint8')

  for tt in range(0,t):
    frame = video[tt,:,:,:]
    rescaled_video[tt,:,:,:] = cv2.resize(frame,(xii,yii), interpolation = cv2.INTER_CUBIC)

  return rescaled_video



def main(video,method,model,time_window,um_window,pixel_size,gap,conf,intensity,overlap): 

  tracker = CentroidTracker(gap)
  
  n_frames = video.shape[0]
  
  pad = time_window

  #spatial_window = um_window

  spatial_window = np.round(um_window/pixel_size)

  time = pad

  padded_video = pad_video(video, pad)
  
  heatmap = np.zeros(padded_video.shape)

  trajectories = []

  boundingBoxes = []

  rawData = []

  for image_frame in video:
    
    frame_processed,prob_map,nms_boxes,raw_boxes,proba = process_frame(padded_video,time,time_window,spatial_window,model,'fast',conf, intensity, overlap)

    print(time)

    boundingBoxes.append(nms_boxes)

    if raw_boxes == []:
      rawData.append([])

    else:
      rawData.append(np.concatenate((raw_boxes,proba),axis = 1 ))
      heatmap[time,:,:,:] = prob_map

    time += 1

  tracks = getTracks(boundingBoxes,tracker)

  filtered_tracks,count,curve = process_tracks(tracks,len(video),gap)

  processed_video = display(video,filtered_tracks)

  return boundingBoxes,rawData,tracks,filtered_tracks,processed_video,curve,heatmap

def read_tif(im):
  imarray = np.array(im)
  imarray.shape
  video = np.zeros((imarray.shape[0],imarray.shape[1],imarray.shape[2],3), dtype='uint8')

  n = im.shape[0]

  for fr in range(0,n):
    frame = imarray[fr,:,:]
    mn = np.min(frame)
    mx = np.max(frame)
    im2 = np.uint8(255*(frame-mn)/(mx-mn))

    min_q,max_q =np.quantile(frame,[0.001,0.999])
    scaled = (frame - min_q)*(255/(max_q-min_q))
    clipped = np.clip(scaled,0,255)
    uint8 = clipped.astype(np.uint8)


    video[fr,:,:,0] = uint8
    video[fr,:,:,1] = uint8  
    video[fr,:,:,2] = uint8

  return video


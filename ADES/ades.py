from ADES.utils import*

def predict(path_video, threshold, f1, f2):
  
  im = io.imread(path_video)[:,:,:,0]
  
  video = read_tif(im)
  
  crop = video[:,0:250,0:250,:]

  model = load_model('ADES/model_in_vitro.h5')
  
  boundingBoxes,rawTracks,tracks,filtered,processed_video,curve,heatmap = main(crop[f1+6:f2+6,:,:,:],method= 'fast',model = model,time_window = 9,um_window = 32, pixel_size = 1,gap = 4,conf = threshold,intensity = 40,overlap = 0.1)

  return filtered, processed_video
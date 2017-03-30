from watershed import Watershed
import numpy as np
import h5py
import cPickle as pickle


# stage 1
w = Watershed()

with h5py.File('aff.h5') as f:
  aff = f['main'][:]
  aff = aff.transpose((3,2,1,0))
  
  with h5py.File('0-0-0.h5','w') as seg:
    ws_result = w.stage_1(aff[:513,:,:,:], dataset_size=(1024,1024,256), chunk_pos=(0,0,0))
    segmentation_0 = ws_result['segmentation']
    dendrogram_0 = ws_result['dendogram']
    segment_sizes_front = ws_result['segment_sizes']
    seg.create_dataset('main', data= segmentation_0)

  with h5py.File('1-0-0.h5','w') as seg:
    ws_result = w.stage_1(aff[511:,:,:,:], dataset_size=(1024,1024,256), chunk_pos=(511,0,0))
    ws_result['segmentation']
    segmentation_1 = ws_result['segmentation']
    dendrogram_1 = ws_result['dendogram']
    segment_sizes_back = ws_result['segment_sizes']
    seg.create_dataset('main', data= segmentation_1 )

with open('test.pickle','wb') as f:
  pickle.dump({
        "seg_border_front":segmentation_0[-2:-1,1:-1,1:-1],
        "seg_border_back":segmentation_1[0:1,1:-1,1:-1],
        "aff_border":aff[512,1:-1,1:-1,:],
        "dendogram_front": dendrogram_0,
        "dendogram_back": dendrogram_1,
        "segment_sizes_front":segment_sizes_front,
        "segment_sizes_back":segment_sizes_back}, f)

# stage 2
with open('test.pickle','rb') as f:
  obj = pickle.load(f)

w = Watershed()
mapping = w.stage_2(**obj)

with h5py.File('0-0-0.h5','r+') as seg:
  seg['main'][:] = w.stage_3(seg['main'][:], mapping)

with h5py.File('1-0-0.h5','r+') as seg:
  seg['main'][:] = w.stage_3(seg['main'][:] + len(segment_sizes_front), mapping)
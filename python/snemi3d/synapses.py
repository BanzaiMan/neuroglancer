import pandas
import h5py
import numpy as np
from collections import defaultdict

import pickle 

def get_box_slice(center, volume, size):

  z,y,x = center
  max_z , max_y, max_x = volume.shape

  slicing =  (slice(max(0,z-size),min(max_z,z+size)),
              slice(max(0,y-(size+3)**2),min(max_y,y+(size+3)**2)),
              slice(max(0,x-(size+3)**2),min(max_x,x+(size+3)**2)))

  return volume[slicing], max(0,z-size), max(0,y-(size+3)**2), max(0,x-(size+3)**2)

def find_pre_and_post(centroid, pre, pos, volume):
  for size in xrange(1,50):
    box, min_z, min_y, min_x = get_box_slice(centroid, volume, size)
    min_corner = np.array([min_z, min_y, min_x])
    pre_where = np.where(box == pre)
    pos_where = np.where(box == pos)
    if len(pos_where[0]) and len(pre_where[0]):
      pre_rel =  np.array(map(lambda x: x[0], pre_where))
      pos_rel =  np.array(map(lambda x: x[0], pos_where))
      
      assert np.all(pre_rel + min_corner < volume.shape)
      assert np.all(pos_rel + min_corner < volume.shape)
      return pre_rel + min_corner, pos_rel + min_corner

  raise Exception('couldnt find point ', centroid, pre, pos)

with h5py.File('/usr/people/it2/evaluate/golden_cube_seg_c.h5') as f:

  df = pandas.read_csv('/usr/people/it2/evaluate/golden_cube_cons_edges.csv',sep=';',header=None)
  parsed = []
  premap = defaultdict(list)
  postmap = defaultdict(list)
  for row in df.iterrows():
    # print 
    i, row = row
    synapse_id, pre_pos, centroid, _, _ = row
    pre, pos = eval(pre_pos)
    centroid = eval(centroid)
    centroid = np.array(centroid[::-1])

    positions = find_pre_and_post(centroid, pre, pos, f['main'])


    premap[pre].append(tuple(positions[0][::-1]))
    premap[pre].append(tuple(positions[1][::-1]))
    postmap[pos].append(tuple(positions[0][::-1]))
    postmap[pos].append(tuple(positions[1][::-1]))
  

def save():

  with file('synapses.picke','w') as f:
    pickle.dump([premap, postmap], f)

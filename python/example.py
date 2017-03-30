from __future__ import print_function

import numpy as np
import h5py
import webbrowser
import neuroglancer

viewer = neuroglancer.Viewer()
neuroglancer.set_static_content_source(url='http://localhost:8080')

# viewer.add(volume_type='image', data=np.zeros(shape=(100,100,100), dtype=np.uint8), name='image', voxel_size=[6, 6, 40])
# viewer.add(volume_type='point', name='point')
# viewer.add(volume_type='synapse', name='synapse')

# 0 pad is useful to make the meshes that are in contact with the borders
# of the volume have a planar cap
# seg = np.pad(f['main'][:], 1, 'constant', constant_values=0)
chunk0 = h5py.File('./ext/third_party/watershed/0-0-0.h5','r')
viewer.add(
  volume_type='segmentation', 
  data=chunk0['main'][:].transpose((2,1,0)), 
  name='0-0-0', 
  voxel_size=[6, 6, 40],
  graph=None
)
chunk1 = h5py.File('./ext/third_party/watershed/1-0-0.h5','r')
viewer.add(
  volume_type='segmentation', 
  data=chunk1['main'][:].transpose((2,1,0)), 
  name='1-0-0', 
  voxel_size=[6, 6, 40],
  offset=[511*6,0,0],
  graph=None
)

aff = h5py.File('./ext/third_party/watershed/aff.h5','r')
viewer.add(
  volume_type='image', 
  data=aff['main'][:], 
  name='affinities', 
  voxel_size=[6, 6, 40])

print(viewer.get_viewer_url())
webbrowser.open(viewer.get_viewer_url())

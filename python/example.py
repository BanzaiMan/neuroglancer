from __future__ import print_function

import numpy as np
import h5py
import webbrowser
from collections import defaultdict
from snemi3d import synapses
# class Synapses(object):
#   premap = defaultdict(list)
#   postmap = defaultdict(list)

# synapses = Synapses

import neuroglancer

# Obtain the bundled Neuroglancer client code (HTML, CSS, and JavaScript) from
# the demo server, so that this example works even if
#
#   python setup.py bundle_client
#
# has not been run.
neuroglancer.set_static_content_source(url='http://localhost:8080')
viewer = neuroglancer.Viewer()

# def initialize(state):
  # state['layers']['point_layer']['points'] = [[1,2,3],[3,4,5]]
#   return state

# viewer.initialize_state = initialize
img =  h5py.File('/usr/people/it2/evaluate/channel.h5')['main']
seg = h5py.File('/usr/people/it2/evaluate/golden_cube_seg_c.h5')['main']

last_visible =  set()

def filter_duplicates(l):
  already = set()
  clean = []

  for i in xrange(0, len(l), 2):
    if i+1 == len(l):
      break

    synapse = (l[i], l[i+1])
    if synapse not in already:
      already.add(synapse)
      clean.append(l[i])
      clean.append(l[i+1])

  return clean

def on_state_changed(state):
  global last_visible
  try:
    visible_segments =  set(map(int, state['layers']['segmentation']['segments']))
  except KeyError:
    visible_segments = set()
    
  if visible_segments == last_visible:
    return

  print(visible_segments)

  # delete old
  print ('deleting old')
  for _id in last_visible:
    try:
      synapses.premap[_id] = []
      synapses.postmap[_id] = []
    except KeyError, e:
      print ('failed to delete old', _id)
 

  print ('Saving new')
  for i in xrange(0, len(state['layers']['synapse']['points']), 2):
    if i+1 == len(state['layers']['synapse']['points']):
      break

    pre = tuple(state['layers']['synapse']['points'][i])
    post = tuple(state['layers']['synapse']['points'][i+1])
    pre_value = seg[int(pre[2]), int(pre[1]), int(pre[0])]
    post_value = seg[int(post[2]), int(post[1]), int(post[0])]

    print ('saving ', pre_value, post_value)
    synapses.premap[pre_value].append(pre)
    synapses.premap[pre_value].append(post)
    synapses.premap[post_value].append(pre)
    synapses.premap[post_value].append(post)
  

  # Load new synpases
  state['layers']['synapse']['points'] = []
  for _id in visible_segments:
    try:
      print ('loading')
      state['layers']['synapse']['points'].extend(synapses.premap[_id])
      state['layers']['synapse']['points'].extend(synapses.postmap[_id])
    except KeyError, e:
      print ('nothing to show for', _id)
 
  state['layers']['synapse']['points'] =  filter_duplicates(state['layers']['synapse']['points']) #remove duplicates
  last_visible = visible_segments


  # pickle changes
  synapses.save()

  return state
viewer.on_state_changed = on_state_changed



# img = np.pad(f['main'][:], 1, 'constant', constant_values=0)
viewer.add(volume_type='image', data=img, name='image', voxel_size=[4, 4, 40])

# if you add this layer by itself neuroglancer doesn't know the dataset size
# viewer.add(volume_type='point', name='point_layer')

# if you add this layer by itself neuroglancer doesn't know the dataset size
viewer.add(volume_type='synapse', name='synapse')


# viewer.add(
#   volume_type='segmentation', 
#   data=seg, 
#   name='segmentation', 
#   voxel_size=[4, 4, 40], 
#   graph=False
# )

webbrowser.open(viewer.get_viewer_url())
print(viewer.get_viewer_url())


# capture arrow keys
try:
   # Python2
    import Tkinter as tk
except ImportError:
    # Python3
    import tkinter as tk

def key(event):
    """shows key or tk code for the key"""
    if event.keysym == 'Escape':
        root.destroy()
    if event.char == event.keysym:
     # normal number and letter characters
        print( 'Normal Key %r' % event.char )
    elif len(event.char) == 1:
      # charcters like []/.,><#$ also Return and ctrl/key
        print( 'Punctuation Key %r (%r)' % (event.keysym, event.char) )
    else:
      # f1 to f12, shift keys, caps lock, Home, End, Delete ...
        print( 'Special Key %r' % event.keysym )

    if event.keysym == 'Up':

    if event.keysym == 'Down':
      

root = tk.Tk()
print( "Press a key (Escape key to exit):" )
root.bind_all('<Key>', key)
# don't show the tk window
# root.withdraw()
root.mainloop()

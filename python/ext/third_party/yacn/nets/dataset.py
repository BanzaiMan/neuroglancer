import h5py
import numpy as np
import random
import itertools
from collections import defaultdict
import os.path
import gc 
import logging

from utils import static_shape, random_sample
from yacn.kernels.graph_naming import tfscope

import tensorflow as tf

class MultiDataset():

    @tfscope
    def __init__(self, directories, types, patch_size,
                 train_vols=[], test_vols=[], device='/cpu:0'):
        self.n = len(directories)
        self._directories = directories
        self._types = types
        self._placeholders = defaultdict(list)
        self._variables = defaultdict(list)
        self._multivolume = {}
        self._patch_size = patch_size
        self._train_vols = train_vols
        self._test_vols = test_vols

        with tf.device(device):
            for (name, path) in self._types.items():
                for d in directories:
                    full_path = os.path.join(d, path)
                    ph = self._create_placeholder(full_path, name)
                    self._create_variable(name, ph)
                self._create_multivolume(name)
    
    def _create_placeholder(self, full_path, name):
        with h5py.File(full_path) as f:
            ph = tf.placeholder(cast_type(f['main'].dtype),
                shape=cast_shape(f['main'].shape))
        self._placeholders[name].append(ph)
        return ph

    def _set_placeholder(self, full_path, name, idx, sess):
        ph = self._placeholders[name][idx]
        var = self._variables[name][idx]

        with h5py.File(full_path) as f:
            data = prep(name, f['main'][:])
        sess.run(var.initializer,
            feed_dict={ ph: data })

        # free memory
        del data
        gc.collect()

    def _create_variable(self, name, ph):
        var = tf.Variable(
            ph, name=name,
            trainable=False, collections=[])
        self._variables[name].append(var)

    def _create_multivolume(self, name):
        if name == 'samples':
            patch_size = (1,3)
            indexing = 'CORNER'
        else:
            patch_size = (1,) + self._patch_size + (1,)
            indexing = 'CENTRAL'

        self._multivolume[name] = MultiVolume(
            self._variables[name], patch_size, indexing)
    
    @tfscope
    def initialize(self, sess):
        logging.info("reading datasets")
        for (name, path) in self._types.items():
            for idx, d in enumerate(self._directories):
                full_path = os.path.join(d, path)
                self._set_placeholder(full_path, name, idx, sess)

    @tfscope
    def __getitem__(self, name):
        assert type(name) is str
        return self._multivolume[name]

    @tfscope
    def get_random_volume_id(self, is_train):
        """
        Get a random volume id conditioned if we are on a trainning iteration
        or a test one.
        """
        return tf.cond(is_train,
                lambda: random_sample(tf.constant(self._train_vols)),
                lambda: random_sample(tf.constant(self._test_vols)),
            )

def cast_type(t):
    if t == np.uint32:
        return tf.int32
    elif t == np.uint8:
        return tf.float32
    elif t == np.int64: #casting for samples
        return tf.int32
    else:
        raise NotImplementedError(t)

def cast_shape(s):
    if len(s)==3:
        return (1,)+s+(1,)
    elif len(s)==4:
        return (1,)+s
    elif len(s)==5:
        return s
    else:
        return s

def prep(typ,data):
    if typ in ["image", "errors"]:
        tmp=autopad(data.astype(np.float32))
        if tmp.max() > 10:
            #print "dividing by 256"
            return tmp/256
        else:
            return tmp
    elif typ in ["human_labels", "machine_labels", "labels"]:
        return autopad(data.astype(np.int32))
    elif typ in ["labels64"]:
        return autopad(data.astype(np.int64))
    elif typ in ["valid"]:
        return data.astype(np.int32)
    elif typ in ["samples"]:
        return data.astype(np.int32)
    elif typ in ["visited"]:
        return autopad(data.astype(np.int16))

def autopad(A):
    if len(A.shape)==3:
        return np.reshape(A,(1,)+A.shape+(1,))
    elif len(A.shape)==4:
        return np.reshape(A,(1,)+A.shape)
    elif len(A.shape)==5:
        return A
    else:
        raise Exception("Can't autopad")


class MultiVolume():
    def __init__(self, As, patch_size, indexing='CENTRAL'):
        self.As = map(lambda A: Volume(A, patch_size, indexing=indexing), As)
        self._patch_size = patch_size
    
    def __getitem__(self, index):
        vol_index, focus = index

        def focus_factory(i):
            def f():
                #return tf.Print(self.As[i][focus],[i])
                return self.As[i][focus]
            return f

        cases = [(tf.equal(vol_index,i), focus_factory(i))  for i in xrange(len(self.As))]
        failure_case = lambda: tf.Print(self.As[0][focus],[vol_index],message="volume not found")
        vol = tf.case(cases, default=failure_case, exclusive=True)
        vol.set_shape(self._patch_size)
        return vol

class Volume():

    def __init__(self, A, patch_size, indexing='CENTRAL', device="/cpu:0"):
        """
        Given a tensor and a patch size
        it possible to get an slice or write and slice
        where the slice is described by a point and indexing scheme
        `CENTRAL` given the central pixel get/set an slice 
        `CORNER` given the top left corner get/set an slice
        
        focus can be a tensorflow tensor or a tuple in which some values
        can be set to `RAND` where a random value is taken.
            
        Args:
            A (tensor): input array
            patch_size (list)
            indexing (str, optional)
            device (str, optional)
        """
        self._A = A
        self._patch_size = patch_size
        self._indexing = indexing
        self._device = device

    def _get_corner(self, focus):
        if self._indexing == 'CENTRAL':
            return focus - np.array([x/2 for x in self._patch_size],dtype=np.int32)
        elif self._indexing =='CORNER':
            return focus
        else:
            raise Exception("bad indexing scheme")

    def _fill_rand(self, focus):
        focus = list(focus)
        for i,s in enumerate(static_shape(self._A)):
            if focus[i] == 'RAND':
                if self._indexing == 'CENTRAL':
                    min_value = self._patch_size[i] / 2
                    max_value =  s - self._patch_size[i] / 2

                if self._indexing == 'CORNER':
                    min_value =  0
                    max_value = s - self._patch_size[i]

                focus[i] = tf.random_uniform([],minval=min_value,
                                             maxval=max_value,dtype=tf.int32)

        return focus
    
    def _get_slc(self, corner):
        corner = tf.unstack(self._get_corner(focus))
        return tuple([slice(corner[i],corner[i]+ self._patch_size[i]) for i in xrange(len(patch_size))])

    def __getitem__(self, focus):
        with tf.device(self._device):
            if type(focus) is tuple:
                focus = self._fill_rand(focus)
            corner = self._get_corner(focus)
            return tf.stop_gradient(tf.slice(self._A, corner, self._patch_size))

    def __setitem__(self, focus, val):
        with tf.device(self._device):
            slcs =  self._get_slc(self._get_corner(focus))
            return tf.stop_gradient(self._A[slcs].assign(val))


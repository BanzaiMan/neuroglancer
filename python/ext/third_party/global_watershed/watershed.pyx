# distutils: language = c++
# distutils: sources = cWatershed.cpp

# Cython interface file for wrapping the object
#
#

from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.string cimport string
cimport numpy as np
import numpy as np

# c++ interface to cython
cdef extern from "cWatershed.h":
    cdef struct stage1_obj:
        vector[unsigned int] seg
        vector[unsigned int] dendr_u;
        vector[unsigned int] dendr_v;
        vector[float] dendr_aff;
        vector[size_t] segment_sizes;

    cdef cppclass cWatershed:
        cWatershed() except +
        stage1_obj stage_1(vector[float],
                           unsigned int, unsigned int, unsigned int,
                           float, float, size_t, float, size_t,
                           bool, bool, bool, bool, bool, bool)
        vector[unsigned int] stage_2(vector[unsigned int], vector[unsigned int], vector[float],
                     vector[unsigned int], vector[unsigned int], vector[float],
                     vector[unsigned int], vector[unsigned int], vector[float],
                     vector[size_t], vector[size_t])
        vector[unsigned int] stage_3(vector[unsigned int], vector[unsigned int])

# creating a cython wrapper class
cdef class Watershed:
    cdef cWatershed *thisptr      # hold a C++ instance which we're wrapping
    def __cinit__(self):
        self.thisptr = new cWatershed()
    def __dealloc__(self):
        del self.thisptr
    def stage_1(self, aff, dataset_size=(0,0,0), chunk_pos=(0,0,0),
                high=0.97, 
                low=0.006821,
                merge_size=800,
                merge_low=0.3,
                dust_size=100):
        
        assert merge_low > low

        obj = self.thisptr.stage_1(aff.flatten(),
                                   aff.shape[0],
                                   aff.shape[1],
                                   aff.shape[2],
                                   high, low, merge_size, merge_low, dust_size,
                                   chunk_pos[0] != 0, #is there a border between chunks
                                   chunk_pos[0] + aff.shape[0] != dataset_size[0], #is there a border between chunks
                                   chunk_pos[1] != 0, #is there a border between chunks
                                   chunk_pos[1] + aff.shape[1] != dataset_size[1], #is there a border between chunks
                                   chunk_pos[2] != 0, #is there a border between chunks
                                   chunk_pos[2] + aff.shape[2] != dataset_size[2]) #is there a border between chunks
        return { 
            'segmentation': 
                np.ndarray(dtype=np.uint32, buffer=np.array(obj.seg, dtype = np.uint32),shape=aff.shape[:3], order='C'),
            'dendogram': [ t for t in zip(obj.dendr_u, obj.dendr_v, obj.dendr_aff)],
            'segment_sizes': obj.segment_sizes }

    def stage_2(self, seg_border_front, seg_border_back, aff_border,
                dendogram_front, dendogram_back,
                segment_sizes_front, segment_sizes_back):

        cdef vector[unsigned int] u_front;
        cdef vector[unsigned int] v_front;
        cdef vector[float] aff_front;
        for u, v, aff in dendogram_front:
            u_front.append(u)
            v_front.append(v)
            aff_front.append(aff)

        cdef vector[unsigned int] u_back;
        cdef vector[unsigned int] v_back;
        cdef vector[float] aff_back;
        for u, v, aff in dendogram_back:
            u_back.append(u)
            v_back.append(v)
            aff_back.append(aff)

        return self.thisptr.stage_2(seg_border_front.flatten(), seg_border_back.flatten(), aff_border.flatten(),
                             u_front, v_front, aff_front,
                             u_back, v_back, aff_back,
                             segment_sizes_front, segment_sizes_back)

    def stage_3(self, seg , mapping):
        shape = seg.shape
        new_seg = self.thisptr.stage_3(seg.flatten() , mapping)
        return np.ndarray(dtype=np.uint32, buffer=np.array(new_seg, dtype = np.uint32),shape=shape, order='C')



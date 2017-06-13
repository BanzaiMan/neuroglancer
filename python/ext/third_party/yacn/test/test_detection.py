from __future__ import print_function
import numpy as np
import tensorflow as tf
from yacn.nets.loss_functions import error_free

# def test_inference():
#     from ext.third_party.yacn.nets.discriminate3_inference import main_model

#     image =  np.random.randint(0,255, dtype=np.uint8, size=(400,400,64))
#     segmentation =  np.random.randint(0, 255, dtype=np.uint32, size=(400,400,64))
#     samples = [[130,130,50],[128,128,50]]
#     output = main_model.inference(image, segmentation, samples)
#     print (output)


def error_free_conv(obj_ml, hl, window_size=None, window_overlap=None):
    max_pool = tf.nn.max_pool(obj_ml * hl, ksize=[1,2,2,1], strides=[1,1,1,1], padding="VALID")
    upsample = tf.nn.conv2d_transpose(
            max_pool, filter=tf.ones([2,2,1,1]),
            output_shape=[1,2,2,1], strides=[1,2,2,1], padding='VALID')
    max_id = tf.to_float(tf.equal(upsample, hl))

    hl_sum = tf.nn.conv2d(max_id, filter=tf.ones([2,2,1,1]), strides=[1,2,2,1], padding="VALID")
    ml_sum = tf.nn.conv2d(obj_ml, filter=tf.ones([2,2,1,1]), strides=[1,2,2,1], padding="VALID")
    hl_ml_sum = tf.nn.conv2d(obj_ml * max_id, filter=tf.ones([2,2,1,1]), strides=[1,2,2,1], padding="VALID")
    is_segmentation_equal = tf.logical_and( tf.equal(hl_sum, hl_ml_sum),  tf.equal(ml_sum, hl_ml_sum))
    return tf.logical_or(tf.less(ml_sum , 0.5) , is_segmentation_equal)

def create_2d(a,b,c,d):
    return tf.constant([[[[float(a)],[float(b)]],[[float(c)],[float(d)]]]], dtype=tf.float32)

def create_2d_large(a,b,c,d,e,f,g,h):
    return tf.constant([[[[float(a)],[float(b)],[float(c)],[float(d)]],
                         [[float(e)],[float(f)],[float(g)],[float(h)]]]], dtype=tf.float32)

class ErrorLocalizationWindow(tf.test.TestCase):
    """
    Single window
    """
    def assert_error(self, obj_ml, hl, has_error):
        self.assertEqual(error_free(obj_ml, hl).eval() , has_error)
        self.assertEqual(error_free_conv(obj_ml, hl).eval()[0,0,0,0], has_error)

    def testSame(self):
        with self.test_session() as sess:
            obj = create_2d(1,1,0,0)
            human_labels = create_2d(5,5,0,6)
            self.assert_error(obj, human_labels, True)

    def testOutside(self):
        with self.test_session() as sess:
            obj = create_2d(0,0,0,0)
            human_labels = create_2d(5,5,0,6)
            self.assert_error(obj, human_labels, True)

    def testMergerBoundary(self):
        with self.test_session() as sess:
            obj = create_2d(1,1,1,0)
            human_labels = create_2d(5,5,0,6)
            self.assert_error(obj, human_labels, False)

    def testMerger(self):
        with self.test_session() as sess:
            obj = create_2d(1,1,0,1)
            human_labels = create_2d(5,5,0,6)
            self.assert_error(obj, human_labels, False)

    def testSplit(self):
        with self.test_session() as sess:
            obj = create_2d(1,0,0,0)
            human_labels = create_2d(5,5,0,6)
            self.assert_error(obj, human_labels, False)

    def testMergerSplit(self):
        with self.test_session() as sess:
            obj = create_2d(1,0,0,1)
            human_labels = create_2d(5,5,0,6)
            self.assert_error(obj, human_labels, False)


class ErrorLocalization(tf.test.TestCase):
    """
    Non Overlapping Window
    """
    def test(self):
        create_2d_large

        # upds = map(lambda x: f(*[V[x] for V in us]), slices)
        # tf.scatter_nd(indices=inds, updates=upds, shape=shape[1:4])

if __name__ == '__main__':
    tf.test.main()
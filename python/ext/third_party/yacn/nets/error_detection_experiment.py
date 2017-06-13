#copy of discrimante3.py

import os

# the default is 64gb we need to use more memory
# undocumented feature
# we found this by searching in the code for a memory limit
os.environ['TF_CUDA_HOST_MEM_LIMIT_IN_MB'] = '200000'

import tensorflow as tf
import logging

from utils import (Model, get_device_list, static_constant_multivolume, random_sample,
    random_occlusion, equal_to_centre, image_summary, static_shape, compose,
    slices_to_shape, shape_to_slices, EMA)

from loss_functions import localized_errors, upsample_mean
from dataset import MultiDataset
import dataset_path
import augment


from error_detection_estimator import (ErrorDetectionEstimator, patch_size_suggestions,
    range_expanders)

class ErrorDetectionExperiment(Model):
    def __init__(self, patch_size, dataset, devices, train_vols, test_vols, name=None):
        """
        Error detection is composed of a large network which goal is to reconstruct an object
        which has being partially ocluded. (segment_completion loss)
        The deep supervision loss which is to tell if there is an error in the object is what
        we actually care about.
        
        Args:
            patch_size (tuple(int)): field of view of the network
            dataset (Dataset): Object containing all the data
            devices (str): Were to train on
            train_vols (list(int)): This should be part of dataset
            test_vols (list(int)): This should be part of dataset
            name (None, optional): Experiment Description
        """
        
        # put this inside dataset
        self.train_vols = train_vols
        self.test_vols = test_vols

        self._patch_size = patch_size
        self.padded_patch_size = (1,) + patch_size + (1,)
        self._dataset = dataset
        self._devices = devices
        self._name = name

        self.summaries = []

        self._create_session()
        self._read_dataset(dataset)
        self._create_forward_network()
        self._data_parallelism()
        self._create_optimizer()


    def _create_session(self):
        config = tf.ConfigProto(
            allow_soft_placement=True,
            #gpu_options=tf.GPUOptions(
            #  per_process_gpu_memory_fraction=0.9,
            #  allow_growth=True),
            #log_device_placement=True,
        )
        self.sess = tf.Session(config=config)
        self.run_metadata = tf.RunMetadata()

    def _read_dataset(self, dataset):
        with tf.name_scope('dataset'):
            with tf.device("/cpu:0"):
                n_volumes = len(dataset.image)
                self._human_labels = static_constant_multivolume(
                    self.sess, dataset.human_labels, self.padded_patch_size)
                self._machine_labels = static_constant_multivolume(
                    self.sess, dataset.machine_labels, self.padded_patch_size)
                self._image = static_constant_multivolume(
                    self.sess, dataset.image, self.padded_patch_size)
                self._samples = static_constant_multivolume(
                    self.sess, dataset.samples, (1,3), indexing='CORNER')
            logging.info("finished loading data")

    def _create_forward_network(self):
        self.is_train_iter = tf.placeholder(shape=[], dtype=tf.bool)
        with tf.name_scope('estimator'):
            self.step = tf.Variable(0, trainable=False, name="step")
            self._estimator = ErrorDetectionEstimator(self._patch_size, 2, 1)

    def _data_parallelism(self):
        """
        Data parallelism
        Every gpu trains on different data and the gradients 
        are combined
        """
        with tf.name_scope('data_parallelism'):
            loss=0
            segment_completion_loss=0
            for i,d in enumerate(self._devices):
                with tf.name_scope(d.replace('/','_').replace(':','')):
                    tower_loss, tower_segment_completion_loss = self._create_tower(d)
                    loss += tower_loss
                    segment_completion_loss += tower_segment_completion_loss

            self._loss = loss/len(self._devices)
            self._segment_completion_loss = segment_completion_loss/len(self._devices)

    def _create_tower(self, d):
        """Summary
        
        Args:
            d (str): device identifier
        """
        with tf.device(d):
            (single_object_human_labels_glimpse, single_object_machine_labels_glimpse,
             occluded_glimpse, image_glimpse,
             human_labels_glimpse) = self._create_glimpse()

        with tf.device("/cpu:0"): 
            # We do this in the cpu because boolean operations are faster
            # for some unkown reason, reduce should be fast in the gpu in theory
            any_error = self._is_there_any_error_in_the_glimpse(
                single_object_human_labels_glimpse, 
                single_object_machine_labels_glimpse)

        with tf.device(d):
            # just use an identity to copy this to the gpu
            # if you don't do this, this gets copied every time
            gpu_any_error = tf.identity(any_error)
            segment_completion_loss = self._create_segment_completion_task(image_glimpse, 
                occluded_glimpse, single_object_human_labels_glimpse)

            (error_prediction_loss, machine_labels_error_prediction,
            human_labels_error_prediction) = self._create_error_detection_task(image_glimpse, 
                single_object_machine_labels_glimpse, single_object_human_labels_glimpse,
                gpu_any_error)


        with tf.device("/cpu:0"):
            single_object_machine_labels_glimpse_cpu = tf.identity(single_object_machine_labels_glimpse)
            human_labels_glimpse_cpu = tf.identity(human_labels_glimpse)

            error_prediction_loss += self._create_error_localization_task(machine_labels_error_prediction,
                single_object_machine_labels_glimpse_cpu, human_labels_glimpse_cpu, any_error, 
                human_labels_error_prediction)

        
        return error_prediction_loss, segment_completion_loss

    def _create_glimpse(self):
        """
        This takes a random crop from a random volume.
        This random crop that we refer as a glimpse is randomly
        modified and return.

        More specifically five glimpses are return:
        `image_glimpse`: F eletron microscopy image.
        `single_object_machine_labels_glimpse`: randomly cropped data augmented boolean
           mask that contains 1s in the segment contained by the center of the volume 
           and 0s everywhere else. This was created by running mean affinity 
           agglomeration on top of watershed.
        `human_labels_glimpse`:random cropped data augment uint32 segmentation 
            produced by manual annotation.
        `single_object_human_labels_glimpse`: same as human_labels_glimpse but 
            transformed into a binary mask which contains 1s for the segment in the 
            center of the glimpse and 0s everywhere else.
        `occluded_glimpse`: It randomly blacks out halfs of the 
            single_object_human_labels_glimpse. This is used to train an Estimator to 
            go complete the blacked out part.
        """
        vol_id = self._get_volume_id()
        focus = self._get_random_focus(vol_id)
        rr = augment.RandomRotationPadded()

        image_glimpse = rr(self._image[vol_id,focus])
        single_object_machine_labels_glimpse = rr(equal_to_centre(self._machine_labels[vol_id, focus]))

        human_labels = self._human_labels[vol_id,focus]
        single_object_human_labels_glimpse = rr(equal_to_centre(human_labels))
        human_labels_glimpse = rr(human_labels)

        occluded_glimpse = random_occlusion(single_object_human_labels_glimpse)
        
        self.summaries.append(image_summary("single_object_machine_labels_glimpse", single_object_machine_labels_glimpse))
        self.summaries.append(image_summary("single_object_human_labels_glimpse", single_object_human_labels_glimpse))
        self.summaries.append(image_summary("human_labels_glimpse", tf.to_float(human_labels_glimpse)))
        self.summaries.append(image_summary("occluded", occluded_glimpse))

        return (single_object_human_labels_glimpse, single_object_machine_labels_glimpse, 
                occluded_glimpse, image_glimpse, human_labels_glimpse)

    def _get_volume_id(self):
        """
        Get a random volume id conditioned if we are on a trainning iteration
        or a test one.
        """
        return tf.cond(self.is_train_iter,
                lambda: random_sample(tf.constant(self.train_vols)),
                lambda: random_sample(tf.constant(self.test_vols)),
            )
    def _get_random_focus(self, vol_id):
        """given a volume id, it gets a random sample from the list of points to sample
           for that given volume."""

        with tf.name_scope("random_focus"):
            sample = self._samples[vol_id,('RAND',0)]
            reshaped_sample = tf.reshape(sample,(3,))
            focus = tf.concat([[0], reshaped_sample ,[0]],0) # vector of the from [0 x y z 0]

            #for debugging
            focus = tf.Print(focus,[vol_id, focus], message="focus", summarize=10)
            return focus


    def _is_there_any_error_in_the_glimpse(self, object_0, object_1):
        # the single_object_human_labels_glimpse were produced by merging
        # and splitting supervoxels.
        # the single_object_machine_labels_glimpse were produced by 
        # applying mean affinity agglomeration to the same supervoxels
        # So any_error is a tf.bool which equals True when the single
        # object of the human_labels is composed by a different set 
        # of supervoxels than the human labels.
        any_error = tf.logical_not(tf.reduce_all(
            tf.equal(object_0, object_1)))

        # don't propagate any gradients to the data provider
        # our dataset is stored as a tensorflow variable
        # so we wan't to make sure we don't modified it
        # in theory dataset indexing already stops the gradient but 
        # we can add this to be more sure (we are lazy to double check
        # it is actually working)
        any_error = tf.stop_gradient(any_error)
        return any_error

    def _create_segment_completion_task(self, image_glimpse , occluded_glimpse,
        single_object_human_labels_glimpse):
        """
        Notice how the input to both task is first a binary mask 
        and secondly an EM image
        """
        
        with tf.name_scope("segment_completion_task"):
            segment_completion = self._estimator.segment_completion(tf.concat(
                [occluded_glimpse, image_glimpse],4))
            segment_completion_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=segment_completion, labels=single_object_human_labels_glimpse))
            self.summaries.append(
                image_summary("segment_completion", tf.nn.sigmoid(segment_completion)))

        return segment_completion_loss

    def _create_error_detection_task(self, image_glimpse, 
        single_object_machine_labels_glimpse, single_object_human_labels_glimpse,
        gpu_any_error):

        with tf.name_scope("error_detection_task"):
            human_labels_error_prediction = self._estimator.error_prediction(
                    tf.concat([single_object_human_labels_glimpse, image_glimpse],4))
                
            # We save computation by only running error_prediction on the machine labels
            # if there is an error on them.
            machine_labels_error_prediction = tf.cond(gpu_any_error,
                    lambda:  self._estimator.error_prediction(
                        tf.concat([single_object_machine_labels_glimpse, image_glimpse],4)),
                    lambda: map(tf.identity, human_labels_error_prediction),
                    name="machine_labels_error_prediction")

            machine_labels_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.reduce_sum(machine_labels_error_prediction[-1]),
                labels=tf.to_float(gpu_any_error),
                name="machine_labels_loss")

            human_labels_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=tf.reduce_sum(human_labels_error_prediction[-1]),
                labels=tf.constant(0, dtype=tf.float32),
                name="human_labels_loss")

            return (machine_labels_loss + human_labels_loss,
             machine_labels_error_prediction, human_labels_error_prediction)

    def _create_error_localization_task(self, machine_labels_error_prediction,
        single_object_machine_labels_glimpse, human_labels_glimpse_cpu, any_error,
        human_labels_error_prediction):

        error_localization_loss = 0
        for i in range(4,6):
            ds_shape = static_shape(machine_labels_error_prediction[i])
            expander = compose(*reversed(range_expanders[0:i]))

            assert tuple(slices_to_shape(expander(shape_to_slices(ds_shape[1:4])))) == tuple(self._patch_size)
            def get_localized_errors():
                print(ds_shape)
                x = localized_errors(
                    single_object_machine_labels_glimpse, human_labels_glimpse_cpu, 
                    ds_shape = ds_shape, expander=expander)
                return tf.Print(x,[any_error],message="any error")

            errors = tf.cond(
                    any_error,
                    lambda: get_localized_errors(),
                    lambda: tf.zeros(ds_shape))

            error_localization_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=machine_labels_error_prediction[i], labels=errors))
            error_localization_loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=human_labels_error_prediction[i], 
                labels=tf.zeros_like(human_labels_error_prediction[i])))

            self.summaries.append(
                image_summary("error_localization_prediction_level_"+str(i), 
                    upsample_mean(tf.nn.sigmoid(machine_labels_error_prediction[i]),
                                  self.padded_patch_size, expander), zero_one=True))
            self.summaries.append(
                image_summary("error_localization_desired_level_"+str(i), 
                    upsample_mean(errors, self.padded_patch_size, expander)))

        return error_localization_loss

    def _train_op(self):
        optimizer = tf.train.AdamOptimizer(0.0001, beta1=0.95, beta2=0.9995, epsilon=0.1)
        op = optimizer.minimize(8e5*self._loss + self._segment_completion_loss, 
            colocate_gradients_with_ops=True)

        ema_loss = EMA(decay=0.99)
        ema_loss.update(self._loss)

        ema_segment_completion_loss = EMA(decay=0.99)
        ema_segment_completion_loss.update(self._segment_completion_loss)

        with tf.control_dependencies([op]):
            with tf.control_dependencies([self.step.assign_add(1)]):
                op = tf.group(
                        tf.Print(0, [tf.identity(self.step), self._loss], message="step|loss"),
                        )
        quick_summary_op = tf.summary.merge([
            tf.summary.scalar("loss", self._loss),
            tf.summary.scalar("segment_completion_loss", self._segment_completion_loss),
            tf.summary.scalar("ema_segment_completion_loss", ema_segment_completion_loss.val),
            tf.summary.scalar("ema_loss", ema_loss.val)])

        return op, quick_summary_op

    def _test_op(self):
        ema_test_loss = EMA(decay=0.9)
        ema_test_loss.update(self._loss)

        ema_test_segment_completion_loss = EMA(decay=0.9)
        ema_test_segment_completion_loss.update(self._segment_completion_loss)
        quick_summary_op = tf.summary.merge([
            tf.summary.scalar("test_loss", self._loss),
            tf.summary.scalar("test_segment_completion_loss", self._segment_completion_loss),
            tf.summary.scalar("ema_test_segment_completion_loss", ema_test_segment_completion_loss.val),
            tf.summary.scalar("ema_test_loss", ema_test_loss.val)])
        return tf.no_op(), quick_summary_op

    def _create_optimizer(self):
        """
        The gradients created by the optimizer will be computed in the same
        place the forward pass is done
        """
        self.iter_op, self.quick_summary_op = tf.cond(self.is_train_iter,
                self._train_op, self._test_op)

        init = self.get_uninitialized_variables()
        self.sess.run(init)

        self.summary_op = tf.summary.merge(self.summaries)
        self.saver = tf.train.Saver(
            var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES),
            keep_checkpoint_every_n_hours=2)

    def get_filename(self):
        return os.path.splitext(os.path.basename(__file__))[0]

    def get_uninitialized_variables(self):
        """Get uninitialized variables as a list.
        """
        variables = tf.global_variables()
        init_flag = self.sess.run([tf.is_variable_initialized(v) for v in variables])
        return [v.initializer for v, f in zip(variables, init_flag) if not f]
    
if __name__ == '__main__':
    TRAIN = MultiDataset(
        [   
            dataset_path.get_path("1_1_1"),
            # dataset_path.get_path("1_2_1"),
            # dataset_path.get_path("1_3_1"),
            # dataset_path.get_path("2_1_1"),
            # dataset_path.get_path("2_2_1"),
            # dataset_path.get_path("2_3_1"),
            # dataset_path.get_path("3_1_1"),
        ],
        {
            "machine_labels": "lzf_mean_agg_tr.h5",
            "human_labels": "lzf_proofread.h5",
            "image": "image.h5",
            "samples": "padded_valid_samples.h5",
        }
    )
    args = {
    "devices": ['/cpu:0'], #get_device_list(),
    "patch_size": tuple(patch_size_suggestions([2,3,3])[0]),
    "name": "test",
    "dataset": TRAIN,
    # "train_vols": [0,1,2,3,4,5],
    "train_vols": [0],
    "test_vols": [0]
    # "test_vols": [6]

    }
    ErrorDetectionExperiment(**args).train(
        nsteps=1000000, checkpoint_interval=3000, test_interval=15)
#!/usr/bin/env python2

from datetime import datetime
import logging
import time
import os.path

from yacn.nets import enviroment #has to happen before tf is imported(maybe)

import tensorflow as tf
from tensorflow.python.client import timeline
from tqdm import tqdm


class Experiment(object):
    """
    Provides inputs and desired labels to Estimators
    """
    pass


class Estimator(object):

    def fit(train_input_fn):
        pass

    def evaluate(eval_input_fn):
        pass

class Runner(object):
    """
    Runs experiments

    TODO the runner should hold the session and not the experiment
    """
    def __init__(self, experiment, logdir=None):
        
        self._experiment = experiment

        if logdir:
            self._restore(logdir)
        else:
            self._initialize()
    
    def _initialize(self):
        date = datetime.now().strftime("%j-%H-%M-%S")
        self._logdir = os.path.expanduser("~/experiments/{}_{}/".format(
            date, type(self._experiment).__name__))

        logging.info('logging to {}'.format(self._logdir))
        if not os.path.exists(self._logdir):
            os.makedirs(self._logdir)

        self._summary_writer = tf.summary.FileWriter(
            self._logdir, graph=self._experiment.sess.graph)

    def _restore(self, logdir):
        logging.info('Restoring checkpoint')
        modelpath = os.path.realpath(os.path.expanduser(logdir))
        self._experiment.saver.restore(self._experiment.sess, modelpath)

    def train(self, n_steps = 100000,
        quick_summary_train_interval = 5,
        quick_summary_test_interval = 10,
        long_summary_train_interval = 50,
        long_summary_test_interval = 50,
        profile_interval = 500,
        checkpoint_interval = 1000):
        """
        If we restard training i won't be the less than
        self._experiment.step

        step is only advance on train operations
        so we do first test and then train otherwise we would
        write two summaries with the same step which is not allowed
        """
        last_step = 0
        i = 0
        progressbar = tqdm(total=n_steps)
        while last_step < n_steps:
            i += 1
            if i % quick_summary_test_interval == 0:
                step , _ , summary_proto = self._experiment.sess.run(
                    [self._experiment.step, 
                     self._experiment.iter_op,
                     self._experiment.quick_summary_op],
                    feed_dict={self._experiment.is_train_iter: False})
                self._summary_writer.add_summary(summary_proto, step)

            if i % quick_summary_train_interval == 0:
                step , _ , summary_proto = self._experiment.sess.run(
                    [self._experiment.step, 
                     self._experiment.iter_op,
                     self._experiment.quick_summary_op],
                    feed_dict={self._experiment.is_train_iter: True})
                self._summary_writer.add_summary(summary_proto, step)

            if i % long_summary_test_interval == 0:
                step , _ , summary_proto = self._experiment.sess.run(
                    [self._experiment.step, 
                     self._experiment.iter_op,
                     self._experiment.long_summary_op],
                    feed_dict={self._experiment.is_train_iter: False})
                self._summary_writer.add_summary(summary_proto, step)

            if i % long_summary_train_interval == 0:
                step , _ , summary_proto = self._experiment.sess.run(
                    [self._experiment.step, 
                     self._experiment.iter_op,
                     self._experiment.long_summary_op],
                    feed_dict={self._experiment.is_train_iter: True})
                self._summary_writer.add_summary(summary_proto, step)

            if i % profile_interval == 0 and False: #seems to be too large of a message
                step , _  = self._experiment.sess.run(
                    [self._experiment.step, 
                     self._experiment.iter_op],
                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), 
                     run_metadata=self._experiment.run_metadata, 
                     feed_dict={self._experiment.is_train_iter: True})
                trace = timeline.Timeline(step_stats=self._experiment.run_metadata.step_stats)
                filepath = '{}/timeline-step-{}.ctf.json'.format(self._logdir, step)
                with  open(filepath, 'w') as f:
                    f.write(trace.generate_chrome_trace_format(show_memory=True, show_dataflow=True))
                self._summary_writer.add_run_metadata(self._experiment.run_metadata, str(step))

            if i % checkpoint_interval == 0:
                logging.info('saving checkpoint')
                step = self._experiment.sess.run(self._experiment.step)
                filename = '{}/step-{}.ckpt'.format(self._logdir, step)
                self._experiment.saver.save(
                    self._experiment.sess,
                    filename,
                    write_meta_graph=False)
                self._summary_writer.flush()

            # train
            step, _ = self._experiment.sess.run(
                [self._experiment.step, self._experiment.iter_op],
                 feed_dict={self._experiment.is_train_iter: True})

            progressbar.update(n=step-last_step)
            last_step = step



if __name__ == '__main__':
    from error_detection_experiment import ErrorDetectionExperiment
    from dataset import MultiDataset
    import dataset_path
    from error_detection_estimator import patch_size_suggestions

    patch_size = tuple(patch_size_suggestions([2,3,3])[0])
    TRAIN = MultiDataset(
        directories=[   
            dataset_path.get_path("1_1_1"), #0
            dataset_path.get_path("1_2_1"), #1 
            dataset_path.get_path("1_3_1"), #2
            dataset_path.get_path("2_1_1"), #3
            dataset_path.get_path("2_2_1"), #4
            dataset_path.get_path("2_3_1"), #5
            dataset_path.get_path("3_1_1"), #6
        ],
        types={
            "machine_labels": "lzf_mean_agg_tr.h5",
            "human_labels": "lzf_proofread.h5",
            "image": "image.h5",
            "samples": "padded_valid_samples.h5",
        },
        train_vols=[0,1,2,3,4,5],
        test_vols=[6],
        patch_size=patch_size)

    args = {
    "devices": enviroment.get_device_list(), #['/gpu:0'], #,
    "patch_size": patch_size,
    "dataset": TRAIN,
    }
    ex = ErrorDetectionExperiment(**args)
    Runner(ex, logdir='/usr/people/it2/experiments/188-10-16-08_ErrorDetectionExperiment/step-9775.ckpt.index').train()

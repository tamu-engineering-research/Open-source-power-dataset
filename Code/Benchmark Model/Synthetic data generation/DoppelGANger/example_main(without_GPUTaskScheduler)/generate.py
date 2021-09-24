import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import sys
sys.path.append("..")

from gan import output
sys.modules["output"] = output

from gan.doppelganger import DoppelGANger
from gan.util import add_gen_flag, normalize_per_sample, renormalize_per_sample
from gan.load_data import load_data
from gan.network import DoppelGANgerGenerator, Discriminator, \
    RNNInitialStateType, AttrDiscriminator

import os
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import pickle

from scipy.interpolate import InterpolatedUnivariateSpline


def min_max_scale(data):
    min_val = np.min(np.min(data, axis = 0), axis = 0)
    data = data - min_val
    max_val = np.max(np.max(data, axis = 0), axis = 0)
    norm_data = data / (max_val + 1e-7)
    return norm_data, min_val, max_val

if __name__ == "__main__":
    sample_len = 10
    num_channels = 91
    
    data_npz = np.load("../../data/real_train.npz")
    data_feature, min_val, max_val = min_max_scale(data_npz["trans"])
    data_attribute = data_npz["y"]
    data_gen_flag = np.ones((data_feature.shape[0], data_feature.shape[1]))
    
    data_feature_outputs = [ # 91 channels
        output.Output(output.OutputType.CONTINUOUS, 1, output.Normalization.ZERO_ONE, is_gen_flag=False)
        for _ in range(num_channels)]
    
    data_attribute_outputs = [ # 1 categorical feature with 5 possibilities
        output.Output(output.OutputType.DISCRETE, 5, None, is_gen_flag=False),]
    
    num_real_attribute = len(data_attribute_outputs)


    (data_feature, data_attribute, data_attribute_outputs,
     real_attribute_mask) = \
        normalize_per_sample(
            data_feature, data_attribute, data_feature_outputs,
            data_attribute_outputs)
    
    data_feature, data_feature_outputs = add_gen_flag(
        data_feature, data_gen_flag, data_feature_outputs, sample_len)

    generator = DoppelGANgerGenerator(
        feed_back=False,
        noise=True,
        feature_outputs=data_feature_outputs,
        attribute_outputs=data_attribute_outputs,
        real_attribute_mask=real_attribute_mask,
        sample_len=sample_len,
        initial_state=RNNInitialStateType.RANDOM,
        feature_num_units=256,
        feature_num_layers=2)
    discriminator = Discriminator()
    attr_discriminator = AttrDiscriminator()
    
    checkpoint_dir = os.path.join("./log/checkpoint")
    sample_dir = os.path.join("./log/sample")
    time_path = os.path.join("./log/time.txt")
    
    epoch = 400
    batch_size = 100
    vis_freq = 200
    vis_num_sample = 5
    d_rounds = 1
    g_rounds = 1
    d_gp_coe = 10.0
    attr_d_gp_coe = 10.0
    g_attr_d_coe = 1.0
    extra_checkpoint_freq = 5
    num_packing = 1
    
    run_config = tf.ConfigProto()
    with tf.Session(config=run_config) as sess:
        gan = DoppelGANger(
            sess=sess,
            checkpoint_dir=checkpoint_dir,
            sample_dir=sample_dir,
            time_path=time_path,
            epoch=epoch,
            batch_size=batch_size,
            data_feature=data_feature,
            data_attribute=data_attribute,
            real_attribute_mask=real_attribute_mask,
            data_gen_flag=data_gen_flag,
            sample_len=sample_len,
            data_feature_outputs=data_feature_outputs,
            data_attribute_outputs=data_attribute_outputs,
            vis_freq=vis_freq,
            vis_num_sample=vis_num_sample,
            generator=generator,
            discriminator=discriminator,
            attr_discriminator=attr_discriminator,
            d_gp_coe=d_gp_coe,
            attr_d_gp_coe=attr_d_gp_coe,
            g_attr_d_coe=g_attr_d_coe,
            d_rounds=d_rounds,
            g_rounds=g_rounds,
            extra_checkpoint_freq=extra_checkpoint_freq,
            num_packing=num_packing)
        gan.build()
        print("Finished building")
        
        print("Generating...")
        generate_num_train_sample = 400
        generate_num_test_sample = 400
        total_generate_num_sample = generate_num_train_sample + generate_num_test_sample

        if data_feature.shape[1] % sample_len != 0:
            raise Exception("length must be a multiple of sample_len")
        length = int(data_feature.shape[1] / sample_len)
        real_attribute_input_noise = gan.gen_attribute_input_noise(
            total_generate_num_sample)
        addi_attribute_input_noise = gan.gen_attribute_input_noise(
            total_generate_num_sample)
        feature_input_noise = gan.gen_feature_input_noise(
            total_generate_num_sample, length)
        input_data = gan.gen_feature_input_data_free(
            total_generate_num_sample)

        for epoch_id in range(extra_checkpoint_freq - 1,
                              epoch,
                              extra_checkpoint_freq):
            if epoch_id != 399: continue
            print("Processing epoch_id: {}".format(epoch_id))
            mid_checkpoint_dir = os.path.join(
                checkpoint_dir, "epoch_id-{}".format(epoch_id))
            if not os.path.exists(mid_checkpoint_dir):
                print("Not found {}".format(mid_checkpoint_dir))
                continue

            save_path = './log/generated_samples/epoch_id-{}'.format(epoch_id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            train_path_ori = os.path.join(
                save_path, "generated_data_train_ori.npz")
            test_path_ori = os.path.join(
                save_path, "generated_data_test_ori.npz")
            train_path = os.path.join(
                save_path, "generated_data_train.npz")
            test_path = os.path.join(
                save_path, "generated_data_test.npz")
            if os.path.exists(test_path):
                print("Save_path {} exists".format(save_path))
                continue

            gan.load(mid_checkpoint_dir)

            print("Finished loading")
            split = generate_num_train_sample

            features, attributes, gen_flags, lengths = gan.sample_from(
                real_attribute_input_noise, addi_attribute_input_noise,
                feature_input_noise, input_data)

            features, attributes = renormalize_per_sample(
                features, attributes, data_feature_outputs,
                data_attribute_outputs, gen_flags,
                num_real_attribute=num_real_attribute)
            
            features = features * max_val + min_val
            np.savez(
                "generated_data_train.npz",
                data_feature=features[0: split],
                data_attribute=attributes[0: split],
                data_gen_flag=gen_flags[0: split])
            np.savez(
                "generated_data_test.npz",
                data_feature=features[split:],
                data_attribute=attributes[split:],
                data_gen_flag=gen_flags[split:])
            
            print("Done")
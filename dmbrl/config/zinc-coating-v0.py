from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import sys

import numpy as np
import tensorflow as tf
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC

from zinc_coating import zinc_coating_environment

"""
    Module name,
    MODEL_IN, MODEL_OUT,
    import env, env_name
"""


class OwnZincCoatingConfigModule:
    ENV_NAME = "MBRLGYM_zinc_coating_po-v0"
    TASK_HORIZON = 2500
    NTRAIN_ITERS = 100
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 11
    INIT_VAR = 0.25
    MODEL_IN, MODEL_OUT = 14, 13  # obs -> 13, action -> 1
    GP_NINDUCING_POINTS = 300

    def __init__(self):
        # self.ENV = gym.make(self.ENV_NAME)
        self.ENV = zinc_coating_environment.ZincCoatingV0()
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)
        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "GBPRandom": {
                "popsize": 2500
            },
            "GBPCEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POPLIN-P": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            },
            "POPLIN-A": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        """ @brief: no cheating of the observation function
        """
        if isinstance(obs, np.ndarray):
            return obs
        else:
            return obs

    @staticmethod
    def obs_postproc(obs, pred):
        if isinstance(obs, np.ndarray):
            return obs + pred
        else:
            return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        """ @brief:
                see mbbl.env.gym_env.walker.py for reward details

            def reward(data_dict):
                action = data_dict['action']
                true_action = action * self._env.env.max_torque

                max_torque = self._env.env.max_torque
                torque = np.clip(true_action, -max_torque, max_torque)[0]

                y, x, thetadot = data_dict['start_state']

                costs = y + .1 * x + .1 * (thetadot ** 2) + .001 * (torque ** 2)
                # note: reward is the negative cost
                return -costs
        """

        diff = obs[:, 3] * 220 - 50
        # clipped = tf.clip_by_value(diff, -1, 21)
        scaled = (diff - 8) / 6
        reward = -tf.math.exp(-scaled) - scaled + 2
        return -reward

    @staticmethod
    def ac_cost_fn(acs):
        if isinstance(acs, np.ndarray):
            return 0
        else:
            return tf.constant(0, dtype="float32")
            # return tf.reduce_sum(tf.multiply(acs, 0))

    def nn_constructor(self, model_init_cfg, misc=None):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model", num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS, load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None),
            misc=misc
        ))
        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(200, activation="swish", weight_decay=0.00005))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))
        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.0005})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = OwnZincCoatingConfigModule

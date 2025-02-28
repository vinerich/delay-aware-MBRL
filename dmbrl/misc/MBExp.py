from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import localtime, strftime

from scipy.io import savemat
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent
from dmbrl.misc import logger
import copy
import numpy as np
import pandas as pd

class MBExperiment:

    def __init__(self, params):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """
        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        self._params = params
        params.sim_cfg.misc = copy.copy(params)
        if params.sim_cfg.get("stochastic", False):
            self.agent = Agent(DotMap(
                env=self.env, noisy_actions=True,
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                ),
                params=params
            ))
        else:
            self.agent = Agent(DotMap(env=self.env, noisy_actions=False, params=params))

        self.plan_hor = get_required_argument(
            params.exp_cfg, "plan_hor", "Must provide planning horizon."
        )
        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)
        self.delay_hor = params.sim_cfg.get("delay_hor", 0)

        self.logdir = os.path.join(get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."), "dats-delay-" + str(self.delay_hor))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        self.logdir = self.logdir + "/zinc-coating-v0_" + str(len(os.listdir(self.logdir)))
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

        logger.set_file_handler(path=self.logdir)
        logger.info('Starting the experiments')

        print(f"PLAN_HORIZON {self.plan_hor}")
        self.policy.set_env(self.env)
        self.policy.logger = logger
        # self.eval_callback = MyEvalCallback(self.env, log_path=self.logdir + "/evaluations.npz")
        # self.eval_callback.init_callback(self.policy)

    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []
        test_traj_obs, test_traj_acs, test_traj_rets = [], [], []
        episode_iter_id = []

        np_stats_timesteps, np_stats_results, np_stats_results_lows, np_stats_results_highs, np_stats_ep_lengths, np_stats_train_time = [], [], [], [], [], []

        # Perform initial rollouts
        samples = []
        needed_num_steps = self.ninit_rollouts * self.task_hor
        finished_num_steps = 0
        """
        # TODO DEBUG
        needed_num_steps = 64
        self.task_hor = 64
        """
        while True:
            samples.append(
                self.agent.sample(
                    self.task_hor, self.policy, self.delay_hor
                )
            )
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])
            finished_num_steps += len(samples[-1]["ac"])
            print(finished_num_steps)

            if finished_num_steps >= needed_num_steps:
                break

        if self.ninit_rollouts > 0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples]
            )

        # self.eval_callback.on_training_start(None, None)

        self.steps = 0
        # Training loop
        for i in range(self.ntrain_iters):

            logger.info("####################################################################")
            logger.info("Starting training iteration %d." % (i + 1))
            t0 = pd.Timestamp.utcnow()

            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            samples = []
            assert self.nrecord == 0

            needed_num_steps = self.task_hor * \
                (max(self.neval, self.nrollouts_per_iter) - self.nrecord)
            finished_num_steps = 0
            # self.eval_callback.on_rollout_start()
            while True:
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy, self.delay_hor
                    )
                )
                finished_num_steps += len(samples[-1]["ac"])

                # setattr(self.policy, "num_timesteps", self.steps + finished_num_steps)
                # self.eval_callback.on_step()
                
                if finished_num_steps >= needed_num_steps:
                    self.steps += finished_num_steps
                    # self.eval_callback.on_rollout_end()
                    break

                # self.policy.set_num_timesteps(finished_num_steps)

            # test the policy if needed
            if self._params.misc.ctrl_cfg.cem_cfg.test_policy > 0:
                test_data = []
                for _ in range(5):
                    test_data.append(
                        self.agent.sample(self.task_hor, self.policy, self.delay_hor,
                                          test_policy=True, average=False)
                    )
                test_traj_rets.extend([
                    np.mean([i_test_data["reward_sum"] for i_test_data in test_data])
                ])
                test_traj_obs.extend(
                    [i_test_data["obs"] for i_test_data in test_data]
                )
                test_traj_acs.extend(
                    [i_test_data["ac"] for i_test_data in test_data]
                )

            traj_obs.extend([sample["obs"] for sample in samples])
            traj_acs.extend([sample["ac"] for sample in samples])
            traj_rets.extend([sample["reward_sum"] for sample in samples])
            traj_rews.extend([sample["rewards"] for sample in samples])
            episode_iter_id.extend([i] * len(samples))
            samples = samples[:self.nrollouts_per_iter]

            self.policy.dump_logs(self.logdir, iter_dir)
            savemat(
                os.path.join(self.logdir, "logs.mat"),
                {
                    "observations": traj_obs,
                    "actions": traj_acs,
                    "returns": traj_rets,
                    "rewards": traj_rews,
                    "test_returns": test_traj_rets,
                    "test_obs": test_traj_obs,
                    "test_acs": test_traj_acs,
                    'episode_iter_id': episode_iter_id
                }
            )

            np_stats_timesteps.append(self.steps)
            current_rewards = [sample["rewards"] for sample in samples[:self.neval]][0]
            np_stats_ep_lengths.append(len(current_rewards))
            np_stats_results.append([sum(current_rewards)])
            np_stats_results_lows.append([sum(map(lambda x: 1 if x == -2 else 0, current_rewards))])
            np_stats_results_highs.append([sum(map(lambda x: 1 if x == -1 else 0, current_rewards))])
            if len(np_stats_train_time):
                np_stats_train_time.append(np_stats_train_time[-1] + (pd.Timestamp.utcnow() - t0).total_seconds())
            else:
                np_stats_train_time.append((pd.Timestamp.utcnow() - t0).total_seconds())

            logger.info("Rewards obtained: {}, Lows: {}, Highs: {}, Total time: {}".format(np_stats_results[-1], np_stats_results_lows[-1], np_stats_results_highs[-1], np_stats_train_time[-1]))

            np.savez(
                self.logdir + "/evaluations.npz",
                timesteps=np_stats_timesteps,
                results=np_stats_results,
                results_lows=np_stats_results_lows,
                results_highs=np_stats_results_highs,
                ep_lengths=np_stats_ep_lengths,
                train_time=np_stats_train_time
            )

            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)

            if i < self.ntrain_iters - 1:
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples]
                )

                # TODO: train the policy network

        # self.eval_callback.on_training_end()

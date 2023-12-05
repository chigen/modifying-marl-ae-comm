from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch
import time, os, sys
from collections import deque
from loss import policy_gradient_loss
from torch.utils.tensorboard import SummaryWriter
import csv
import datetime


def set_requires_grad(modules, value):
    for m in modules:
        for p in m.parameters():
            p.requires_grad = value


class Master(object):
    """ A master network. Think of it as a container that holds weight and the
    optimizer for the workers

    Args
        net: a neural network A3C model
        opt: shared optimizer
        gpu_id: gpu device id
    """

    def __init__(self, net, opt, global_iter, global_done, master_lock,
                 writer_dir, max_iteration=100):
        self.lock = master_lock
        self.iter = global_iter
        self.done = global_done
        self.max_iteration = max_iteration
        self.net = net
        self.opt = opt
        self.net.share_memory()
        self.writer_dir = writer_dir

    def init_tensorboard(self):
        """ initializes tensorboard by the first worker """
        with self.lock:
            if not hasattr(self, 'writer'):
                self.writer = SummaryWriter(self.writer_dir)
        return

    def copy_weights(self, net, with_lock=False):
        """ copy weight from master """

        if with_lock:
            with self.lock:
                for p, mp in zip(net.parameters(), self.net.parameters()):
                    p.data.copy_(mp.data)
            return self.iter.value
        else:
            for p, mp in zip(net.parameters(), self.net.parameters()):
                p.data.copy_(mp.data)
            return self.iter.value

    def _apply_gradients(self, net):
        # backward prop and clip gradients
        self.opt.zero_grad()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 40.0)

        for p, mp in zip(net.parameters(), self.net.parameters()):
            if p.grad is not None:
                mp.grad = p.grad.cpu()
        self.opt.step()

    def apply_gradients(self, net, with_lock=False):
        """ apply gradient to the master network """
        if with_lock:
            with self.lock:
                self._apply_gradients(net)
        else:
            self._apply_gradients(net)
        return
    
    def record_thermodynamic_charts(self, thermodynamic_charts):
        with self.iter.get_lock():
            # the thermodynamic_charts is a list of np array, its format is:
            # [eval_id][agent_id][np.array]
            # create different files for each eval_id and agent_id

            if self.iter.value % 100 == 0:
            
                for eval_id in range(len(thermodynamic_charts)):
                    for agent_id in range(len(thermodynamic_charts[eval_id])):
                        # the csv file name is like: time_eval_id_agent_id_thermo.csv
                        csv_path = datetime.datetime.now().strftime("%m%d-%H%M%S") + "_eval_" + str(eval_id) + "_agent_" + str(agent_id) + "_thermo.csv"
                        with open(csv_path, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow(thermodynamic_charts[eval_id][agent_id])


    def increment(self, progress_str=None,
                  comm_encode=None, comm_out=None, comm_decoded=None, 
                  traj_comm_encode=None, traj_comm_out=None, traj_comm_decoded=None):
        with self.iter.get_lock():
            self.iter.value += 1
            # curr_time = str(datetime.datetime.now())[5:16].replace(' ', '_')
            csv_path = "0729-1727" + "_no_traj_prepro_purple_samerew_om1_agent_"

            # original: %100
            if self.iter.value % 10 == 0:

                if progress_str is not None:
                    print('[{}/{}] {}'.format(
                        self.iter.value, self.max_iteration, progress_str))

                else:
                    print('[{}/{}] workers are working hard.'.format(
                        self.iter.value, self.max_iteration))
                    
                # if comm_out is not None:
                    # comm_encode = comm_encode.detach().cpu().numpy().tolist()
                    # comm_out = comm_out.detach().cpu().numpy().tolist()
                    # comm_decoded = comm_decoded.detach().cpu().numpy().tolist()
                    # traj_comm_encode = traj_comm_encode.detach().cpu().numpy().tolist()
                    # traj_comm_out = traj_comm_out.detach().cpu().numpy().tolist()
                    # traj_comm_decoded = traj_comm_decoded.detach().cpu().numpy().tolist()
                    # for agent_id in range(len(comm_out)):
                    #     comm_encode_path = csv_path + str(agent_id) + "_comm_encode.csv"
                    #     self.append_to_csv(comm_encode_path, comm_encode[agent_id])

                    #     comm_out_path = csv_path + str(agent_id) + "_comm_out.csv"
                    #     # print("agent_{}'s comm is {}".format(agent_id, comm_out))
                    #     self.append_to_csv(comm_out_path, comm_out[agent_id])

                    #     comm_decoded_path = csv_path + str(agent_id) + "_comm_decoded.csv"
                    #     self.append_to_csv(comm_decoded_path, comm_decoded[agent_id])

                    #     traj_comm_encode_path = csv_path + str(agent_id) + "_traj_comm_encode.csv"
                    #     self.append_to_csv(traj_comm_encode_path, traj_comm_encode[agent_id])
                
                    #     traj_comm_out_path = csv_path + str(agent_id) + "_traj_comm_out.csv"
                    #     self.append_to_csv(traj_comm_out_path, traj_comm_out[agent_id])

                    #     traj_comm_decoded_path = csv_path + str(agent_id) + "_traj_comm_decoded.csv"
                    #     self.append_to_csv(traj_comm_decoded_path, traj_comm_decoded[agent_id])

            if self.iter.value > self.max_iteration:
                self.done.value = 1
        return
    
    def append_to_csv(self, csv_path, data):
        with open(csv_path, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(data)

    def record_comm_info(self, agent_id, 
                         comm_out=None, comm_decoded=None, traj_comm_out=None, traj_comm_decoded=None):
        with self.iter.get_lock():
            # self.iter.value += 1

            csv_path = "purple_samerew_om1_agent_"
            if self.iter.value % 100 == 0:
                
                if comm_out is not None:
                    comm_out_path = csv_path + str(agent_id) + "_comm_out.csv"
                    # convert torch tensor to list
                    comm_out = comm_out.detach().cpu().numpy().tolist()
                    # print("agent_{}'s comm is {}".format(agent_id, comm_out))
                    self.append_to_csv(comm_out_path, comm_out)
                if comm_decoded is not None:
                    comm_decoded_path = csv_path + str(agent_id) + "_comm_decoded.csv"
                    comm_decoded = comm_decoded.detach().cpu().numpy().tolist()
                    self.append_to_csv(comm_decoded_path, comm_decoded)
                if traj_comm_out is not None:
                    traj_comm_out_path = csv_path + str(agent_id) + "_traj_comm_out.csv"
                    traj_comm_out = traj_comm_out.detach().cpu().numpy().tolist()
                    self.append_to_csv(traj_comm_out_path, traj_comm_out)
                if traj_comm_decoded is not None:
                    traj_comm_decoded_path = csv_path + str(agent_id) + "_traj_comm_decoded.csv"
                    traj_comm_decoded = traj_comm_decoded.detach().cpu().numpy().tolist()
                    self.append_to_csv(traj_comm_decoded_path, traj_comm_decoded)
        return 

    def is_done(self):
        return self.done.value

    def save_ckpt(self, weight_iter, save_path):
        torch.save({'net': self.net.state_dict(),
                    'opt': self.opt.state_dict(),
                    'iter': weight_iter}, save_path)

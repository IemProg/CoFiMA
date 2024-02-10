import logging
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FinetuneIncrementalNet
from torchvision import transforms
from torch.distributions.multivariate_normal import MultivariateNormal
import random
from utils.toolkit import tensor2numpy, accuracy
import copy
import os
import wandb
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

epochs = 20
lrate = 0.01 
milestones = [60,100,140]
lrate_decay = 0.1
batch_size = 128
split_ratio = 0.1
T = 2
weight_decay = 5e-4
num_workers = 8
ca_epochs = 5

fishermax = 0.0001


def interpolate_weights(theta_0, theta_1, alpha, fisher=False, fisher_mat=None):
    # interpolate between checkpoints with mixing coefficient alpha
    if not fisher:
        theta = {
            key: (1 - alpha) * theta_0[key] + alpha * theta_1[key]
            for key in theta_0.keys()
        }

        # Find the additional head weights
        unique_keys = set(theta_1.keys()) - set(theta_0.keys())
        for item in unique_keys:
            theta[item] = theta_1[item]

    else:
        assert len(fisher_mat) == 2
        # weights of current task model, index: 1
        F_theta1 = {
            key: fisher_mat[1][key] * theta_1[key]
            for key in theta_1.keys()
        }

        # weights of previous task model, index: 1
        F_theta0 = {
            key: fisher_mat[0][key] * theta_0[key]
            for key in theta_0.keys()
        }

        # Weighted average of the weights using Fisher coeff, and normalize
        # new_theta = ((1 - alpha) * F0 *theta0 + alpha * F1 *theta1) / ((1 - alpha) * F0 + alpha * F1)

        theta = {
            key: ((1 - alpha) * F_theta0[key] + alpha * F_theta1[key]) /
                                ((1 - alpha) * fisher_mat[0][key] + alpha * fisher_mat[1][key])
            for key in theta_0.keys()
        }

        # Find the additional head weights
        unique_keys = set(theta_1.keys()) - set(theta_0.keys())
        for item in unique_keys:
            theta[item] = F_theta1[item]

    return theta

class CoFiMA(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self._network = FinetuneIncrementalNet(args['convnet_type'], pretrained=True, args=args)

        self.log_path = "logs/{}/{}/{}_{}".format(args['exp_grp'], args['experiment_name'],
                                                  args['model_name'], args['model_postfix'])
        os.makedirs(self.log_path, exist_ok=True)
        self.model_prefix = args['prefix']
        if 'epochs' in args.keys():
            global epochs
            epochs = args['epochs'] 
        if 'milestones' in args.keys():
            global milestones
            milestones = args['milestones']
        if 'lr' in args.keys():
            global lrate
            lrate = args['lr']
            print('set lr to ', lrate)
        if 'bcb_lrscale' in args.keys():
            self.bcb_lrscale = args['bcb_lrscale']
        else:
            self.bcb_lrscale = 1.0/100
        if self.bcb_lrscale == 0:
            self.fix_bcb = True
        else:
            self.fix_bcb = False
        print('fic_bcb', self.fix_bcb)
        logging.info('fic_bcb: {}'.format(self.fix_bcb))
        
        if 'save_before_ca' in args.keys() and args['save_before_ca']:
            self.save_before_ca = True
        else:
            self.save_before_ca = False

        if 'ca_epochs' in args.keys():
            global ca_epochs
            ca_epochs = args['ca_epochs'] 

        if 'ca_with_logit_norm' in args.keys() and args['ca_with_logit_norm']>0:
            self.logit_norm = args['ca_with_logit_norm']
        else:
            self.logit_norm = None

        self.run_id = args['run_id']
        self.seed = args['seed']
        self.task_sizes = []

        # Store previous weights
        self.prev_nets = []
        self.init_nets = []

        # Append the pre-trained model (task 0: assuming having the most generalizable)
        if self.args["init_w"] == 0:
            self.prev_nets.append(copy.deepcopy(self._network.state_dict()))

        # Store diag fisher matrices
        self.fisher_mat = []

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info('Exemplar size: {}'.format(self.exemplar_size))
        self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}'.format(self.seed), head_only=self.fix_bcb)
        self._network.fc.recall()

    def incremental_train(self, data_manager):
        self._cur_task += 1
        task_size = data_manager.get_task_size(self._cur_task)
        self.task_sizes.append(task_size)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self.topk = self._total_classes if self._total_classes<5 else 5
        self._network.update_fc(data_manager.get_task_size(self._cur_task))
        logging.info('Learning on {}-{}'.format(self._known_classes, self._total_classes))

        self._network.to(self._device)

        train_dset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes),
                                                  source='train', mode='train',
                                                  appendent=[], with_raw=False)
        test_dset = data_manager.get_dataset(np.arange(0, self._total_classes), source='test', mode='test')
        dset_name = data_manager.dataset_name.lower()

        self.train_loader = DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.test_loader = DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        self._stage1_training(self.train_loader, self.test_loader)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        # CA
        self._network.fc.backup()
        if self.save_before_ca:
            self.save_checkpoint(self.log_path+'/'+self.model_prefix+'_seed{}_before_ca'.format(self.seed), head_only=self.fix_bcb)

        # compute the fisher information matrix
        if self.args["fisher_weighting"]:
            self.fisher_mat.append(self.getFisherDiagonal(self.train_loader, self.optimizer))

        if self._cur_task > 0:
            self.init_nets.append(copy.deepcopy(self._network.state_dict()))
            theta_0 = self.prev_nets[self.args["init_w"]]
            theta_1 = self._network.state_dict()

            if self.args["fisher_weighting"]:
                theta = interpolate_weights(theta_0, theta_1,
                                                    alpha=self.args["wt_alpha"],
                                                    fisher=True, fisher_mat=self.fisher_mat[-2:])
            else:
                theta = interpolate_weights(theta_0, theta_1, alpha=self.args["wt_alpha"])

            # update the model according to the new weights
            self._network.load_state_dict(theta, strict=True)


        self._compute_class_mean(data_manager, check_diff=False, oracle=False)

        if self._cur_task>0 and ca_epochs>0:
            self._stage2_compact_classifier(task_size)
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module

        self.prev_nets.append(copy.deepcopy(self._network.state_dict()))

    def getFisherDiagonal(self, train_loader, optimizer):
        fisher = {
            n: torch.zeros(p.shape).to(self._device)
            for n, p in self._network.named_parameters()
            #if p.requires_grad
        }
        self._network.train()
        for i, (_, inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            logits = self._network(inputs, bcb_no_grad=self.fix_bcb)['logits']
            loss = torch.nn.functional.cross_entropy(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            for n, p in self._network.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.pow(2).clone()

        for n, p in fisher.items():
            fisher[n] = p / len(train_loader)
            fisher[n] = torch.min(fisher[n], torch.tensor(fishermax))
        return fisher

    def _run(self, train_loader, test_loader, optimizer, scheduler):
        run_epochs = epochs
        for epoch in range(1, run_epochs+1):
            self._network.train()
            losses = 0.
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                logits = self._network(inputs, bcb_no_grad=self.fix_bcb)['logits']
                cur_targets = torch.where(targets-self._known_classes>=0,targets-self._known_classes,-100)
                loss = F.cross_entropy(logits[:, self._known_classes:], cur_targets)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            if epoch%5==0:
                train_acc = self._compute_accuracy(self._network, train_loader)
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.3f}, Test_accy {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader), train_acc, test_acc)
                wandb.log({'Train acc': train_acc, 'Test acc': test_acc})
            else:
                info = 'Task {}, Epoch {}/{} => Loss {:.3f}'.format(
                    self._cur_task, epoch, epochs, losses/len(train_loader))
            logging.info(info)
            wandb.log({'Train loss': losses / len(train_loader)})

            if self.args["ema"]:
                if epoch % self.args["ema_update"] == 0:
                    self.update_ema()

    def _stage1_training(self, train_loader, test_loader):
        '''
        if self._cur_task == 0:
            loaded_dict = torch.load('./dict_0.pkl')
            self._network.load_state_dict(loaded_dict['model_state_dict'])
            self._network.to(self._device)
            return
        '''
        base_params = self._network.convnet.parameters()
        base_fc_params = [p for p in self._network.fc.parameters() if p.requires_grad==True]
        head_scale = 1. if 'moco' in self.log_path else 1.
        if not self.fix_bcb:
            base_params = {'params': base_params, 'lr': lrate*self.bcb_lrscale, 'weight_decay': weight_decay}
            #base_params = {'params': base_params, 'lr': 0.01, 'weight_decay': 0.005}
            base_fc_params = {'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}
            network_params = [base_params, base_fc_params]
        else:
            for p in base_params:
                p.requires_grad = False
            network_params = [{'params': base_fc_params, 'lr': lrate*head_scale, 'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=lrate_decay)

        self.optimizer = optimizer

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._run(train_loader, test_loader, optimizer, scheduler)


    def _stage2_compact_classifier(self, task_size):
        for p in self._network.fc.parameters():
            p.requires_grad=True
            
        run_epochs = ca_epochs
        crct_num = self._total_classes    
        param_list = [p for p in self._network.fc.parameters() if p.requires_grad]
        network_params = [{'params': param_list, 'lr': lrate,
                           'weight_decay': weight_decay}]
        optimizer = optim.SGD(network_params, lr=lrate, momentum=0.9, weight_decay=weight_decay)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[4], gamma=lrate_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)

        self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self._network.eval()
        for epoch in range(run_epochs):
            losses = 0.

            sampled_data = []
            sampled_label = []
            num_sampled_pcls = 256
        
            for c_id in range(crct_num):
                t_id = c_id // task_size
                decay = (t_id + 1) / (self._cur_task + 1) * 0.1
                cls_mean = torch.tensor(self._class_means[c_id], dtype=torch.float64).to(self._device) * (
                            0.9 + decay)  # torch.from_numpy(self._class_means[c_id]).to(self._device)
                cls_cov = self._class_covs[c_id].to(self._device)

                m = MultivariateNormal(cls_mean.float(), cls_cov.float())

                sampled_data_single = m.sample(sample_shape=(num_sampled_pcls,))
                sampled_data.append(sampled_data_single)
                sampled_label.extend([c_id] * num_sampled_pcls)

                sampled_data = torch.cat(sampled_data, dim=0).float().to(self._device)
                sampled_label = torch.tensor(sampled_label).long().to(self._device)

            inputs = sampled_data
            targets= sampled_label

            sf_indexes = torch.randperm(inputs.size(0))
            inputs = inputs[sf_indexes]
            targets = targets[sf_indexes]

            
            for _iter in range(crct_num):
                inp = inputs[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                tgt = targets[_iter*num_sampled_pcls:(_iter+1)*num_sampled_pcls]
                outputs = self._network(inp, bcb_no_grad=True, fc_only=True)
                logits = outputs['logits']

                if self.logit_norm is not None:
                    per_task_norm = []
                    prev_t_size = 0
                    cur_t_size = 0
                    for _ti in range(self._cur_task+1):
                        cur_t_size += self.task_sizes[_ti]
                        temp_norm = torch.norm(logits[:, prev_t_size:cur_t_size], p=2, dim=-1, keepdim=True) + 1e-7
                        per_task_norm.append(temp_norm)
                        prev_t_size += self.task_sizes[_ti]
                    per_task_norm = torch.cat(per_task_norm, dim=-1)
                    norms = per_task_norm.mean(dim=-1, keepdim=True)
                        
                    norms_all = torch.norm(logits[:, :crct_num], p=2, dim=-1, keepdim=True) + 1e-7
                    decoupled_logits = torch.div(logits[:, :crct_num], norms) / self.logit_norm
                    loss = F.cross_entropy(decoupled_logits, tgt)

                else:
                    loss = F.cross_entropy(logits[:, :crct_num], tgt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

            scheduler.step()
            test_acc = self._compute_accuracy(self._network, self.test_loader)
            info = 'CA Task {} => Loss {:.3f}, Test_accy {:.3f}'.format(
                self._cur_task, losses/self._total_classes, test_acc)
            logging.info(info)
            wandb.log({'CA Task': losses/self._total_classes})
            wandb.log({'CA Acc': test_acc})




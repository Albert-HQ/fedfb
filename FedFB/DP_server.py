import torch, copy, time, random, warnings, os
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
from ray import tune
import torch.nn as nn
from utils import compute_eod

################## MODEL SETTING ########################
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
os.environ['KMP_DUPLICATE_LIB_OK']='True'
#########################################################

class Server(object):
    def __init__(self, model, dataset_info, seed = 123, num_workers = 4, ret = False, 
                train_prn = False, metric = "Demographic disparity", select_round = False,
                batch_size = 128, print_every = 1, fraction_clients = 1, Z = 2, prn = True, trial = False):
        """
        Server execution.

        Parameters
        ----------
        model: torch.nn.Module object.

        dataset_info: a list of three objects.
            - train_dataset: Dataset object.
            - test_dataset: Dataset object.
            - clients_idx: a list of lists, with each sublist contains the indexs of the training samples in one client.
                    the length of the list is the number of clients.

        seed: random seed.

        num_workers: number of workers.

        ret: boolean value. If true, return the accuracy and fairness measure and print nothing; else print the log and return None.

        train_prn: boolean value. If true, print the batch loss in local epochs.

        metric: three options, "Risk Difference", "pRule", "Demographic disparity".

        batch_size: a positive integer.

        print_every: a positive integer. eg. print_every = 1 -> print the information of that global round every 1 round.

        fraction_clients: float from 0 to 1. The fraction of clients chose to update the weights in each round.
        """

        self.model = model
        if torch.cuda.device_count()>1:
            self.model = nn.DataParallel(self.model)
        self.model.to(DEVICE)

        self.seed = seed
        self.num_workers = num_workers

        self.ret = ret
        self.prn = prn
        self.train_prn = False if ret else train_prn

        self.metric = metric
        if metric == "Risk Difference":
            self.disparity = riskDifference
        elif metric == "pRule":
            self.disparity = pRule
        elif metric == "Demographic disparity":
            self.disparity = DPDisparity
        else:
            warnings.warn("Warning message: metric " + metric + " is not supported! Use the default metric Demographic disparity. ")
            self.disparity = DPDisparity
            self.metric = "Demographic disparity"

        self.batch_size = batch_size
        self.print_every = print_every
        self.fraction_clients = fraction_clients

        self.train_dataset, self.test_dataset, self.clients_idx = dataset_info
        self.num_clients = len(self.clients_idx)
        self.Z = Z

        self.trial = trial
        self.select_round = select_round

        self.trainloader, self.validloader = self.train_val(self.train_dataset, batch_size)
    
    def train_val(self, dataset, batch_size, idxs_train_full = None, split = False):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(self.seed)
        
        # split indexes for train, validation (90, 10)
        if idxs_train_full == None: idxs_train_full = np.arange(len(dataset))
        idxs_train = idxs_train_full[:int(0.9*len(idxs_train_full))]
        idxs_val = idxs_train_full[int(0.9*len(idxs_train_full)):]
    
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                    batch_size=batch_size, shuffle=True)

        if split:
            validloader = {}
            for sen in range(self.Z):
                sen_idx = np.where(dataset.sen[idxs_val] == sen)[0]
                validloader[sen] = DataLoader(DatasetSplit(dataset, idxs_val[sen_idx]),
                                        batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        else:
            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def FedAvg(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = "adam"):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()
            m = max(1, int(self.fraction_clients * self.num_clients)) # the number of clients to be chosen in each round_
            idxs_users = np.random.choice(range(self.num_clients), m, replace=False)

            for idx in idxs_users:
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss = local_model.standard_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = average_weights(local_weights, self.clients_idx, idxs_users)
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
            self.model.eval()
            for c in range(m):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model) 
                list_acc.append(acc)

                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)

        # Test inference after completion of training
        test_acc, n_yz, test_eod = self.test_inference(self.model, self.test_dataset)  # ★
        dp_gap = self.disparity(n_yz)  # ★ 改变量名更直观

        if self.prn:
            print(f'\n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

            # 打印两个公平指标：DP Disp + EOD
            print("|---- Test DP Disp : {:.4f}".format(dp_gap))  # ★
            print("|---- Test EOD     : {:.4f}".format(test_eod))  # ★

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time() - start_time))

        # ★ 返回值也带上 EOD，外层可继续用
        if self.ret:
            return test_acc, dp_gap, test_eod, self.model  # ★

    def FedFB(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3, bits = False):
        # only support 2 groups
        # if self.Z == 2 and bits is None:
        #     # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
        #     # set seed
        #     np.random.seed(self.seed)
        #     random.seed(self.seed)
        #     torch.manual_seed(self.seed)

        #     # Training
        #     train_loss, train_accuracy = [], []
        #     start_time = time.time()
        #     weights = self.model.state_dict()
        #     if self.select_round: best_fairness = float('inf')

        #     # the number of samples whose label is y and sensitive attribute is z
        #     m_yz, lbd = {}, {}
        #     for y in [0,1]:
        #         for z in range(self.Z):
        #             m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

        #     for y in [0,1]:
        #         for z in range(self.Z):
        #             lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

        #     for round_ in tqdm(range(num_rounds)):
        #         local_weights, local_losses, nc = [], [], []
        #         if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

        #         self.model.train()

        #         for idx in range(self.num_clients):
        #             local_model = Client(dataset=self.train_dataset,
        #                                         idxs=self.clients_idx[idx], batch_size = self.batch_size, 
        #                                     option = "FB-Variant1", 
        #                                     seed = self.seed, prn = self.train_prn, Z = self.Z)

        #             w, loss, nc_ = local_model.fb_update(
        #                             model=copy.deepcopy(self.model), global_round=round_, 
        #                                 learning_rate = learning_rate / (round_+1), local_epochs = local_epochs, 
        #                                 optimizer = optimizer, lbd = lbd, m_yz = m_yz)
        #             nc.append(nc_)
        #             local_weights.append(copy.deepcopy(w))
        #             local_losses.append(copy.deepcopy(loss))

        #         # update global weights
        #         weights = weighted_average_weights(local_weights, nc, sum(nc))
        #         self.model.load_state_dict(weights)

        #         loss_avg = sum(local_losses) / len(local_losses)
        #         train_loss.append(loss_avg)

        #         # Calculate avg training accuracy over all clients at every round
        #         list_acc = []
        #         # the number of samples which are assigned to class y and belong to the sensitive group z
        #         n_yz, loss_yz = {}, {}
        #         for y in [0,1]:
        #             for z in range(self.Z):
        #                 n_yz[(y,z)] = 0
        #                 loss_yz[(y,z)] = 0

        #         self.model.eval()
        #         for c in range(self.num_clients):
        #             local_model = Client(dataset=self.train_dataset,
        #                                         idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1", 
        #                                         seed = self.seed, prn = self.train_prn, Z = self.Z)
        #             # validation dataset inference
        #             acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference(model = self.model, bits = bits) 
        #             list_acc.append(acc)
                    
        #             for yz in n_yz:
        #                 n_yz[yz] += n_yz_c[yz]
        #                 loss_yz[yz] += loss_yz_c[yz]
                        
        #             if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
        #                 c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))
                    
        #         # update the lambda according to the paper -> see Section A.1 of FairBatch
        #         # works well! The real batch size would be slightly different from the setting
        #         for y, z in loss_yz:
        #             loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

        #         y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
        #         y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
        #         if y0_diff > y1_diff:
        #             lbd[(0,0)] -= alpha / (round_+1)
        #             lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
        #             lbd[(1,0)] = 1 - lbd[(0,0)]
        #             lbd[(0,1)] += alpha / (round_+1)
        #             lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
        #             lbd[(1,1)] = 1 - lbd[(0,1)]
        #         else:
        #             lbd[(0,0)] += alpha / (round_+1)
        #             lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
        #             lbd[(0,1)] = 1 - lbd[(0,0)]
        #             lbd[(1,0)] -= alpha / (round_+1)
        #             lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
        #             lbd[(1,1)] = 1 - lbd[(1,0)]

        #         train_accuracy.append(sum(list_acc)/len(list_acc))

        #         # print global training loss after every 'i' rounds
        #         if self.prn:
        #             if (round_+1) % self.print_every == 0:
        #                 print(f' \nAvg Training Stats after {round_+1} global rounds:')
        #                 print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
        #                     np.mean(np.array(train_loss)), 
        #                     100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

        #         if self.trial:
        #             with tune.checkpoint_dir(round_) as checkpoint_dir:
        #                 path = os.path.join(checkpoint_dir, "checkpoint")
        #                 torch.save(self.model.state_dict(), path)
                        
        #             tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)
        #         if self.select_round: 
        #             if best_fairness > self.disparity(n_yz): 
        #                 best_fairness = self.disparity(n_yz)
        #                 test_model = copy.deepcopy(self.model.state_dict())

        #     # Test inference after completion of training
        #     if self.select_round: self.model.load_state_dict(test_model)
        #     test_acc, n_yz = self.test_inference(self.model, self.test_dataset)
        #     rd = self.disparity(n_yz)

        #     if self.prn:
        #         print(f' \n Results after {num_rounds} global rounds of training:')
        #         print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
        #         print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

        #         # Compute fairness metric
        #         print("|---- Test "+ self.metric+": {:.4f}".format(rd))

        #         print('\n Total Run Time: {0:0.4f} sec'.format(time.time()-start_time))

        #     if self.ret: return test_acc, rd, self.model

        # support more than 2 groups
        # else:
            # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
            # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        if self.select_round: best_fairness = float('inf')

        # the number of samples whose label is y and sensitive attribute is z
        m_yz, lbd = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

        for y in [0,1]:
            for z in range(self.Z):
                lbd[(y,z)] = (m_yz[(1,z)] + m_yz[(0,z)])/len(self.train_dataset)

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses, nc = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[idx], batch_size = self.batch_size, 
                                        option = "FB-Variant1", 
                                        seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss, nc_ = local_model.fb2_update(
                                model=copy.deepcopy(self.model), global_round=round_, 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, m_yz = m_yz, lbd = lbd)
                nc.append(nc_)
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz, f_z = {}, {}
            for z in range(self.Z):
                for y in [0,1]:
                    n_yz[(y,z)] = 0

            for z in range(1, self.Z):
                f_z[z] = 0

            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1", 
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, f_z_c = local_model.inference(model = self.model, train = True, bits = bits, truem_yz= m_yz) 
                list_acc.append(acc)
                
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                
                for z in range(1, self.Z):
                    f_z[z] += f_z_c[z] + m_yz[(0,0)]/(m_yz[(0,0)] + m_yz[(1,0)]) - m_yz[(0,z)]/(m_yz[(0,z)] + m_yz[(1,z)])
                    
                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            for z in range(self.Z):
                if z == 0:
                    lbd[(0,z)] -= alpha / (round_ + 1) ** .5 * sum([f_z[z] for z in range(1, self.Z)])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                else:
                    lbd[(0,z)] += alpha / (round_ + 1) ** .5 * f_z[z]
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
            
            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)
            if self.select_round: 
                if best_fairness > self.disparity(n_yz): 
                    best_fairness = self.disparity(n_yz)
                    test_model = copy.deepcopy(self.model.state_dict())

        # Test inference after completion of training
        if self.select_round: self.model.load_state_dict(test_model)

        # Test inference after completion of training
        test_acc, n_yz, test_eod = self.test_inference(self.model, self.test_dataset)  # ★
        dp_gap = self.disparity(n_yz)  # ★ 改变量名更直观

        if self.prn:
            print(f'\n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

            # 打印两个公平指标：DP Disp + EOD
            print("|---- Test DP Disp : {:.4f}".format(dp_gap))  # ★
            print("|---- Test EOD     : {:.4f}".format(test_eod))  # ★

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time() - start_time))

        # ★ 返回值也带上 EOD，外层可继续用
        if self.ret:
            return test_acc, dp_gap, test_eod, self.model  # ★

    def CFLFB(self, outer_rounds = 10,  inner_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = 0.3):
        # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
        if self.Z == 2:
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy = [], []
            start_time = time.time()

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                            momentum=0.5) # 
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for round_ in tqdm(range(outer_rounds)):
                if self.prn and (round_+1) % 10 == 0: print(f'\n | Global Training Round : {round_+1} |\n')

                self.model.train()
                batch_loss = []
                for _ in range(inner_epochs):
                    for _, (features, labels, sensitive) in enumerate(self.trainloader):
                        features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                        sensitive = sensitive.to(DEVICE)
                        _, logits = self.model(features)

                        v = torch.randn(len(labels)).type(torch.DoubleTensor)
                        
                        group_idx = {}
                        
                        for y, z in lbd:
                            group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                            v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                        loss = weighted_loss(logits, labels, v)

                        optimizer.zero_grad()
                        if not np.isnan(loss.item()): loss.backward()
                        optimizer.step()
                        batch_loss.append(loss.item())

                loss_avg = sum(batch_loss)/len(batch_loss)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0

                self.model.eval()
                    # validation dataset inference
                acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference(model = self.model, option = 'FairBatch')
                list_acc.append(acc)

                    
                if self.prn and (round_+1) % 10 == 0: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    acc_loss, fair_loss, self.metric, self.disparity(n_yz)))
                    
                # update the lambda according to the paper -> see Section A.1 of FairBatch
                # works well! The real batch size would be slightly different from the setting
                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
                y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
                if y0_diff > y1_diff:
                    lbd[(0,0)] -= alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(1,0)] = 1 - lbd[(0,0)]
                    lbd[(0,1)] += alpha / (round_+1)
                    lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                    lbd[(1,1)] = 1 - lbd[(0,1)]
                else:
                    lbd[(0,0)] += alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(0,1)] = 1 - lbd[(0,0)]
                    lbd[(1,0)] -= alpha / (round_+1)
                    lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                    lbd[(1,1)] = 1 - lbd[(1,0)]

                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn and (round_+1) % 10 == 0:
                    if (round_+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_+1} global rounds:')
                        print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                            np.mean(np.array(train_loss)), 
                            100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

                if self.trial:
                    with tune.checkpoint_dir(round_) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(self.model.state_dict(), path)
                        
                    tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)     


            # Test inference after completion of training
            test_acc, n_yz, test_eod = self.test_inference(self.model, self.test_dataset)  # ★
            dp_gap = self.disparity(n_yz)  # ★ 改变量名更直观

            if self.prn:
                print(f'\n Results after {num_rounds} global rounds of training:')
                print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
                print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

                # 打印两个公平指标：DP Disp + EOD
                print("|---- Test DP Disp : {:.4f}".format(dp_gap))  # ★
                print("|---- Test EOD     : {:.4f}".format(test_eod))  # ★

                print('\n Total Run Time: {0:0.4f} sec'.format(time.time() - start_time))

            # ★ 返回值也带上 EOD，外层可继续用
            if self.ret:
                return test_acc, dp_gap, test_eod, self.model  # ★
        else:
            # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy = [], []
            start_time = time.time()

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate,
                                            momentum=0.5) # 
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = (m_yz[(1,z)] + m_yz[(0,z)])/len(self.train_dataset)

            for round_ in tqdm(range(outer_rounds)):
                if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

                self.model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = self.model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)
                    
                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])

                    loss = weighted_loss(logits, labels, v, False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                loss_avg = sum(batch_loss) / len(batch_loss)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0

                self.model.eval()
                acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference(model = self.model, option = 'FairBatch') 
                list_acc.append(acc)

                if self.prn and (round_+1) % 50 == 0: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    acc_loss, fair_loss, self.metric, self.disparity(n_yz)))
                    
                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                for z in range(self.Z):
                    if z == 0:
                        lbd[(0,z)] -= alpha / (round_ + 1) ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(self.Z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                    else:
                        lbd[(0,z)] += alpha / (round_ + 1) ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                
                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                if self.prn and (round_+1) % 50 == 0:
                    if (round_+1) % self.print_every == 0:
                        print(f' \nAvg Training Stats after {round_+1} global rounds:')
                        print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                            np.mean(np.array(train_loss)), 
                            100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

                if self.trial:
                    with tune.checkpoint_dir(round_) as checkpoint_dir:
                        path = os.path.join(checkpoint_dir, "checkpoint")
                        torch.save(self.model.state_dict(), path)
                        
                    tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)

            # Test inference after completion of training
            test_acc, n_yz, test_eod = self.test_inference(self.model, self.test_dataset)  # ★
            dp_gap = self.disparity(n_yz)  # ★ 改变量名更直观

            if self.prn:
                print(f'\n Results after {num_rounds} global rounds of training:')
                print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
                print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

                # 打印两个公平指标：DP Disp + EOD
                print("|---- Test DP Disp : {:.4f}".format(dp_gap))  # ★
                print("|---- Test EOD     : {:.4f}".format(test_eod))  # ★

                print('\n Total Run Time: {0:0.4f} sec'.format(time.time() - start_time))

            # ★ 返回值也带上 EOD，外层可继续用
            if self.ret:
                return test_acc, dp_gap, test_eod, self.model  # ★

    def UFLFB(self, num_epochs = 300, learning_rate = (0.005, 0.005, 0.005), alpha = (0.08,0.1,0.1), optimizer = 'adam'):
        models = []
        for c in range(self.num_clients):
            local_model = Client(dataset=self.train_dataset,
                                idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1",
                                seed = self.seed, prn = self.train_prn, Z = self.Z)
            models.append(local_model.uflfb_update(copy.deepcopy(self.model).to(DEVICE), num_epochs, learning_rate[c], optimizer, alpha[c]))
        
        # Test inference after completion of training
        test_acc, n_yz, test_eod = self.ufl_inference(models, self.test_dataset)
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_epochs} local epochs of training:')
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))
            print("|---- Test EOD     : {:.4f}".format(test_eod))

        if self.ret:
            return test_acc, rd, test_eod, self.model

    def FFLFB(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, optimizer = 'adam', alpha = (0.3,0.3,0.3)):
        # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
        # set seed
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()
        if self.select_round: best_fairness = float('inf')

        lbd, m_yz, nc = [None for _ in range(self.num_clients)], [None for _ in range(self.num_clients)], [None for _ in range(self.num_clients)]

        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses = [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "FB-Variant1", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss, nc_, lbd_, m_yz_ = local_model.local_fb(
                                model=copy.deepcopy(self.model), 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, alpha = alpha[idx], lbd = lbd[idx], m_yz = m_yz[idx])
                lbd[idx], m_yz[idx], nc[idx] = lbd_, m_yz_, nc_
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nc, sum(nc))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

            # Calculate avg training accuracy over all clients at every round
            list_acc = []
            # the number of samples which are assigned to class y and belong to the sensitive group z
            n_yz, loss_yz = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
                    loss_yz[(y,z)] = 0

            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset,
                                            idxs=self.clients_idx[c], batch_size = self.batch_size, option = "FB-Variant1", 
                                            seed = self.seed, prn = self.train_prn, Z = self.Z)
                # validation dataset inference
                acc, loss, n_yz_c, acc_loss, fair_loss, loss_yz_c = local_model.inference(model = self.model) 
                list_acc.append(acc)
                
                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]
                    loss_yz[yz] += loss_yz_c[yz]
                    
                if self.prn: print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)  
            if self.select_round: 
                if best_fairness > self.disparity(n_yz): 
                    best_fairness = self.disparity(n_yz)
                    test_model = copy.deepcopy(self.model.state_dict())

        # Test inference after completion of training
        if self.select_round: self.model.load_state_dict(test_model)
        test_acc, n_yz, test_eod = self.test_inference(self.model, self.test_dataset)  # ★
        dp_gap = self.disparity(n_yz)  # ★ 改变量名更直观

        if self.prn:
            print(f'\n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

            # 打印两个公平指标：DP Disp + EOD
            print("|---- Test DP Disp : {:.4f}".format(dp_gap))  # ★
            print("|---- Test EOD     : {:.4f}".format(test_eod))  # ★

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time() - start_time))

        # ★ 返回值也带上 EOD，外层可继续用
        if self.ret:
            return test_acc, dp_gap, test_eod, self.model  # ★

    # only support binary sensitive attribute
    # assign a higher weight to clients that have similar local fairness to the global fairness metric
    def FairFed(self, num_rounds = 10, local_epochs = 30, learning_rate = 0.005, beta = 0.3, alpha = 0.1, optimizer = 'adam'):
        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Training
        train_loss, train_accuracy = [], []
        start_time = time.time()
        weights = self.model.state_dict()

        lbd, m_yz = [None for _ in range(self.num_clients)], [None for _ in range(self.num_clients)]
        
        for round_ in tqdm(range(num_rounds)):
            local_weights, local_losses, nw = [], [], []
            if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

            # get local fairness metric
            list_acc = []
            n_yz = {}
            for y in [0,1]:
                for z in range(self.Z):
                    n_yz[(y,z)] = 0
            self.model.eval()
            for c in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[c], 
                            batch_size = self.batch_size, option = "unconstrained", seed = self.seed, prn = self.train_prn, Z = self.Z)
                acc, loss, n_yz_c, acc_loss, fair_loss, _ = local_model.inference(model = self.model, train = True) 
                list_acc.append(acc)

                for yz in n_yz:
                    n_yz[yz] += n_yz_c[yz]

                nw.append(self.disparity(n_yz_c))
                    
                if self.prn: 
                    print("Client %d: accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                            c+1, acc_loss, fair_loss, self.metric, self.disparity(n_yz_c)))

            train_accuracy.append(sum(list_acc)/len(list_acc))
            
            for c in range(self.num_clients):
                nw[c] = np.exp(-beta * abs(nw[c] - self.disparity(n_yz))) * len(self.clients_idx[c]) / len(self.train_dataset)

            # print global training loss after every 'i' rounds
            if self.prn:
                if (round_+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {round_+1} global rounds:')
                    print("Training loss: %.2f | Validation accuracy: %.2f%% | Validation %s: %.4f" % (
                        np.mean(np.array(train_loss)), 
                        100*train_accuracy[-1], self.metric, self.disparity(n_yz)))

            if self.trial:
                with tune.checkpoint_dir(round_) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = train_accuracy[-1], disp = self.disparity(n_yz), iteration = round_+1)

            self.model.train()

            for idx in range(self.num_clients):
                local_model = Client(dataset=self.train_dataset, idxs=self.clients_idx[idx], 
                            batch_size = self.batch_size, option = "FB-Variant1", seed = self.seed, prn = self.train_prn, Z = self.Z)

                w, loss, _, lbd_, m_yz_ = local_model.local_fb(
                                model=copy.deepcopy(self.model), 
                                    learning_rate = learning_rate, local_epochs = local_epochs, 
                                    optimizer = optimizer, alpha = alpha, lbd = lbd[idx], m_yz = m_yz[idx])
                lbd[idx], m_yz[idx] = lbd_, m_yz_
                local_weights.append(copy.deepcopy(w))
                local_losses.append(copy.deepcopy(loss))

            # update global weights
            weights = weighted_average_weights(local_weights, nw, sum(nw))
            self.model.load_state_dict(weights)

            loss_avg = sum(local_losses) / len(local_losses)
            train_loss.append(loss_avg)

        # Test inference after completion of training
        test_acc, n_yz, test_eod = self.test_inference(self.model, self.test_dataset)  # ★
        dp_gap = self.disparity(n_yz)  # ★ 改变量名更直观

        if self.prn:
            print(f'\n Results after {num_rounds} global rounds of training:')
            print("|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print("|---- Test Accuracy: {:.2f}%".format(100 * test_acc))

            # 打印两个公平指标：DP Disp + EOD
            print("|---- Test DP Disp : {:.4f}".format(dp_gap))  # ★
            print("|---- Test EOD     : {:.4f}".format(test_eod))  # ★

            print('\n Total Run Time: {0:0.4f} sec'.format(time.time() - start_time))

        # ★ 返回值也带上 EOD，外层可继续用
        if self.ret:
            return test_acc, dp_gap, test_eod, self.model  # ★

    def FAFL(self, num_epochs = 300, learning_rate = 0.005, penalty = 2):
        def loss_with_agnostic_fair(logits, targets, sensitive, sen_bar, larg = 1):

            acc_loss = F.cross_entropy(logits, targets, reduction = 'sum')
            fair_loss0 = torch.mul(sensitive - sen_bar, logits.T[0] - torch.mean(logits.T[0]))
            fair_loss0 = torch.mean(torch.mul(fair_loss0, fair_loss0)) 
            fair_loss1 = torch.mul(sensitive - sen_bar, logits.T[1] - torch.mean(logits.T[1]))
            fair_loss1 = torch.mean(torch.mul(fair_loss1, fair_loss1)) 
            fair_loss = fair_loss0 + fair_loss1

            return acc_loss, larg*fair_loss, acc_loss+larg*fair_loss

        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        sen_bar = self.train_dataset.sen.mean()
        if self.select_round: best_fairness = float('inf')

        for epoch in tqdm(range(num_epochs)):
            self.model.train()

            for _, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)

                _, Theta_X = self.model(features)

                _,_,loss = loss_with_agnostic_fair(Theta_X, labels, sensitive, sen_bar, penalty)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            # validation dataset inference
            acc, loss, n_yz, acc_loss, fair_loss, _ = self.inference(model = self.model)

                
            if self.prn and (epoch+1) % 10 == 0: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                acc_loss, fair_loss, self.metric, self.disparity(n_yz)))
                

            # print global training loss after every 'i' rounds
            if self.prn and (epoch+1) % 10 == 0:
                if (epoch+1) % self.print_every == 0:
                    print(f' \nAvg Training Stats after {epoch+1} global rounds:')
                    print("Training accuracy: %.4f%% | Training %s: %.4f" % (acc*100, self.metric, self.disparity(n_yz)))  
            
            if self.trial:
                with tune.checkpoint_dir(epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save(self.model.state_dict(), path)
                    
                tune.report(loss = loss, accuracy = acc, disp = self.disparity(n_yz), iteration = epoch)     
            if self.select_round: 
                if best_fairness > self.disparity(n_yz): 
                    best_fairness = self.disparity(n_yz)
                    test_model = copy.deepcopy(self.model.state_dict())

        # Test inference after completion of training
        if self.select_round: self.model.load_state_dict(test_model)
        test_acc, n_yz, test_eod = self.test_inference(self.model, self.test_dataset)
        rd = self.disparity(n_yz)

        if self.prn:
            print(f' \n Results after {num_epochs} global rounds of training:')
            print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

            # Compute fairness metric
            print("|---- Test "+ self.metric+": {:.4f}".format(rd))
            print("|---- Test EOD     : {:.4f}".format(test_eod))

        if self.ret:
            return test_acc, rd, test_eod, self.model

    def inference(self, option = 'unconstrained', penalty = 100, model = None, validloader = None):
        """ 
        Returns the inference accuracy, 
                                loss, 
                                N(sensitive group, pos), 
                                N(non-sensitive group, pos), 
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        if model == None: model = self.model
        if validloader == None: 
            validloader = self.validloader
        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz, loss_yz = {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0
        
        for _, (features, labels, sensitive) in enumerate(validloader):
            features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            sensitive = sensitive.type(torch.LongTensor).to(DEVICE)
            
            # Inference
            outputs, logits = model(features)
            outputs, logits = outputs.to(DEVICE), logits.to(DEVICE)

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1

            group_boolean_idx = {}
            
            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()     
                
                if option == "FairBatch":
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("FB_inference", logits[group_boolean_idx[yz]].to(DEVICE), 
                                                    labels[group_boolean_idx[yz]].to(DEVICE), 
                                         outputs[group_boolean_idx[yz]].to(DEVICE), sensitive[group_boolean_idx[yz]].to(DEVICE), 
                                         penalty)
                    loss_yz[yz] += loss_yz_
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(option, logits, 
                                                        labels, outputs, sensitive, penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if option in ["FairBatch", "FB-Variant1"]:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, loss_yz
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None

    def test_inference(self, model=None, test_dataset=None):
        """
        Returns:
            accuracy  : float
            n_yz      : dict, 预测标签分布 (原有, 用于 DP 计算)
            eod_value : float, Equal‑Opportunity Difference (新增)
        """
        # 1) 固定随机种子（原本就有）
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # 2) 默认参数处理
        if model is None:
            model = self.model
        if test_dataset is None:
            test_dataset = self.test_dataset

        # 3) 推断准备
        model.eval()
        total, correct = 0.0, 0.0
        n_yz = {(y, z): 0 for y in [0, 1] for z in range(self.Z)}

        # >>> 新增：收集整体向量，稍后算 EOD
        y_true_all, y_pred_all, z_all = [], [], []
        # <<< 新增

        testloader = DataLoader(test_dataset,
                                batch_size=self.batch_size,
                                shuffle=False)

        # 4) 批量推断
        for _, (features, labels, sensitive) in enumerate(testloader):
            features = features.to(DEVICE)
            labels = labels.to(DEVICE).type(torch.long)

            # Forward
            outputs, _ = model(features)

            # 取预测标签
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1)

            # 准确率
            correct += torch.sum(pred_labels.eq(labels)).item()
            total += len(labels)

            # 组别计数（用于 DP）
            for y, z in n_yz:
                n_yz[(y, z)] += torch.sum((sensitive == z) & (pred_labels == y)).item()

            # >>> 新增：累积向量
            y_true_all.append(labels.cpu())
            y_pred_all.append(pred_labels.cpu())
            z_all.append(sensitive.cpu())
            # <<< 新增

        accuracy = correct / total

        # >>> 新增：计算 EOD，仅用一次并返回
        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        z_all = torch.cat(z_all)
        eod_value = compute_eod(y_true_all, y_pred_all, z_all)
        # <<< 新增

        # 5) 返回值多一个 eod
        return accuracy, n_yz, eod_value

    def ufl_inference(self, models, test_dataset = None):
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        if test_dataset == None: test_dataset = self.test_dataset

        total, correct = 0.0, 0.0
        n_yz = {}
        for y in [0,1]:
            for z in range(self.Z):
                n_yz[(y,z)] = 0

        y_true_all, y_pred_all, z_all = [], [], []

        for model in models:
            model.eval()

        testloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        for _, (features, labels, sensitive) in enumerate(testloader):
            features = features.to(DEVICE)
            labels =  labels.type(torch.LongTensor).to(DEVICE)
            sensitive = sensitive.to(DEVICE)

            # Inference
            outputs = torch.zeros((len(labels),2))
            for c in range(self.num_clients): 
                output, _ = models[c](features)
                output = output/output.sum()
                outputs += output * len(self.clients_idx[c])
            outputs = outputs / np.array(list(map(len, self.clients_idx))).sum()

            # Prediction
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            
            for y,z in n_yz:
                n_yz[(y,z)] += torch.sum((sensitive == z) & (pred_labels == y)).item()

            y_true_all.append(labels.cpu())
            y_pred_all.append(pred_labels.cpu())
            z_all.append(sensitive.cpu())

        accuracy = correct/total

        y_true_all = torch.cat(y_true_all)
        y_pred_all = torch.cat(y_pred_all)
        z_all = torch.cat(z_all)
        eod_value = compute_eod(y_true_all, y_pred_all, z_all)

        return accuracy, n_yz, eod_value

class Client(object):
    def __init__(self, dataset, idxs, batch_size, option, seed = 0, prn = True, penalty = 500, Z = 2):
        self.seed = seed 
        self.dataset = dataset
        self.idxs = idxs
        self.option = option
        self.prn = prn
        self.Z = Z
        self.trainloader, self.validloader = self.train_val(dataset, list(idxs), batch_size)
        self.penalty = penalty
        self.disparity = DPDisparity

    def train_val(self, dataset, idxs, batch_size):
        """
        Returns train, validation for a given local training dataset
        and user indexes.
        """
        torch.manual_seed(self.seed)
        
        # split indexes for train, validation (90, 10)
        idxs_train = idxs[:int(0.9*len(idxs))]
        idxs_val = idxs[int(0.9*len(idxs)):len(idxs)]

        self.train_dataset = DatasetSplit(dataset, idxs_train)
        self.test_dataset = DatasetSplit(dataset, idxs_val)

        trainloader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)

        validloader = DataLoader(self.test_dataset,
                                     batch_size=max(int(len(idxs_val)/10),10), shuffle=False)
        return trainloader, validloader

    def standard_update(self, model, global_round, learning_rate, local_epochs, optimizer): 
        # Set mode to train model
        model.train()
        epoch_loss = []

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        ) # momentum=0.5
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                # we need to set the gradients to zero before starting to do backpropragation 
                # because PyTorch accumulates the gradients on subsequent backward passes. 
                # This is convenient while training RNNs
                
                probas, logits = model(features)
                loss, _, _ = loss_func(self.option, logits, labels, probas, sensitive, self.penalty)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def fb_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz):
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                sensitive = sensitive.to(DEVICE)
                _, logits = model(features)

                logits = logits.to(DEVICE)
                v = torch.randn(len(labels)).type(torch.DoubleTensor).to(DEVICE)
                
                group_idx = {}
                
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                    nc += v[group_idx[(y,z)]].sum().item()

                # print(logits)
                loss = weighted_loss(logits, labels, v)
                # if global_round == 1: print(loss)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc

    def fb2_update(self, model, global_round, learning_rate, local_epochs, optimizer, lbd, m_yz):
        # Set mode to train model
        model.train()
        epoch_loss = []
        nc = 0

        # set seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Set optimizer for the local updates
        if optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                        momentum=0.5) # 
        elif optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                        weight_decay=1e-4)
        for i in range(local_epochs):
            batch_loss = []
            for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                sensitive = sensitive.to(DEVICE)
                _, logits = model(features)

                v = torch.ones(len(labels)).type(torch.DoubleTensor)
                
                group_idx = {}
                for y, z in lbd:
                    group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                    v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                    nc += v[group_idx[(y,z)]].sum().item()

                loss = weighted_loss(logits, labels, v, False)

                optimizer.zero_grad()
                if not np.isnan(loss.item()): loss.backward()
                optimizer.step()

                if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                        global_round + 1, i, batch_idx * len(features),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # weight, loss
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc

    def uflfb_update(self, model, num_epochs, learning_rate, optimizer, alpha):
        if self.Z == 2:
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy = [], []

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) # 
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for round_ in tqdm(range(num_epochs)):
                if self.prn and (round_+1) % 50 == 0: print(f'\n | Global Training Round : {round_+1} |\n')

                model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.randn(len(labels)).type(torch.DoubleTensor).to(DEVICE)
                    
                    group_idx = {}
                    
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] * sum([m_yz[(y,z)] for z in range(self.Z)]) / m_yz[(y,z)]
                    loss = weighted_loss(logits.to(DEVICE), labels, v)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                loss_avg = sum(batch_loss)/len(batch_loss)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0

                model.eval()
                    # validation dataset inference
                acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference(model = model, train = True) 
                list_acc.append(acc)

                    
                if self.prn and (round_+1) % 10 == 0: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    acc_loss, fair_loss, "DP Disparity", self.disparity(n_yz)))
                    
                # update the lambda according to the paper -> see Section A.1 of FairBatch
                # works well! The real batch size would be slightly different from the setting
                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
                y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
                if y0_diff > y1_diff:
                    lbd[(0,0)] -= alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(1,0)] = 1 - lbd[(0,0)]
                    lbd[(0,1)] += alpha / (round_+1)
                    lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                    lbd[(1,1)] = 1 - lbd[(0,1)]
                else:
                    lbd[(0,0)] += alpha / (round_+1)
                    lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                    lbd[(0,1)] = 1 - lbd[(0,0)]
                    lbd[(1,0)] -= alpha / (round_+1)
                    lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                    lbd[(1,1)] = 1 - lbd[(1,0)]

                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                # if (round_+1) % 10 == 0:
                #     print(f' \nAvg Training Stats after {round_+1} global rounds:')
                #     print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                #         np.mean(np.array(train_loss)), 
                #         100*train_accuracy[-1], "DP Disparity", self.disparity(n_yz)))

        else:
            # new algorithm for demographic parity, add weights directly, signed gradient-based algorithm
            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Training
            train_loss, train_accuracy = [], []

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) # 
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            # the number of samples whose label is y and sensitive attribute is z
            m_yz, lbd = {}, {}
            for y in [0,1]:
                for z in range(self.Z):
                    m_yz[(y,z)] = ((self.train_dataset.y == y) & (self.train_dataset.sen == z)).sum()

            for y in [0,1]:
                for z in range(self.Z):
                    lbd[(y,z)] = (m_yz[(1,z)] + m_yz[(0,z)])/len(self.train_dataset)

            for round_ in tqdm(range(num_epochs)):
                if self.prn: print(f'\n | Global Training Round : {round_+1} |\n')

                model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)
                    
                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])

                    loss = weighted_loss(logits, labels, v, False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())

                loss_avg = sum(batch_loss) / len(batch_loss)
                train_loss.append(loss_avg)

                # Calculate avg training accuracy over all clients at every round
                list_acc = []
                # the number of samples which are assigned to class y and belong to the sensitive group z
                n_yz, loss_yz = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        n_yz[(y,z)] = 0
                        loss_yz[(y,z)] = 0

                model.eval()
                acc, loss, n_yz, acc_loss, fair_loss, loss_yz = self.inference(model = model, train = True) 
                list_acc.append(acc)

                if self.prn and (round_+1) % 10 == 0: print("Accuracy loss: %.2f | fairness loss %.2f | %s = %.2f" % (
                    acc_loss, fair_loss, "DP Disparity", self.disparity(n_yz)))
                    
                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                for z in range(self.Z):
                    if z == 0:
                        lbd[(0,z)] -= alpha / (round_ + 1) ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(self.Z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                    else:
                        lbd[(0,z)] += alpha / (round_ + 1) ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                        lbd[(0,z)] = lbd[(0,z)].item()
                        lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                        lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                
                train_accuracy.append(sum(list_acc)/len(list_acc))

                # print global training loss after every 'i' rounds
                # if (round_+1) % 10 == 0:
                #     print(f' \nAvg Training Stats after {round_+1} global rounds:')
                #     print("Training loss: %.2f | Training accuracy: %.2f%% | Training %s: %.4f" % (
                #         np.mean(np.array(train_loss)), 
                #         100*train_accuracy[-1], "DP Disparity", self.disparity(n_yz)))

        return model

    def local_fb(self, model, learning_rate, local_epochs, optimizer, alpha, lbd = None, m_yz = None):
        if self.Z == 2:
            # Set mode to train model
            epoch_loss = []
            nc = 0

            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) # 
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)
            
            if lbd == None:
                m_yz, lbd = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        m_yz[(y,z)] = ((self.dataset.y == y) & (self.dataset.sen == z)).sum()

                for y in [0,1]:
                    for z in range(self.Z):
                        lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for epoch in range(local_epochs):
                model.train()
                batch_loss = []
                for _, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)
                    
                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                        nc += v[group_idx[(y,z)]].sum().item()

                    loss = weighted_loss(logits, labels, v, False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

                model.eval()
                # validation dataset inference
                _, _, _, _, _, loss_yz = self.inference(model = model, train = True) 

                for y, z in loss_yz:
                    loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

                
                    y0_diff = loss_yz[(0,0)] - loss_yz[(0,1)]
                    y1_diff = loss_yz[(1,0)] - loss_yz[(1,1)]
                    if y0_diff > y1_diff:
                        lbd[(0,0)] -= alpha / (epoch+1)
                        lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                        lbd[(1,0)] = 1 - lbd[(0,0)]
                        lbd[(0,1)] += alpha / (epoch+1)
                        lbd[(0,1)] = min(max(0, lbd[(0,1)]), 1)
                        lbd[(1,1)] = 1 - lbd[(0,1)]
                    else:
                        lbd[(0,0)] += alpha / (epoch+1)
                        lbd[(0,0)] = min(max(0, lbd[(0,0)]), 1)
                        lbd[(0,1)] = 1 - lbd[(0,0)]
                        lbd[(1,0)] -= alpha / (epoch+1)
                        lbd[(1,0)] = min(max(0, lbd[(1,0)]), 1)
                        lbd[(1,1)] = 1 - lbd[(1,0)]
            # weight, loss
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc, lbd, m_yz

        else:
            epoch_loss = []
            nc = 0

            # set seed
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)

            # Set optimizer for the local updates
            if optimizer == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                                            momentum=0.5) # 
            elif optimizer == 'adam':
                optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                            weight_decay=1e-4)

            if lbd == None:
                m_yz, lbd = {}, {}
                for y in [0,1]:
                    for z in range(self.Z):
                        m_yz[(y,z)] = ((self.dataset.y == y) & (self.dataset.sen == z)).sum()

                for y in [0,1]:
                    for z in range(self.Z):
                        lbd[(y,z)] = m_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for i in range(local_epochs):
                batch_loss = []
                for batch_idx, (features, labels, sensitive) in enumerate(self.trainloader):
                    features, labels = features.to(DEVICE), labels.to(DEVICE).type(torch.LongTensor)
                    sensitive = sensitive.to(DEVICE)
                    _, logits = model(features)

                    v = torch.ones(len(labels)).type(torch.DoubleTensor)
                    
                    group_idx = {}
                    for y, z in lbd:
                        group_idx[(y,z)] = torch.where((labels == y) & (sensitive == z))[0]
                        v[group_idx[(y,z)]] = lbd[(y,z)] / (m_yz[(1,z)] + m_yz[(0,z)])
                        nc += v[group_idx[(y,z)]].sum().item()

                    loss = weighted_loss(logits, labels, v, False)

                    optimizer.zero_grad()
                    if not np.isnan(loss.item()): loss.backward()
                    optimizer.step()

                    if self.prn and (100. * batch_idx / len(self.trainloader)) % 50 == 0:
                        print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tBatch Loss: {:.6f}'.format(
                            global_round + 1, i, batch_idx * len(features),
                            len(self.trainloader.dataset),
                            100. * batch_idx / len(self.trainloader), loss.item()))
                    batch_loss.append(loss.item())
                epoch_loss.append(sum(batch_loss)/len(batch_loss))

            model.eval()
            # validation dataset inference
            _, _, _, _, _, loss_yz = self.inference(model = model, train = True) 

            for y, z in loss_yz:
                loss_yz[(y,z)] = loss_yz[(y,z)]/(m_yz[(0,z)] + m_yz[(1,z)])

            for z in range(self.Z):
                if z == 0:
                    lbd[(0,z)] -= alpha ** .5 * sum([(loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)]) for z in range(self.Z)])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]
                else:
                    lbd[(0,z)] += alpha ** .5 * (loss_yz[(0,0)] + loss_yz[(1,0)] - loss_yz[(0,z)] - loss_yz[(1,z)])
                    lbd[(0,z)] = lbd[(0,z)].item()
                    lbd[(0,z)] = max(0, min(lbd[(0,z)], 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset)))
                    lbd[(1,z)] = 2*(m_yz[(1,0)]+m_yz[(0,0)])/len(self.train_dataset) - lbd[(0,z)]

            # weight, loss
            return model.state_dict(), sum(epoch_loss) / len(epoch_loss), nc, lbd, m_yz

    def inference(self, model, train = False, bits = False, truem_yz = None):
        """ 
        Returns the inference accuracy, 
                                loss, 
                                N(sensitive group, pos), 
                                N(non-sensitive group, pos), 
                                N(sensitive group),
                                N(non-sensitive group),
                                acc_loss,
                                fair_loss
        """

        model.eval()
        loss, total, correct, fair_loss, acc_loss, num_batch = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        n_yz, loss_yz, m_yz, f_z = {}, {}, {}, {}
        for y in [0,1]:
            for z in range(self.Z):
                loss_yz[(y,z)] = 0
                n_yz[(y,z)] = 0
                m_yz[(y,z)] = 0

        dataset = self.validloader if not train else self.trainloader
        for _, (features, labels, sensitive) in enumerate(dataset):
            features, labels = features.to(DEVICE), labels.type(torch.LongTensor).to(DEVICE)
            sensitive = sensitive.type(torch.LongTensor).to(DEVICE)
            
            # Inference
            outputs, logits = model(features)
            outputs, logits = outputs.to(DEVICE), logits.to(DEVICE)

            # Prediction
            
            _, pred_labels = torch.max(outputs, 1)
            pred_labels = pred_labels.view(-1).to(DEVICE)
            bool_correct = torch.eq(pred_labels, labels)
            correct += torch.sum(bool_correct).item()
            total += len(labels)
            num_batch += 1

            group_boolean_idx = {}
            
            for yz in n_yz:
                group_boolean_idx[yz] = (labels == yz[0]) & (sensitive == yz[1])
                n_yz[yz] += torch.sum((pred_labels == yz[0]) & (sensitive == yz[1])).item()     
                m_yz[yz] += torch.sum((labels == yz[0]) & (sensitive == yz[1])).item()    
                
                if self.option in["FairBatch", "FB-Variant1"]:
                # the objective function have no lagrangian term

                    loss_yz_,_,_ = loss_func("standard", logits[group_boolean_idx[yz]].to(DEVICE), 
                                                    labels[group_boolean_idx[yz]].to(DEVICE), 
                                         outputs[group_boolean_idx[yz]].to(DEVICE), sensitive[group_boolean_idx[yz]].to(DEVICE), 
                                         self.penalty)
                    loss_yz[yz] += loss_yz_
            
            batch_loss, batch_acc_loss, batch_fair_loss = loss_func(self.option, logits, 
                                                        labels, outputs, sensitive, self.penalty)
            loss, acc_loss, fair_loss = (loss + batch_loss.item(), 
                                         acc_loss + batch_acc_loss.item(), 
                                         fair_loss + batch_fair_loss.item())
        accuracy = correct/total
        if self.option in ["FairBatch", "FB-Variant1"]:
            for z in range(1, self.Z):
                f_z[z] = - loss_yz[(0,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(1,0)]/(truem_yz[(0,0)] + truem_yz[(1,0)]) + loss_yz[(0,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)]) - loss_yz[(1,z)]/(truem_yz[(0,z)] + truem_yz[(1,z)]) 
            if bits: 
                bins = np.linspace(-2, 2, 2**bits // (self.Z - 1))
                for z in range(1, self.Z):
                    f_z[z] = bins[np.digitize(f_z[z].item(), bins)-1]
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, f_z
        else:
            return accuracy, loss, n_yz, acc_loss / num_batch, fair_loss / num_batch, None

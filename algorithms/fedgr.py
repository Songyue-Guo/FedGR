'''
@ Author: Yue 
@ Date: 2022-06-22 02:31:27
@ LastEditTime: 2022-09-27 09:05:58
@ FilePath: /gsy/FedGR/algorithms/fedgr.py
@ Description: 
@ Email: 21S151140@stu.hit.edu.cn , Tel: +86-13184023012 
@ Copyright (c) 2022 by HITSZ / Songyue Guo, All Rights Reserved. 
'''
'''
@                   江城子 . 程序员之歌
@ 
@               十年生死两茫茫，写程序，到天亮。
@                   千行代码，Bug何处藏。
@               纵使上线又怎样，朝令改，夕断肠。
@ 
@               领导每天新想法，天天改，日日忙。
@                   相顾无言，惟有泪千行。
@               每晚灯火阑珊处，夜难寐，加班狂。
@ 
'''
from ast import arg
import copy
from http import client
from telnetlib import GA
from unittest import loader
from cv2 import log, mean
from matplotlib.pyplot import axes
import numpy as np
from pyrsistent import m, s

import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from utils import Averager
from utils import count_acc
from utils import append_to_logs
from utils import format_logs
from utils import setup_seed
from tools import construct_dataloaders
from tools import construct_optimizer

setup_seed()

class FedGR():
    '''/******* 
     @ description: 
     @ param {*} self
     @ param {Dictionary} csets  :   client_dataset {client:(train_set,test_set)}
     @ param {*} gset   :   global test sets
     @ param {Model} model
     @ param {*} args
     @ return {*}
     */'''
    def __init__(self,csets,gset,model,args) :
        #initialization 
        self.csets = csets
        self.gset = gset
        self.model = model
        self.args = args

        self.clients = list(csets.keys())
        
        #construct datasets
        self.train_loaders, self.test_loaders, self.global_test_loader = \
            construct_dataloaders(
                self.clients, self.csets, self.gset, self.args
            ) 

        #client_counts
        self.client_counts = self.get_client_dists(
            csets = self.csets,
            args = self.args
        )

        self.logs = {
            "ROUNDS": [],
            "LOSSES": [],
            "GLO_TACCS": [],
            "LOCAL_TACCS": [],
            "LOCAL_MF1S": [],
        }

    '''/******* 
     @ description: calculate each class number in each client, store in client_cnts
     @ param {*} csets : client sets
     @ param {*} args  :  train args
     @ return {*} client _cnts : 
     */'''
    def get_client_dists(self,csets,args):
        client_cnts = {}
        for client in csets.keys():
            client_class_info = csets[client]
            # calculate each class number in each client, 
            # client_class_info is the detail sample index of each class in client
            # client_class_info[0] is the train_data
            counts_per_client = [
                np.sum(client_class_info[0].ys == c) for c in range(args.n_classes)
            ]

            counts_per_client = torch.FloatTensor(np.array(counts_per_client))
            client_cnts[client] = counts_per_client 

        return client_cnts

    def train(self):
        #Training
        setup_seed()
        for r in range(1, self.args.max_round + 1):
            n_sam_clients = int(self.args.fed_ratio * len(self.clients))
            participate_client = np.random.choice(self.clients,n_sam_clients,replace=False)

            local_models = {}

            avg_loss = Averager()
            all_per_accs = []
            all_per_mf1s = []
            classifier_w_list = []

            weights = {}
            total_cnts = 0.0
            for client in participate_client:
                self.set_model(client,local_models)
                classifier_w_list.append(local_models[client].classifier.weight)
            assert len(local_models) == len(participate_client) , "load client classifier error!"
            for client in participate_client:
                #counts store the detail number of class
                counts = self.client_counts[client]
                dist = counts / counts.sum() 
              
                local_model, per_accs, per_mf1, loss = \
                    self.local_update(
                        client = client,
                        r =r,
                        model = copy.deepcopy(self.model),
                        args = self.args,
                        train_loader = self.train_loaders[client],
                        test_loader = self.test_loaders[client],
                        dist = dist,
                        classifier_w_list=classifier_w_list
                    ) 

                local_models[client] = copy.deepcopy(local_model)
                avg_loss.add(loss)

                all_per_accs.append(per_accs)
                all_per_mf1s.append(per_mf1)
                weights[client] = counts.sum()
                total_cnts +=  counts.sum()  
            # calculate the weight of federated process W = (1 / K) * weight * W_i
            weights = {k:v / total_cnts for k,v in weights.items() }

            train_loss = avg_loss.item()
            #calculate the average acc or mf1 of all client in different rounds
            per_accs = list(np.array(all_per_accs).mean(axis = 0))
            per_mf1s = list(np.array(all_per_mf1s).mean(axis = 0))

            self.global_update(
                r = r,
                global_model = self.model,
                all_client_model = local_models,
                weights = weights
            )

            if r % self.args.test_round == 0:
                global_test_acc , _ = self.test(
                    model = self.model,
                    test_loader = self.global_test_loader
                )

                # add to log
                self.logs["ROUNDS"].append(r)
                self.logs["LOSSES"].append(train_loss)
                self.logs["GLO_TACCS"].append(global_test_acc)
                self.logs["LOCAL_TACCS"].extend(per_accs)
                self.logs["LOCAL_MF1S"].extend(per_mf1s)
                print("[R:{}] [Ls:{}] [TAc:{}] [PAc:{},{}] [PF1:{},{}]".format(
                        r, train_loss, global_test_acc, per_accs[0], per_accs[-1],
                        per_mf1s[0], per_mf1s[-1]
                ))    


    def local_update(self,client,r,model,args,train_loader,test_loader,dist,w_list):
        '''/******* 
        @ description           :      client local update process
        @ param {*} self
        @ param {int} r         :      communication round
        @ param {Model} model   :      local model
        @ param {*} args        :      args
        @ param {dataloader} train_loader:  local train loader
        @ param {dataloader} test_loader:   local test loader
        @ param {float tensor} dist    :    each class percentage  
        @ return {*}        :      model, per_accs, per_mf1s, loss
        */'''
        lr = args.lr
        print(w_list[0].shape)
        optimizer = construct_optimizer(model,lr,args)

        if args.local_steps is not None:
            n_total_bs = args.local_steps
        elif args.local_epochs is not None:
            n_total_bs = max(
                int(args.local_epochs * len(train_loader)) , 5
            )
        else:
            raise ValueError(
                "local_steps or local_epochs must not be None together"
            )     

        model.train()

        avg_loss = Averager()
        per_accs = []
        per_mf1s = []

        if args.cuda:
            dist = dist.cuda()
               
        padding_tensor = torch.FloatTensor([[4] *10])
        padding_tensor = padding_tensor.cuda()
        # 计算各种类比例 compute the ratio of each in-client class  
        dist = dist/dist.max()   # Normalization in client ? 
        dist = torch.where(dist==0,padding_tensor,dist)
        #dist = dist * (1.0 - self.args.alpha) + self.args.alpha
        #Calculate the gamma of each classes in client
        Gamma = 1 / dist 
       

        for i in range(1,args.local_epochs):
            load_iter = iter(train_loader)
            Gamma = Gamma / Gamma.max()
            #Gamma = Gamma + dist 
            Gamma = Gamma.reshape((1, -1))
            # iterate each mini-batch 
            for t in range(1,len(train_loader)+1):
                #before-train test F1 & after-train test F1
                if t in [1, (len(train_loader))]:
                    per_acc,per_mf1 = self.test(
                        model = model,
                       test_loader = test_loader
                    )
    
                    per_accs.append(per_acc)
                    per_mf1s.append(per_mf1)
                
                model.train()

                try:
                    batch_x,batch_y = load_iter.next()
                except Exception:
                    print("*"*50,"Train load batch Error","*"*50,)
                    #load_iter = iter()

                if args.cuda == True:
                    batch_x = batch_x.cuda()    
                    batch_y = batch_y.cuda() 
                    
                hs,_ = model(batch_x)
                ws = model.classifier.weight
                if args.cuda:
                    Gamma = Gamma.cuda()

                #calculate the logits of network 
                #logits = dist * hs.mm(ws.transpose(0, 1)) 
                logits = Gamma * hs.mm(ws.transpose(0, 1)) 
                criterion = nn.CrossEntropyLoss()
                local_loss = criterion(logits,batch_y)
 
                #TODO  Gravitation  Regulation Loss
                client_weight = model.classifier.weight
                similiar_clients = set()
                repulsion_reg_loss = 0
                attarction_reg_loss = 0
                class_dic = self.csets_class_info()
                with torch.no_grad:
                    copy_classifier_weight = copy.deepcopy(w_list)
                #
                for yi in range(args.n_classes):
                    if client in class_dic[str(yi)]:
                        w_yi_T = client_weight.T[yi][None,:]
                        #for each class in client to calculate the in-client positive pulling
                        positive_pull_in_client += w_yi_T.mm(w_yi_T.T)
                        for other_client in class_dic[str(yi)]:
                            if other_client != client:
                                similiar_clients.add(other_client)
                
                    negative_push_cross_client += client_weight.T[yi].mm(client_weight[yi]) #shape (10,10)
                    
                    positive_pull_cross_client = 0
                #TODO 
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    model.parameters(),args.max_grad_norm
                )
                optimizer.step()

                avg_loss.add(loss.item())

            loss = avg_loss.item()
        return model, per_accs, per_mf1s, loss


    def test(self, model,test_loader):
        '''/******* 
         @ description: 
         @ param {*} self
         @ param {*} model
         @ param {*} test_loader
         @ return {*}
         */'''
        model.eval()

        acc_avg = Averager()

        preds = []    
        reals = []

        with torch.no_grad():
            for i, (batch_x,batch_y) in enumerate(test_loader):
                if self.args.cuda:
                    batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                _,logits = model(batch_x)

                acc = count_acc(logits,batch_y)
                acc_avg.add(acc)

                preds.append(np.argmax(logits.cpu().detach().numpy(), axis=1))
                reals.append(batch_y.cpu().detach().numpy())
            
            preds = np.concatenate(preds, axis=0)
            reals = np.concatenate(reals, axis=0)

            acc = acc_avg.item()
        
        #MACRO F1
        mf1 = f1_score(y_true=reals,y_pred=preds,average="micro")

        return acc, mf1
    
    def save_logs(self,fpath):
        all_logs_str =[]
        all_logs_str.append(str(self.logs))

        logs_str = format_logs(self.logs)
        all_logs_str.extend(logs_str)

        append_to_logs(fpath,all_logs_str)

    #TODO finish global regulation 
    
    def global_update(self,r,global_model,all_client_model,weights):
        '''/******* 
     @ description: 
     @ param {*} self
     @ param {*} r
     @ param {*} global_model
     @ param {*} all_client_model
     @ param {*} weights
     @ return {*}
     */'''
        mean_state_dict = {}

        for name, param in global_model.state_dict().items():
            vs = []
            for client in all_client_model.keys():
                w = weights[client]
                vs.append(w * all_client_model[client].state_dict()[name])
            vs = torch.stack(vs, dim=0)

            try:
                mean_value = vs.sum(dim=0)
            except Exception:
                # for BN's cnt
                mean_value = (1.0 * vs).sum(dim=0).long()
            mean_state_dict[name] = mean_value

        global_model.load_state_dict(mean_state_dict, strict=False)

    def set_model(self,client,local_models):
        local_models[client] = copy.deepcopy(self.model)

    def csets_class_info(self):
        #class_dic store class is contained by client 
        class_dic = {}
        for client in self.csets.keys():
            
            counts_per_client = [
                np.sum(self.csets[client][0].ys == c) for c in range(self.args.n_classes)
            ]
            
            counts_per_client = torch.FloatTensor(counts_per_client)
            counts_per_client = counts_per_client > 0 
            counts_per_client = counts_per_client.numpy()
            for c in range(self.args.n_classes):
                class_dic[str(c)]= []
                if counts_per_client[c] == True:
                    class_dic[str(c)].append(client)
        
        return class_dic    

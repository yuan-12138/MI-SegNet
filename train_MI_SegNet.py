#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:22:04 2020

@author: yuanbi
"""

import torch
import torch.optim as  optim
from torchvision.utils import save_image
from MI_SegNet import Mine, Recon_encoder_fusion, Recon_decoder_fusion, Seg_encoder_DNL,Seg_decoder_DNL
from utils import prepare_test_data, prepare_data_pair_adv
import os
from os import listdir
import shutil
import numpy as np
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt


cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

result_dir = './MI_Netresult'
save_dir = './ckpt'
batch_size = 8
epochs = 200
num_worker = 0

lr = 1e-5
z_dim = 32
input_dim = 256*256
input_channel = 1
init_feature = 32

test_loss_record = []
test_seg_loss_record = []
training_loss_record = []

NUM_DEMO=['Training']
NUM_TEST_DEMO=['ValS']

pre_trained=False

demo_path="./MI_SegNet_dataset"
test_demo_path="./MI_SegNet_dataset"
pretrain_path="./ckpt/trained_model/model_best.pth"

Mine = Mine(z_dim).to(device)
optimizer_mine = optim.Adam(Mine.parameters(), lr=1e-4)
ma_rate=0.001

Rec_encoder = Recon_encoder_fusion(input_channel,z_dim,16).to(device)
optimizer_rec_en = optim.Adam(Rec_encoder.parameters(), lr=lr)

Rec_decoder = Recon_decoder_fusion(input_channel,z_dim,16).to(device)
optimizer_rec_de = optim.Adam(Rec_decoder.parameters(), lr=lr)

Seg_encoder = Seg_encoder_DNL(input_channel,z_dim,32).to(device)
optimizer_seg_en = optim.Adam(Seg_encoder.parameters(), lr=lr)

Seg_decoder = Seg_decoder_DNL(input_channel,z_dim,32).to(device)
optimizer_seg_de = optim.Adam(Seg_decoder.parameters(), lr=lr)

if pre_trained:
    pretrain_para=torch.load(pretrain_path,map_location=device)
    Mine.load_state_dict(pretrain_para['state_dict_mine'])
    Rec_encoder.load_state_dict(pretrain_para['state_dict_rec_en'])
    Rec_decoder.load_state_dict(pretrain_para['state_dict_rec_de'])
    Seg_encoder.load_state_dict(pretrain_para['state_dict_seg_en'])
    Seg_decoder.load_state_dict(pretrain_para['state_dict_seg_de'])
    
    optimizer_mine.load_state_dict(pretrain_para['optimizer_mine'])
    optimizer_rec_en.load_state_dict(pretrain_para['optimizer_rec_en'])
    optimizer_rec_de.load_state_dict(pretrain_para['optimizer_rec_de'])
    optimizer_seg_en.load_state_dict(pretrain_para['optimizer_seg_en'])
    optimizer_seg_de.load_state_dict(pretrain_para['optimizer_seg_de'])
    del pretrain_para
    torch.cuda.empty_cache()

transform_image=transforms.Normalize(0.5,0.5)

def save_checkpoint(state, is_best, outdir):

	if not os.path.exists(outdir):
		os.makedirs(outdir)

	checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
	best_file = os.path.join(outdir, 'model_best.pth')
	torch.save(state, checkpoint_file)
	if is_best:
		shutil.copyfile(checkpoint_file, best_file)

def reset_grad():
    optimizer_mine.zero_grad()
    optimizer_rec_en.zero_grad()
    optimizer_rec_de.zero_grad()
    optimizer_seg_en.zero_grad()
    optimizer_seg_de.zero_grad()


def update_Seg(inputs, labels, train):
    z, enc_1, enc_2, enc_3, enc_4 = Seg_encoder(inputs)
    Seg_results = Seg_decoder((z, enc_1, enc_2, enc_3, enc_4))
    
    smooth = 2
    index = (2*torch.sum(Seg_results*labels)+smooth)/(torch.sum(Seg_results)+torch.sum(labels)+smooth)
    loss_dice = (1-index)
    
    loss_bce = F.binary_cross_entropy(Seg_results, labels, reduction='mean')
    
    loss = loss_bce + loss_dice
    
    if train:
        loss.backward()
        
        optimizer_seg_en.step()
        optimizer_seg_de.step()
        
        reset_grad()
    return loss, Seg_results

def update_Rec(inputs,GT,train):
    z_a, _, _, _, _ = Seg_encoder(inputs)
    z_d = Rec_encoder(inputs)
    
    Recon_result = Rec_decoder(z_a,z_d)
    
    rec_loss = F.l1_loss(torch.squeeze(Recon_result), torch.squeeze(GT), reduction='mean')
    
    if train:
        rec_loss.backward()
        
        optimizer_rec_en.step()
        optimizer_rec_de.step()
        optimizer_seg_en.step()
        
        reset_grad()
    
    return rec_loss, Recon_result

def update_Rec_Adv(inputs_1,inputs_2,inputs_12,train):
    z_a, _, _, _, _ = Seg_encoder(inputs_2)
    z_d = Rec_encoder(inputs_1)
    
    Recon_result = Rec_decoder(z_a,z_d)
    
    rec_loss = F.l1_loss(torch.squeeze(Recon_result), torch.squeeze(inputs_12), reduction='mean')
    
    if train:
        rec_loss.backward()
        
        optimizer_rec_en.step()
        optimizer_rec_de.step()
        optimizer_seg_en.step()
        
        reset_grad()
    
    return rec_loss, Recon_result

def update_MI(inputs,train):
    z_a, _, _, _, _ = Seg_encoder(inputs)
    
    z_d = Rec_encoder(inputs)
       
    z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(device))
    
    mutual_loss,_,_ = mi_estimator(z_a, z_d, z_d_shuffle)
    
    mutual_loss=F.elu(mutual_loss)
    
    if train:
        mutual_loss.backward()
        
        optimizer_rec_en.step()
        optimizer_seg_en.step()
        
        reset_grad()
    
    return mutual_loss

def mi_estimator(x, y, y_):
    joint, marginal = Mine(x, y), Mine(x, y_)
    return torch.mean(joint) - torch.log(torch.mean(torch.exp(marginal))), joint, marginal

def learn_mine(inputs, ma_rate=0.001):
    
    with torch.no_grad():
        z_a, _, _, _, _ = Seg_encoder(inputs)
        z_d = Rec_encoder(inputs)
       
        z_d_shuffle = torch.index_select(z_d, 0, torch.randperm(z_d.shape[0]).to(device))
    
    et = torch.mean(torch.exp(Mine(z_a, z_d_shuffle)))
    if Mine.ma_et is None:
        Mine.ma_et = et.detach().item()
    Mine.ma_et += ma_rate * (et.detach().item() - Mine.ma_et)
    mutual_information = torch.mean(Mine(z_a, z_d)) - torch.log(et) * et.detach() /Mine.ma_et
        
    loss = -mutual_information
    
    loss.backward()
    optimizer_mine.step()
    reset_grad()
    
    return mutual_information


def train():
    

    start_epoch = 0
    best_test_loss = np.finfo('f').max
    
    length=[0]
    files_training_img=[]
    files_training_label=[]
    for i in NUM_DEMO:
        f_img = listdir(os.path.join(demo_path,str(i),"img"))
        f_img.sort()
        files_training_img.append(f_img)
        
        f_label = listdir(os.path.join(demo_path,str(i),"label"))
        f_label.sort()
        files_training_label.append(f_label)
        
        length.append(len(f_img))
    length=np.array(length)
    print(length)
    
    length_test=[0]
    files_test_img=[]
    files_test_label=[]
    for i in NUM_TEST_DEMO:
        f_img = listdir(os.path.join(test_demo_path,str(i),"img"))
        f_img.sort()
        
        f_label = listdir(os.path.join(test_demo_path,str(i),"label"))
        f_label.sort()

        files_test_img.append(f_img)
        files_test_label.append(f_label)
        length_test.append(len(f_img))
    length_test=np.array(length_test)
    
    trainloader = prepare_data_pair_adv(NUM_DEMO, length, files_training_img, files_training_label, demo_path, batch_size, num_worker)
    testloader = prepare_test_data(NUM_TEST_DEMO, length_test, files_test_img, files_test_label, test_demo_path, batch_size, num_worker)
    print(len(testloader.dataset))
    learn_mi_loss_2=0.0
	# training
    for epoch in range(start_epoch, epochs):
        train_avg_seg_loss=0
        for i, data in enumerate(trainloader):
            
            inputs_1=data[0].float().to(device)
            inputs_2=data[1].float().to(device)
            inputs_12=data[2].float().to(device).view(-1,1,256,256) # domain 1 anatomy 2
            inputs_21=data[3].float().to(device).view(-1,1,256,256) # domain 2 anatomy 1
            label_1=data[4].view(-1,1,256,256).float().to(device)
            label_2=data[5].view(-1,1,256,256).float().to(device)
            
            inputs_1_trans = transform_image(inputs_1).view(-1,1,256,256)
            inputs_2_trans = transform_image(inputs_2).view(-1,1,256,256)

            Seg_loss_1, Seg_results_1 = update_Seg(inputs_1_trans, label_1, True)
            Seg_loss_2, Seg_results_2 = update_Seg(inputs_2_trans, label_2, True)
            Seg_loss = Seg_loss_1 + Seg_loss_2
            
            recon_loss_1, Rec_results_1 = update_Rec(inputs_1_trans, inputs_1, True)
            recon_loss_2, Rec_results_2 = update_Rec(inputs_2_trans, inputs_2, True)
            recon_loss = recon_loss_1 + recon_loss_2
            
            rec_adv_loss_1,Rec_results_12=update_Rec_Adv(inputs_1_trans,inputs_2_trans,inputs_12,True)
            rec_adv_loss_2,Rec_results_21=update_Rec_Adv(inputs_2_trans,inputs_1_trans,inputs_21,True)
            rec_adv_loss = rec_adv_loss_1 + rec_adv_loss_2
            
            mi_loss_1 = update_MI(inputs_1_trans, True)
            mi_loss_2 = update_MI(inputs_2_trans, True)
            mi_loss = mi_loss_1 + mi_loss_2

            train_avg_seg_loss+=Seg_loss*0.5
            for _ in range(5):
                learn_mi_loss_1=learn_mine(inputs_1_trans)
                learn_mi_loss_2=learn_mine(inputs_2_trans)
                
            loss = Seg_loss + recon_loss + mi_loss + rec_adv_loss
			# print statistics every 100 batches
            if (i + 1) % 10 == 0:
                print("\r Epoch[{}/{}], Step [{}/{}], Loss: {:.4f}, Mutual loss: {:.4f} Recon loss {:.4f} Seg loss {:.4f} Rec Adv loss {:.4f}".format(epoch + 1, epochs, i + 1, len(trainloader), 
                    loss.item(), mi_loss.item(), recon_loss.item(), Seg_loss.item(), rec_adv_loss.item()), end='')
                if torch.any(torch.isnan(loss)):
                    save_checkpoint({
    					'epoch': epoch,
    					'best_test_loss': best_test_loss,
    					'state_dict_rec_en': Rec_encoder.state_dict(),
                        'state_dict_rec_de': Rec_decoder.state_dict(),
                        'state_dict_seg_en': Seg_encoder.state_dict(),
                        'state_dict_seg_de': Seg_decoder.state_dict(),
                        'state_dict_mine': Mine.state_dict(),
    					'optimizer_rec_en': optimizer_rec_en.state_dict(),
                        'optimizer_rec_de': optimizer_rec_de.state_dict(),
                        'optimizer_seg_en': optimizer_seg_en.state_dict(),
                        'optimizer_seg_de': optimizer_seg_de.state_dict(),
                        'optimizer_mine': optimizer_mine.state_dict(),
    				}, False, save_dir)
                    return

            if i == 0:
                x_concat = torch.cat([inputs_1.view(-1, 1, 256, 256), Seg_results_1.view(-1, 1, 256, 256), 
                                      Rec_results_1.view(-1, 1, 256, 256), inputs_2.view(-1, 1, 256, 256), 
                                      Seg_results_2.view(-1, 1, 256, 256), Rec_results_2.view(-1, 1, 256, 256)], dim=3)
                save_image(x_concat, ("./%s/reconstructed-%d.png" % (result_dir, epoch + 1)),nrow=4)

    		# testing
            if (i + 1) % 50 == 0:
                
                test_avg_seg_loss=0.0
                test_avg_recon_loss=0.0
                test_avg_loss = 0.0
                test_avg_mutual_loss = 0.0
                with torch.no_grad():
                    
                    for idx, test_data in enumerate(testloader):
    					# get the inputs; data is a list of [inputs, labels]
                        
                        test_inputs=test_data[0].float().to(device)
                        test_labels=test_data[1].view(-1,1,256,256).float().to(device)
                        
                        test_inputs_trans=transform_image(test_inputs).view(-1,1,256,256)
                        # test_inputs_trans=test_inputs.view(-1,1,256,256)
                        
    					# forward
                        Seg_loss_test, Seg_result = update_Seg(test_inputs_trans, test_labels, False)
                        recon_loss, Rec_results = update_Rec(test_inputs_trans, test_inputs, False)
                        mi_loss_1 = update_MI(test_inputs_trans, False)
                        # mi_loss_2 = update_MI(test_labels, False)
                        mutual_loss = mi_loss_1
                        # Disentangle_loss = update_Ad_Seg(test_inputs, test_labels, False)
                        test_loss = Seg_loss_test+recon_loss+mutual_loss
                        
                        test_avg_seg_loss += Seg_loss_test
                        test_avg_recon_loss += recon_loss
                        test_avg_loss += test_loss
                        test_avg_mutual_loss += torch.abs(mutual_loss)
    
                    test_avg_seg_loss /= (idx+1)
                    test_avg_recon_loss /= (idx+1)
                    test_avg_loss /= (idx+1)
                    test_avg_mutual_loss /= (idx+1)
                    train_avg_seg_loss /= 50
                    test_loss_record.append(test_avg_loss.item())
                    test_seg_loss_record.append(test_avg_seg_loss.item())
                    training_loss_record.append(train_avg_seg_loss.item())
                    train_avg_seg_loss=0
    
    				# save model
                    is_best = test_avg_loss < best_test_loss
                    best_test_loss = min(test_avg_loss, best_test_loss)
                    save_checkpoint({
    					'epoch': epoch,
    					'best_test_loss': best_test_loss,
    					'state_dict_rec_en': Rec_encoder.state_dict(),
                        'state_dict_rec_de': Rec_decoder.state_dict(),
                        'state_dict_seg_en': Seg_encoder.state_dict(),
                        'state_dict_seg_de': Seg_decoder.state_dict(),
                        'state_dict_mine': Mine.state_dict(),
    					'optimizer_rec_en': optimizer_rec_en.state_dict(),
                        'optimizer_rec_de': optimizer_rec_de.state_dict(),
                        'optimizer_seg_en': optimizer_seg_en.state_dict(),
                        'optimizer_seg_de': optimizer_seg_de.state_dict(),
                        'optimizer_mine': optimizer_mine.state_dict(),
    				}, is_best, save_dir)
                    np.savetxt(os.path.join(save_dir,'training_loss_record.txt'), training_loss_record, fmt='%f')
                    np.savetxt(os.path.join(save_dir,'test_loss_record.txt'), test_loss_record, fmt='%f')
                    np.savetxt(os.path.join(save_dir,'test_seg_loss_record.txt'), test_seg_loss_record, fmt='%f')
                if len(test_loss_record)==2:
                    plt.clf()
                    plt.ylabel('test_loss')
                    plt.xlabel('Training progress #')
                    plt.plot([len(test_loss_record)-1,len(test_loss_record)], [test_loss_record[-2],test_loss_record[-1]],'b', alpha=0.3)
                    plt.plot([len(training_loss_record)-1,len(training_loss_record)], [training_loss_record[-2],training_loss_record[-1]],'g', alpha=0.3)
                    plt.plot([len(test_seg_loss_record)-1,len(test_seg_loss_record)], [test_seg_loss_record[-2],test_seg_loss_record[-1]],'y', alpha=0.3)
                    plt.pause(0.05)
                elif len(test_loss_record)>2:
                    plt.plot([len(test_loss_record)-1,len(test_loss_record)], [test_loss_record[-2],test_loss_record[-1]],'b', alpha=0.3)
                    plt.plot([len(training_loss_record)-1,len(training_loss_record)], [training_loss_record[-2],training_loss_record[-1]],'g', alpha=0.3)
                    plt.plot([len(test_seg_loss_record)-1,len(test_seg_loss_record)], [test_seg_loss_record[-2],test_seg_loss_record[-1]],'y', alpha=0.3)
                    plt.pause(0.05)



if __name__ == '__main__':
	train()

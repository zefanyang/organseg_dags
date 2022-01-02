#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @Time  : 7/2/2021 9:49 AM
# @Author: yzf
"""UNet with edge skip connection, RFP-Head, and deep supervision"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from .unet import Encoder, Decoder, DoubleConv

class DeepSup(nn.Module):
    def __init__(self, in_ch, out_ch, scale_factor):
        super().__init__()
        self.dsup = nn.Sequential(nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
                                  nn.Upsample(scale_factor=scale_factor, mode='trilinear'))
    def forward(self, x):
        return self.dsup(x)

class EGModule(nn.Module):
    def __init__(self, init_ch):
        super(EGModule, self).__init__()

        # 3*3*3 -> 1*1*1 convolution to concentrate the channel
        self.h_conv = nn.Sequential(
            nn.Conv3d(init_ch * 16, init_ch * 2, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU()
        )

        self.e_conv = nn.Sequential(
            nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
            nn.Conv3d(init_ch * 2, init_ch * 2, 3, 1, 1),
            nn.BatchNorm3d(init_ch * 2),
            nn.ReLU(),
        )
        self.out_conv = nn.Conv3d(init_ch*2, 1, 1)

    def forward(self, l_feat, h_feat):
        h_feat = self.h_conv(h_feat)
        h_feat = F.interpolate(h_feat, scale_factor=8, mode='trilinear')

        # add ReLU after addition? Show mild performance drop. The ReLU has no effect.
        feat = h_feat + l_feat
        edge_feat = self.e_conv(feat)
        edge_score = self.out_conv(edge_feat)
        edge_score = F.interpolate(edge_score, scale_factor=2, mode='trilinear')
        return edge_feat, edge_score

def edge_fusion(skip_feat, edge_feat):
    edge_feat = F.interpolate(edge_feat, skip_feat.size()[2:], mode='trilinear')
    return torch.cat((edge_feat, skip_feat), dim=1)

class RFP_UAGs(nn.Module):
    def __init__(self, in_ch, num_neigh='four'):
        super().__init__()
        self.dag_list = None
        if num_neigh == 'four':
            self.dag_list = nn.ModuleList([UAG_RNN_4Neigh(in_ch) for _ in range(64//16)])  # hard-coding '64//8'
        elif num_neigh == 'eight':
            self.dag_list = nn.ModuleList([UAG_RNN_8Neigh(in_ch) for _ in range(64//16)])  # hard-coding '64//8'

    def forward(self, x):
        d = x.shape[-1]
        x_hid = []
        x_adp = x
        
        for i in range(d):
            hid = self.dag_list[i](x_adp[..., i])
            x_hid.append(hid.unsqueeze(-1))
        x_hid = torch.cat(x_hid, dim=-1)

        return x_adp + x_hid
    
class UNetL9DeepSupFullScheme(nn.Module):
    def __init__(self, in_ch, out_ch, num_neigh='four', interpolate=True, init_ch=16, conv_layer_order='cbr'):
        super(UNetL9DeepSupFullScheme, self).__init__()

        self.no_class = out_ch

        ## Encoder
        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False, conv_layer_order=conv_layer_order),
            Encoder(init_ch, 2 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(2 * init_ch, 4 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(4 * init_ch, 8 * init_ch, conv_layer_order=conv_layer_order),
            Encoder(8 * init_ch, 16 * init_ch, conv_layer_order=conv_layer_order)
        ])

        ## Decoder
        self.decoders = nn.ModuleList([
            Decoder(8*init_ch+16*init_ch+32, 8*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(4*init_ch+8*init_ch+32, 4*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(2*init_ch+4*init_ch+32, 2*init_ch, interpolate, conv_layer_order=conv_layer_order),
            Decoder(init_ch+2*init_ch+32, init_ch, interpolate, conv_layer_order=conv_layer_order)
        ])

        # deep supervision
        self.deep_sup4 = DeepSup(8 * init_ch, out_ch=self.no_class, scale_factor=8)
        self.deep_sup3 = DeepSup(4 * init_ch, out_ch=self.no_class, scale_factor=4)
        self.deep_sup2 = DeepSup(2 * init_ch, out_ch=self.no_class, scale_factor=2)
        self.deep_sup1 = nn.Conv3d(init_ch, self.no_class, kernel_size=1)

        ## Edge detection
        self.edge_module = EGModule(init_ch)

        self.final_conv = nn.Sequential(nn.Dropout3d(0.1, False),
                                        nn.Conv3d(self.no_class * 4, self.no_class, 1))

        ## RFP-Head
        trans_ch = 16 * init_ch // 2

        self.adapt = nn.Sequential(
            nn.Conv3d(16*init_ch, trans_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(trans_ch),
            nn.ReLU(),
        )
        self.rfp = RFP_UAGs(in_ch=trans_ch, num_neigh=num_neigh)
        self.rfp_fnl_conv = nn.Sequential(
            nn.Conv3d(trans_ch, trans_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(trans_ch),
            nn.ReLU(),
            nn.Conv3d(trans_ch, self.no_class, 1)
        )

        # Out conv
        self.comb_fnl_conv = nn.Conv3d(self.no_class * 2, self.no_class, 1)

    def forward(self, x):
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        enc4 = self.encoders[3](enc3)
        mid = self.encoders[4](enc4)
        encoders_features = [enc4, enc3, enc2, enc1]

        # Edge detection
        edge_feat, edge_score = self.edge_module(enc2, mid)

        # Edge skip-connections
        skip4 = edge_fusion(enc4, edge_feat)
        skip3 = edge_fusion(enc3, edge_feat)
        skip2 = edge_fusion(enc2, edge_feat)
        skip1 = edge_fusion(enc1, edge_feat)

        dec4 = self.decoders[0](skip4, mid)
        dec3 = self.decoders[1](skip3, dec4)
        dec2 = self.decoders[2](skip2, dec3)
        dec1 = self.decoders[3](skip1, dec2)

        dsup4 = self.deep_sup4(dec4)
        dsup3 = self.deep_sup3(dec3)
        dsup2 = self.deep_sup2(dec2)
        dsup1 = self.deep_sup1(dec1)

        seg_score = self.final_conv(torch.cat((dsup4, dsup3, dsup2, dsup1), dim=1))

        # RFP-Head
        mid_adapt = self.adapt(mid)
        ehn_mid = self.rfp(mid_adapt)
        rfp_seg_score = self.rfp_fnl_conv(ehn_mid)
        rfp_seg_score = F.upsample(rfp_seg_score, scale_factor=16, mode='trilinear', align_corners=True)

        comb_seg_score = self.comb_fnl_conv(torch.cat((seg_score, rfp_seg_score), 1))

        return seg_score, comb_seg_score, edge_score


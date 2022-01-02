from .unet import *
from .utils_graphical_model import UAG_RNN_4Neigh
from .utils_graphical_model import UAG_RNN_8Neigh

class UNetDAGs(nn.Module):
    def __init__(self, in_ch, out_ch, init_ch=16, num_neigh='four'):
        super(UNetDAGs, self).__init__()

        self.no_class = out_ch
       
        self.encoders = nn.ModuleList([
            Encoder(in_ch, init_ch, is_max_pool=False,),
            Encoder(init_ch, 2 * init_ch, ),
            Encoder(2 * init_ch, 4 * init_ch,),
            Encoder(4 * init_ch, 8 * init_ch,),
        ])

        self.decoders = nn.ModuleList([
            Decoder(4 * init_ch + 8 * init_ch, 4 * init_ch, interpolate=True),
            Decoder(2 * init_ch + 4 * init_ch, 2 * init_ch, interpolate=True),
            Decoder(init_ch + 2 * init_ch, init_ch, interpolate=True)
        ])

        self.rfp = DAGs(in_ch=8 * init_ch, num_neigh=num_neigh)

        self.final_conv3 = nn.Conv3d(4*init_ch, self.no_class, 1)
        self.final_conv2 = nn.Conv3d(2*init_ch, self.no_class, 1)
        self.final_conv1 = nn.Conv3d(init_ch, self.no_class, 1)

        self.final_conv = nn.Sequential(
            nn.Dropout3d(0.1),
            nn.Conv3d(init_ch, self.no_class, kernel_size=1))

    def forward(self, x):
        # encoder part
        encoders_features = []
        enc1 = self.encoders[0](x)
        enc2 = self.encoders[1](enc1)
        enc3 = self.encoders[2](enc2)
        mid = self.encoders[3](enc3)
        encoders_features = [enc3, enc2, enc1]

        ehn_mid = self.rfp(mid)

        dec3 = self.decoders[0](enc3, mid)
        dec2 = self.decoders[1](enc2, dec3)
        dec1 = self.decoders[2](enc1, dec2)

        score = self.final_conv(dec1)
        return score

class DAGs(nn.Module):
    def __init__(self, in_ch, num_neigh='four'):
        super().__init__()
        self.dag_list = None
        if num_neigh == 'four':
            self.dag_list = nn.ModuleList([UAG_RNN_4Neigh(in_ch) for _ in range(64//8)])  # hard-coding '64//8'
        elif num_neigh == 'eight':
            self.dag_list = nn.ModuleList([UAG_RNN_8Neigh(in_ch) for _ in range(64//8)])  # hard-coding '64//8'

        self.adapt = nn.Sequential(
            nn.Conv3d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(in_ch),
            nn.ReLU()
        )

    def forward(self, x):
        d = x.shape[-1]
        x_hid = []
        x_adp = x
        
        for i in range(d):
            hid = self.dag_list[i](x_adp[..., i])
            x_hid.append(hid.unsqueeze(-1))
        x_hid = torch.cat(x_hid, dim=-1)
        return x_adp + x_hid

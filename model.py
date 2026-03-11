import torch
import torch.nn as nn

class DownConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=31, stride=2, padding=15):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x):
        return self.prelu(self.norm(self.conv(x)))
    

class UpConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=31, stride=2, padding=15):
        super().__init__()
        self.convt = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding=1)
        self.norm = nn.InstanceNorm1d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self,x):
        return self.prelu(self.norm(self.convt(x)))
    

class DenoiserModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc_depths = [1, 16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]

        self.encoder = nn.ModuleList()
        for i in range(len(self.enc_depths) - 1):
            self.encoder.append(DownConv1d(self.enc_depths[i], self.enc_depths[i+1]))

        self.decoder = nn.ModuleList()
        for i in range(len(self.enc_depths)-1,0,-1):

            if i == 1:
                self.decoder.append(
                    nn.Sequential(
                        nn.ConvTranspose1d(self.enc_depths[i]*2, self.enc_depths[i-1], 31, 2, 15, output_padding = 1),
                        nn.Tanh()
                    )
                )
            else:
                self.decoder.append(UpConv1d(self.enc_depths[i]*2, self.enc_depths[i-1]))

    def forward(self, x, z=None):

        skip_connections = []
        out = x

        for enc_layer in self.encoder:
            out = enc_layer(out)
            skip_connections.append(out)

        if z is None:
            z = torch.randn_like(out)

        out = torch.cat([out, z], dim=1)
        skip_connections.pop()

        for dec_layer in self.decoder:
            out = dec_layer(out)

            if len(skip_connections) > 0:
                skip = skip_connections.pop()
                out = torch.cat([out, skip], dim=1)

        return out


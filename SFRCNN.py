import torch
import torch.nn as nn
import math
from norms import select_norm

'''
    B: Batch number
    N: Channel number
    L: length of sequence
'''


class ConvNormAct(nn.Module):
    def __init__(self, in_channels=512, out_channels=512, kernel_size=5, stride=1, groups=1, d=1, norm='gLN', mode='CNA'):
        '''
            in_channels: number of input channels
            out_channels: number of output channels
            group: the conv channel group
            d: controls the spacing between the kernel points; also known as the à trous algorithm
            mode: Conv -> Norm -> Act == 'CNA' or Conv -> Norm == 'CN' or Norm -> Act == 'NA'
                  DilatedConv -> Norm == 'DCN'.
        '''
        super(ConvNormAct, self).__init__()
        if mode == 'CNA':
            padding = int((kernel_size - 1) / 2)
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                          bias=True, groups=groups),
                select_norm(norm, out_channels),
                nn.PReLU()
            )
        elif mode == 'CN':
            padding = int((kernel_size - 1) / 2)
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                          bias=True, groups=groups),
                select_norm(norm, out_channels)
            )
        elif mode == 'NA':
            self.net = nn.Sequential(
                select_norm(norm, out_channels),
                nn.PReLU()
            )
        elif mode == 'DCN':
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=d,
                          padding=((kernel_size - 1) // 2) * d, groups=groups),
                select_norm(norm, out_channels)
            )

    def forward(self, x):
        '''
            x : [B, N, L]
        '''
        return self.net(x)


class SFRCNN_Block(nn.Module):
    '''
       Unet network structure
    '''

    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 states=4):
        '''
            in_channels: number of input channels
            out_channels: number of output channels
            states: the number of MSRNN states
        '''
        super(SFRCNN_Block, self).__init__()
        self.first = ConvNormAct(out_channels, in_channels, kernel_size=1,
                                 stride=1, groups=1)
        self.states = states
        # Bottom-up
        self.bottom_up = nn.ModuleList([])
        self.bottom_up.append(ConvNormAct(
            in_channels, in_channels, kernel_size=5, stride=1, groups=in_channels, mode='CN'))
        for i in range(1, states):
            self.bottom_up.append(ConvNormAct(
                in_channels, in_channels, kernel_size=5, stride=2, groups=in_channels, mode='CN'))
        # Top-down
        if states > 1:
            self.top_down = torch.nn.Upsample(scale_factor=2)
        # Resual Path
        self.final_norm = ConvNormAct(in_channels, mode='NA')
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x):
        '''
            x : [B, N, L]
        '''
        residual = x.clone()
        output1 = self.first(x)
        # Bottom-up
        output = [self.bottom_up[0](output1)]
        for k in range(1, self.states):
            out_k = self.bottom_up[k](output[-1])
            output.append(out_k)
        # Top-down
        for _ in range(self.states-1):
            resampled_out_k = self.top_down(output.pop(-1))
            output[-1] = output[-1] + resampled_out_k
        # Resual Path
        expanded = self.final_norm(output[-1])
        return self.res_conv(expanded) + residual


class Recurrent(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 states=4,
                 _iter=4,
                 mode='SC'):
        '''
            out_channels: the LayerNorm output channels
            in_channels: the MSRNN input channels
            states: the number of MSRNN states
            _iter: the number of iteration
            mode: DC, CC, SC in the paper
        '''
        super(Recurrent, self).__init__()
        self.iter = _iter
        self.mode = mode
        if mode == 'DC':
            self.msrnn = SFRCNN_Block(out_channels, in_channels, states)
        elif mode == 'CC':
            self.msrnn = SFRCNN_Block(out_channels, in_channels, states)
            self.concat_block = nn.Sequential(
                nn.Conv1d(out_channels*2, out_channels,
                          1, 1, groups=out_channels),
                nn.PReLU()
            )
        elif mode == 'SC':
            self.msrnn = SFRCNN_Block(out_channels, in_channels, states)
            self.concat_block = nn.Sequential(
                nn.Conv1d(out_channels, out_channels,
                          1, 1, groups=out_channels),
                nn.PReLU()
            )

    def forward(self, x):
        '''
            x : [B, N, L]
        '''
        mixture = x.clone()
        if self.mode == 'DC':
            x = self.msrnn(x)
        else:
            for i in range(self.iter):
                if i == 0:
                    x = self.msrnn(x)
                else:
                    if self.mode == 'CC':
                        x = self.msrnn(self.concat_block(
                            torch.cat((mixture, x), dim=1)))
                    else:
                        x = self.msrnn(self.concat_block(mixture+x))
        return x


class SFRCNN(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 states=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2,
                 recurrent_mode='SC'):
        super(SFRCNN, self).__init__()
        '''
            out_channels: the layernrom output channels
            in_channels:  the msrnn block input channels
            num_block:    the msrnn block number
            enc_kernel_size:  the encoder/decoder kernel size
            enc_num_basis: the encoder/decoder channels
            recurrent_mode: recurrent methods: [DC, CC, SC]
        '''
        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.states = states
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.states) // math.gcd(
            self.enc_kernel_size // 2,
            2 ** self.states)

        # Encoder
        self.encoder = nn.Conv1d(in_channels=1, out_channels=enc_num_basis,
                                 kernel_size=enc_kernel_size,
                                 stride=enc_kernel_size // 2,
                                 padding=enc_kernel_size // 2,
                                 bias=False)
        torch.nn.init.xavier_uniform_(self.encoder.weight)
        self.ln = select_norm('gLN', enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # Separation module
        self.sm = Recurrent(out_channels, in_channels,
                            states, num_blocks, mode=recurrent_mode)

        self.mask_net = nn.Sequential(nn.PReLU(),
                                      nn.Conv1d(out_channels,
                                                num_sources * enc_num_basis, 1)
                                      )

        # Decoder
        self.decoder = nn.ConvTranspose1d(
            in_channels=enc_num_basis * num_sources,
            out_channels=num_sources,
            output_padding=(enc_kernel_size // 2) - 1,
            kernel_size=enc_kernel_size,
            stride=enc_kernel_size // 2,
            padding=enc_kernel_size // 2,
            groups=1, bias=False)
        torch.nn.init.xavier_uniform_(self.decoder.weight)
        self.mask_nl_class = nn.ReLU()

    def forward(self, input_wav):
        '''
            input_wav: [T] or [B, T] or [B, 1, T]
        '''
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0).unsqueeze(1)
        if input_wav.ndim == 2:
            input_wav = input_wav.unsqueeze(1)
        # Encoder
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)
        w = x.clone()
        x = self.ln(x)
        x = self.bottleneck(x)

        # Separation module
        x = self.sm(x)
        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * w.unsqueeze(1)

        # Decoder
        estimated_waveforms = self.decoder(x.view(x.shape[0], -1, x.shape[-1]))
        estimated_waveforms = self.remove_trailing_zeros(
            estimated_waveforms, input_wav)
        if was_one_d:
            return estimated_waveforms.squeeze(0)
        return estimated_waveforms

    def pad_to_appropriate_length(self, x):
        values_to_pad = int(x.shape[-1]) % self.lcm
        if values_to_pad:
            appropriate_shape = x.shape
            padded_x = torch.zeros(
                list(appropriate_shape[:-1]) +
                [appropriate_shape[-1] + self.lcm - values_to_pad],
                dtype=torch.float32).to(x.device)
            padded_x[..., :x.shape[-1]] = x
            return padded_x
        return x

    def remove_trailing_zeros(self, padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]

def cal_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    params = 0
    for f in net.parameters():
        if f.requires_grad:
            params += f.numel()
    return round(params / 10**6, 3)

if __name__ == "__main__":
    wav = torch.rand(32000)
    model = SFRCNN(out_channels=512,
                 in_channels=512,
                 num_blocks=16,
                 states=5,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2,
                 recurrent_mode='SC')
    est = model(wav)
    print(est.shape)
    print(cal_parameters(model))

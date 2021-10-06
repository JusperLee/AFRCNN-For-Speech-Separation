import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


class ConvNormAct(nn.Module):
    '''
    This class defines the convolution layer with normalization and a PReLU
    activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.conv(input)
        output = self.norm(output)
        return self.act(output)


class ConvNorm(nn.Module):
    '''
    This class defines the convolution layer with normalization and PReLU activation
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: stride rate for down-sampling. Default is 1
        '''
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding,
                              bias=True, groups=groups)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)


class NormAct(nn.Module):
    '''
    This class defines a normalization and PReLU activation
    '''

    def __init__(self, nOut):
        '''
        :param nOut: number of output channels
        '''
        super().__init__()
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)
        self.act = nn.PReLU()

    def forward(self, input):
        output = self.norm(input)
        return self.act(output)


class DilatedConv(nn.Module):
    '''
    This class defines the dilated convolution.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)

    def forward(self, input):
        return self.conv(input)


class DilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, dilation=d,
                              padding=((kSize - 1) // 2) * d, groups=groups)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(output)

class PixelShuffle1D(torch.nn.Module):
    """
    1D pixel shuffler. https://arxiv.org/pdf/1609.05158.pdf
    Upscales sample length, downscales channel length
    "short" is input, "long" is output
    """
    def __init__(self, upscale_factor):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        batch_size = x.shape[0]
        short_channel_len = x.shape[1]
        short_width = x.shape[2]

        long_channel_len = short_channel_len // self.upscale_factor
        long_width = self.upscale_factor * short_width

        x = x.contiguous().view([batch_size, self.upscale_factor, long_channel_len, short_width])
        x = x.permute(0, 2, 3, 1).contiguous()
        x = x.view(batch_size, long_channel_len, long_width)
        return x


class TransDilatedConvNorm(nn.Module):
    '''
    This class defines the dilated convolution with normalized output.
    '''

    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        self.conv = nn.Conv1d(nIn, nOut*2, kSize, stride=stride, dilation=d,
                                       padding=((kSize - 1) // 2) * d, groups=nIn)
        # self.norm = nn.GroupNorm(1, nOut, eps=1e-08)
        self.pixelshuffle = PixelShuffle1D(2)
        self.norm = GlobLN(nOut)

    def forward(self, input):
        output = self.conv(input)
        return self.norm(self.pixelshuffle(output))

class HRNet(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4):
        super().__init__()
        self.proj_1x1 = ConvNormAct(out_channels, in_channels, 1,
                                    stride=1, groups=1)
        self.depth = upsampling_depth
        self.spp_dw = nn.ModuleList([])
        self.spp_dw.append(DilatedConvNorm(in_channels, in_channels, kSize=5,
                                           stride=1, groups=in_channels, d=1))
        # ----------Down Sample Layer----------
        for i in range(1, upsampling_depth):
            self.spp_dw.append(DilatedConvNorm(in_channels, in_channels,
                                               kSize=5,
                                               stride=2,
                                               groups=in_channels, d=1))
        # ----------Fusion Layer----------
        self.fuse_layers = nn.ModuleList([])
        for i in range(upsampling_depth):
            fuse_layer = nn.ModuleList([])
            for j in range(upsampling_depth):
                if i - j == 1:
                    fuse_layer.append(nn.Sequential(*[DilatedConvNorm(in_channels, in_channels,
                                                      kSize=5,
                                                      stride=2,
                                                      groups=in_channels, d=1) for x in range(i-j)]))
                elif i > j and i -j != 0:
                    fuse_layer.append(None)
                elif i == j:
                    fuse_layer.append(None)
                elif i < j:
                    fuse_layer.append(nn.Sequential(*[TransDilatedConvNorm(in_channels, in_channels,
                                                      kSize=5,
                                                      stride=1,
                                                      groups=in_channels, d=1) for x in range(j - i)]))
            self.fuse_layers.append(fuse_layer)
        self.concat_layer = nn.ModuleList([])
        # ----------Concat Layer----------
        for i in range(upsampling_depth):
            if i == 0 or i == 1:
                self.concat_layer.append(ConvNormAct(
                    in_channels*upsampling_depth, in_channels, 1, 1))
            elif i == upsampling_depth - 1:
                self.concat_layer.append(ConvNormAct(
                    in_channels*2, in_channels, 1, 1))
            elif i == 2:
                self.concat_layer.append(ConvNormAct(
                    in_channels*4, in_channels, 1, 1))
            elif i == 3:
                self.concat_layer.append(ConvNormAct(
                    in_channels*3, in_channels, 1, 1))


        # ----------Upsample Layer----------
        self.upsample_layers = nn.ModuleList()
        for i in range(upsampling_depth-1):
            self.upsample_layers.append(TransDilatedConvNorm(in_channels, in_channels,
                                                      kSize=5,
                                                      stride=1,
                                                      groups=in_channels, d=1))

        self.last_layer = nn.Sequential(
            NormAct(in_channels)
        )
        self.res_conv = nn.Conv1d(in_channels, out_channels, 1)
        # ----------parameters-------------
        self.depth = upsampling_depth

    def forward(self, x):
        '''
        :param x: input feature map
        :return: transformed feature map
        '''
        residual = x.clone()
        # Reduce --> project high-dimensional feature maps to low-dimensional space
        output1 = self.proj_1x1(x)
        output = [self.spp_dw[0](output1)]
        for k in range(1, self.depth):
            out_k = self.spp_dw[k](output[-1])
            output.append(out_k)

        x_fuse = []
        # import pdb; pdb.set_trace()
        for i in range(len(self.fuse_layers)):
            wav_length = output[i].shape[-1]
            _out = torch.Tensor().to(output1.device)
            for j in range(len(self.fuse_layers)):
                if i - j == 1:
                    
                    _out = torch.cat((_out, self.fuse_layers[i][j](output[j])),dim=1)
                if i == j:
                    _out = torch.cat((_out, output[j]), dim=1)
                if i < j:
                    _out = torch.cat((_out, 
                    F.interpolate(self.fuse_layers[i][j](output[j]), size=wav_length, mode='nearest')), dim=1)
            # import pdb; pdb.set_trace()
            x_fuse.append(self.concat_layer[i](_out))


        wav_length = output[0].shape[-1]
        # for i in range(1, len(x_fuse)):
        #     x_fuse[i] = F.interpolate(self.upsample_layers[i](x_fuse[i]), size=wav_length, mode='nearest')
        for i in range(self.depth-1):
            resampled_out_k = self.upsample_layers[i](x_fuse.pop(-1))
            x_fuse[-1] = x_fuse[-1] + F.interpolate(resampled_out_k, size=x_fuse[-1].shape[-1], mode='nearest')

        concat = self.last_layer(torch.cat(x_fuse, dim=1))
        expanded = self.res_conv(concat)
        return expanded + residual
        #return expanded
'''
class Attention_block(nn.Module):
    def __init__(self,out_channels=128):
        super(Attention_block,self).__init__()
        self.W_g = ConvNorm(out_channels, out_channels, 1, 1, groups=out_channels)
        
        self.W_x = ConvNorm(out_channels, out_channels, 1, 1, groups=out_channels)

        self.psi = nn.Sequential(
            ConvNorm(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU()
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        return x*psi
'''
class Recurrent(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 upsampling_depth=4,
                 _iter=4):
        super().__init__()
        self.hrnet = HRNet(out_channels, in_channels, upsampling_depth)
        self.iter = _iter
        #self.attention = Attention_block(out_channels)
        self.concat_block = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1, 1, groups=out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        mixture = x.clone()
        for i in range(self.iter):
            if i == 0:
                x = self.hrnet(x)
            else:
                #m = self.attention(mixture, x)
                x = self.hrnet(self.concat_block(mixture+x))
        return x


class Control2(nn.Module):
    def __init__(self,
                 out_channels=128,
                 in_channels=512,
                 num_blocks=16,
                 upsampling_depth=4,
                 enc_kernel_size=21,
                 enc_num_basis=512,
                 num_sources=2):
        super(Control2, self).__init__()

        # Number of sources to produce
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.upsampling_depth = upsampling_depth
        self.enc_kernel_size = enc_kernel_size
        self.enc_num_basis = enc_num_basis
        self.num_sources = num_sources

        # Appropriate padding is needed for arbitrary lengths
        self.lcm = abs(self.enc_kernel_size // 2 * 2 **
                       self.upsampling_depth) // math.gcd(
            self.enc_kernel_size // 2,
            2 ** self.upsampling_depth)

        # Front end
        self.encoder = nn.Conv1d(in_channels=1, out_channels=enc_num_basis,
                                 kernel_size=enc_kernel_size,
                                 stride=enc_kernel_size // 2,
                                 padding=enc_kernel_size // 2,
                                 bias=False)
        torch.nn.init.xavier_uniform_(self.encoder.weight)

        # Norm before the rest, and apply one more dense layer
        self.ln = GlobLN(enc_num_basis)
        self.bottleneck = nn.Conv1d(
            in_channels=enc_num_basis,
            out_channels=out_channels,
            kernel_size=1)

        # Separation module
        self.sm = Recurrent(out_channels, in_channels, upsampling_depth, num_blocks)

        mask_conv = nn.Conv1d(out_channels, num_sources * enc_num_basis, 1)
        self.mask_net = nn.Sequential(nn.PReLU(), mask_conv)

        # Back end
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
    # Forward pass

    def forward(self, input_wav):
        was_one_d = False
        if input_wav.ndim == 1:
            was_one_d = True
            input_wav = input_wav.unsqueeze(0).unsqueeze(1)
        if input_wav.ndim == 2:
            input_wav = input_wav.unsqueeze(1)
        # Front end
        x = self.pad_to_appropriate_length(input_wav)
        x = self.encoder(x)

        # Split paths
        s = x.clone()
        # Separation module
        x = self.ln(x)
        x = self.bottleneck(x)
        x = self.sm(x)

        x = self.mask_net(x)
        x = x.view(x.shape[0], self.num_sources, self.enc_num_basis, -1)
        x = self.mask_nl_class(x)
        x = x * s.unsqueeze(1)
        # Back end
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

    @staticmethod
    def remove_trailing_zeros(padded_x, initial_x):
        return padded_x[..., :initial_x.shape[-1]]


def check_parameters(net):
    '''
        Returns module parameters. Mb
    '''
    parameters = sum(param.numel() for param in net.parameters())
    return parameters / 10**6


if __name__ == "__main__":
    model = Control2(out_channels=512,
                     in_channels=512,
                     num_blocks=8,
                     upsampling_depth=5,
                     enc_kernel_size=21,
                     enc_num_basis=512,
                     num_sources=2)

    dummy_input = torch.rand(1, 1, 24000)
    estimated_sources = model(dummy_input)
    print(estimated_sources.shape)
    print(check_parameters(model))
    # print(model)
    ps = TransDilatedConvNorm(512, 512, kSize=5, stride=1, groups=512, d=1)
    print(check_parameters(ps))

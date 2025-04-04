'By 2p2o5 [https://github.com/2p2o5]'
_H='linear'
_G='ZH'
_F=1.
_E=True
_D=.0
_C="'"
_B=False
_A=None
import os,sys
dirnow=os.path.dirname(__file__)
sys.path.append(dirnow)
import torch
torch._C._set_grad_enabled(_B)
import math,os,sys,io,cn2an,json,wave,re,jieba_fast.posseg as psg,numpy as np
from torch import nn
from torch.nn import Conv1d,ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import weight_norm,remove_weight_norm
from transformers import AutoTokenizer,AutoModelForMaskedLM
from pypinyin import lazy_pinyin,Style
_DEVICE=torch.device('cpu')
modules__LRELU_SLOPE=.1
class LayerNorm(nn.Module):
	def __init__(A,channels,eps=1e-05):B=channels;super().__init__();A.channels=B;A.eps=eps;A.gamma=nn.Parameter(torch.ones(B));A.beta=nn.Parameter(torch.zeros(B))
	def forward(A,x):x=x.transpose(1,-1);x=F.layer_norm(x,(A.channels,),A.gamma,A.beta,A.eps);return x.transpose(1,-1)
class DDSConv(nn.Module):
	def __init__(A,channels,kernel_size,n_layers,p_dropout=_D):
		E=p_dropout;F=n_layers;C=kernel_size;B=channels;super().__init__();A.channels=B;A.kernel_size=C;A.n_layers=F;A.p_dropout=E;A.drop=nn.Dropout(E);A.convs_sep=nn.ModuleList();A.convs_1x1=nn.ModuleList();A.norms_1=nn.ModuleList();A.norms_2=nn.ModuleList()
		for G in range(F):D=C**G;H=(C*D-D)//2;A.convs_sep.append(nn.Conv1d(B,B,C,groups=B,dilation=D,padding=H));A.convs_1x1.append(nn.Conv1d(B,B,1));A.norms_1.append(LayerNorm(B));A.norms_2.append(LayerNorm(B))
	def forward(B,x,x_mask,g=_A):
		D=x_mask
		if g is not _A:x=x+g
		for C in range(B.n_layers):A=B.convs_sep[C](x*D);A=B.norms_1[C](A);A=F.gelu(A);A=B.convs_1x1[C](A);A=B.norms_2[C](A);A=F.gelu(A);A=B.drop(A);x=x+A
		return x*D
class ResBlock1(torch.nn.Module):
	def __init__(D,channels,kernel_size=3,dilation=(1,3,5)):C=dilation;A=kernel_size;B=channels;super(ResBlock1,D).__init__();D.convs1=nn.ModuleList([weight_norm(Conv1d(B,B,A,1,dilation=C[0],padding=commons__get_padding(A,C[0]))),weight_norm(Conv1d(B,B,A,1,dilation=C[1],padding=commons__get_padding(A,C[1]))),weight_norm(Conv1d(B,B,A,1,dilation=C[2],padding=commons__get_padding(A,C[2])))]);D.convs1.apply(commons__init_weights);D.convs2=nn.ModuleList([weight_norm(Conv1d(B,B,A,1,dilation=1,padding=commons__get_padding(A,1))),weight_norm(Conv1d(B,B,A,1,dilation=1,padding=commons__get_padding(A,1))),weight_norm(Conv1d(B,B,A,1,dilation=1,padding=commons__get_padding(A,1)))]);D.convs2.apply(commons__init_weights)
	def forward(C,x,x_mask=_A):
		B=x_mask
		for(D,E)in zip(C.convs1,C.convs2):
			A=F.leaky_relu(x,modules__LRELU_SLOPE)
			if B is not _A:A=A*B
			A=D(A);A=F.leaky_relu(A,modules__LRELU_SLOPE)
			if B is not _A:A=A*B
			A=E(A);x=A+x
		if B is not _A:x=x*B
		return x
	def remove_weight_norm(B):
		for A in B.convs1:remove_weight_norm(A)
		for A in B.convs2:remove_weight_norm(A)
class ResBlock2(torch.nn.Module):
	def __init__(D,channels,kernel_size=3,dilation=(1,3)):A=dilation;B=kernel_size;C=channels;super(ResBlock2,D).__init__();D.convs=nn.ModuleList([weight_norm(Conv1d(C,C,B,1,dilation=A[0],padding=commons__get_padding(B,A[0]))),weight_norm(Conv1d(C,C,B,1,dilation=A[1],padding=commons__get_padding(B,A[1])))]);D.convs.apply(commons__init_weights)
	def forward(C,x,x_mask=_A):
		B=x_mask
		for D in C.convs:
			A=F.leaky_relu(x,modules__LRELU_SLOPE)
			if B is not _A:A=A*B
			A=D(A);x=A+x
		if B is not _A:x=x*B
		return x
	def remove_weight_norm(A):
		for B in A.convs:remove_weight_norm(B)
class Log(nn.Module):
	def forward(D,x,x_mask,reverse=_B,**E):
		A=x_mask
		if not reverse:B=torch.log(torch.clamp_min(x,1e-05))*A;C=torch.sum(-B,[1,2]);return B,C
		else:x=torch.exp(x)*A;return x
class Flip(nn.Module):
	def forward(B,x,*C,reverse=_B,**D):
		x=torch.flip(x,[1])
		if not reverse:A=torch.zeros(x.size(0)).to(dtype=x.dtype,device=_DEVICE);return x,A
		else:return x
class ElementwiseAffine(nn.Module):
	def __init__(A,channels):B=channels;super().__init__();A.channels=B;A.m=nn.Parameter(torch.zeros(B,1));A.logs=nn.Parameter(torch.zeros(B,1))
	def forward(A,x,x_mask,reverse=_B,**E):
		B=x_mask
		if not reverse:C=A.m+torch.exp(A.logs)*x;C=C*B;D=torch.sum(A.logs*B,[1,2]);return C,D
		else:x=(x-A.m)*torch.exp(-A.logs)*B;return x
class ConvFlow(nn.Module):
	def __init__(A,in_channels,filter_channels,kernel_size,n_layers,num_bins=10,tail_bound=5.):C=num_bins;D=n_layers;E=kernel_size;F=in_channels;B=filter_channels;super().__init__();A.in_channels=F;A.filter_channels=B;A.kernel_size=E;A.n_layers=D;A.num_bins=C;A.tail_bound=tail_bound;A.half_channels=F//2;A.pre=nn.Conv1d(A.half_channels,B,1);A.convs=DDSConv(B,E,D,p_dropout=_D);A.proj=nn.Conv1d(B,A.half_channels*(C*3-1),1);A.proj.weight.data.zero_();A.proj.bias.data.zero_()
	def forward(A,x,x_mask,g=_A,reverse=_B):
		F=reverse;C=x_mask;D,E=torch.split(x,[A.half_channels]*2,1);O=B=A.pre(D);P=B=A.convs(B,C,g=g);Q=B=A.proj(B)*C;G,H,I=D.shape;B=B.reshape(G,H,-1,I).permute(0,1,3,2);J=B[...,:A.num_bins]/math.sqrt(A.filter_channels);K=B[...,A.num_bins:2*A.num_bins]/math.sqrt(A.filter_channels);L=B[...,2*A.num_bins:];E,M=transforms__piecewise_rational_quadratic_transform(E,J,K,L,inverse=F,tails=_H,tail_bound=A.tail_bound);x=torch.cat([D,E],1)*C;N=torch.sum(M*C,[1,2])
		if not F:return x,N
		else:return x
class TransformerCouplingLayer(nn.Module):
	def __init__(A,channels,hidden_channels,kernel_size,n_layers,n_heads,p_dropout=0,filter_channels=0,mean_only=_B,wn_sharing_parameter=_A,gin_channels=0):D=wn_sharing_parameter;E=mean_only;F=n_layers;G=kernel_size;C=channels;B=hidden_channels;assert C%2==0,'channels should be divisible by 2';super().__init__();A.channels=C;A.hidden_channels=B;A.kernel_size=G;A.n_layers=F;A.half_channels=C//2;A.mean_only=E;A.pre=nn.Conv1d(A.half_channels,B,1);A.enc=Encoder(B,filter_channels,n_heads,F,G,p_dropout,isflow=_E,gin_channels=gin_channels)if D is _A else D;A.post=nn.Conv1d(B,A.half_channels*(2-E),1);A.post.weight.data.zero_();A.post.bias.data.zero_()
	def forward(A,x,x_mask,g=_A,reverse=_B):
		C=x_mask;F,B=torch.split(x,[A.half_channels]*2,1);G=A.pre(F)*C;G=A.enc.forward(G,C,g=g);H=A.post(G)*C
		if not A.mean_only:D,E=torch.split(H,[A.half_channels]*2,1)
		else:D=H;E=torch.zeros_like(D)
		if not reverse:B=D+B*torch.exp(E)*C;x=torch.cat([F,B],1);I=torch.sum(E,[1,2]);return x,I
		else:B=(B-D)*torch.exp(-E)*C;x=torch.cat([F,B],1);return x
class TransformerCouplingBlock(nn.Module):
	def __init__(A,channels,hidden_channels,filter_channels,n_heads,n_layers,kernel_size,p_dropout,n_flows=4,gin_channels=0,share_parameter=_B):
		B=n_flows;C=kernel_size;D=n_layers;E=hidden_channels;F=channels;super().__init__();A.channels=F;A.hidden_channels=E;A.kernel_size=C;A.n_layers=D;A.n_flows=B;A.gin_channels=gin_channels;A.flows=nn.ModuleList()
		if share_parameter:raise RuntimeError
		A.wn=_A
		for G in range(B):A.flows.append(TransformerCouplingLayer(F,E,C,D,n_heads,p_dropout,filter_channels,mean_only=_E,wn_sharing_parameter=A.wn,gin_channels=A.gin_channels));A.flows.append(Flip())
	def forward(C,x,x_mask,g=_A,reverse=_B):
		D=x_mask;A=reverse
		if not A:
			for B in C.flows:x,E=B(x,D,g=g,reverse=A)
		else:
			for B in reversed(C.flows):x=B(x,D,g=g,reverse=A)
		return x
class StochasticDurationPredictor(nn.Module):
	def __init__(A,in_channels,filter_channels,kernel_size,p_dropout,n_flows=4,gin_channels=0):
		G=n_flows;D=gin_channels;E=p_dropout;F=in_channels;C=kernel_size;B=filter_channels;super().__init__();B=F;A.in_channels=F;A.filter_channels=B;A.kernel_size=C;A.p_dropout=E;A.n_flows=G;A.gin_channels=D;A.log_flow=Log();A.flows=nn.ModuleList();A.flows.append(ElementwiseAffine(2))
		for H in range(G):A.flows.append(ConvFlow(2,B,C,n_layers=3));A.flows.append(Flip())
		A.post_pre=nn.Conv1d(1,B,1);A.post_proj=nn.Conv1d(B,B,1);A.post_convs=DDSConv(B,C,n_layers=3,p_dropout=E);A.post_flows=nn.ModuleList();A.post_flows.append(ElementwiseAffine(2))
		for I in range(4):A.post_flows.append(ConvFlow(2,B,C,n_layers=3));A.post_flows.append(Flip())
		A.pre=nn.Conv1d(F,B,1);A.proj=nn.Conv1d(B,B,1);A.convs=DDSConv(B,C,n_layers=3,p_dropout=E)
		if D!=0:A.cond=nn.Conv1d(D,B,1)
	def forward(B,x,x_mask,w=_A,g=_A,reverse=_B,noise_scale=_F):
		J=reverse;A=x_mask;x=torch.detach(x);x=B.pre(x)
		if g is not _A:g=torch.detach(g);x=x+B.cond(g)
		x=B.convs(x,A);x=B.proj(x)*A
		if not J:
			D=B.flows;assert w is not _A;K=0;E=B.post_pre(w);E=B.post_convs(E,A);E=B.post_proj(E)*A;O=torch.randn(w.size(0),2,w.size(2)).to(device=_DEVICE,dtype=x.dtype)*A;L=O
			for G in B.post_flows:L,Q=G(L,A,g=x+E);K+=Q
			M,P=torch.split(L,[1,1],1);R=torch.sigmoid(M)*A;H=(w-R)*A;K+=torch.sum((F.logsigmoid(M)+F.logsigmoid(-M))*A,[1,2]);S=torch.sum(-.5*(math.log(2*math.pi)+O**2)*A,[1,2])-K;I=0;H,N=B.log_flow(H,A);I+=N;C=torch.cat([H,P],1)
			for G in D:C,N=G(C,A,g=x,reverse=J);I=I+N
			T=torch.sum(.5*(math.log(2*math.pi)+C**2)*A,[1,2])-I;return T+S
		else:
			D=list(reversed(B.flows));D=D[:-2]+[D[-1]];C=torch.randn(x.size(0),2,x.size(2)).to(device=_DEVICE,dtype=x.dtype)*noise_scale
			for G in D:C=G(C,A,g=x,reverse=J)
			H,P=torch.split(C,[1,1],1);U=H;return U
class DurationPredictor(nn.Module):
	def __init__(A,in_channels,filter_channels,kernel_size,p_dropout,gin_channels=0):
		F=p_dropout;D=gin_channels;E=in_channels;C=kernel_size;B=filter_channels;super().__init__();A.in_channels=E;A.filter_channels=B;A.kernel_size=C;A.p_dropout=F;A.gin_channels=D;A.drop=nn.Dropout(F);A.conv_1=nn.Conv1d(E,B,C,padding=C//2);A.norm_1=LayerNorm(B);A.conv_2=nn.Conv1d(B,B,C,padding=C//2);A.norm_2=LayerNorm(B);A.proj=nn.Conv1d(B,1,1)
		if D!=0:A.cond=nn.Conv1d(D,E,1)
	def forward(A,x,x_mask,g=_A):
		B=x_mask;x=torch.detach(x)
		if g is not _A:g=torch.detach(g);x=x+A.cond(g)
		x=A.conv_1(x*B);x=torch.relu(x);x=A.norm_1(x);x=A.drop(x);x=A.conv_2(x*B);x=torch.relu(x);x=A.norm_2(x);x=A.drop(x);x=A.proj(x*B);return x*B
class TextEncoder(nn.Module):
	def __init__(A,n_vocab,out_channels,hidden_channels,filter_channels,n_heads,n_layers,kernel_size,p_dropout,gin_channels=0):C=p_dropout;D=kernel_size;E=n_layers;F=n_heads;G=filter_channels;H=out_channels;B=hidden_channels;super().__init__();A.n_vocab=n_vocab;A.out_channels=H;A.hidden_channels=B;A.filter_channels=G;A.n_heads=F;A.n_layers=E;A.kernel_size=D;A.p_dropout=C;A.gin_channels=gin_channels;A.emb=nn.Embedding(len(symbols__symbols),B);nn.init.normal_(A.emb.weight,_D,B**-.5);A.tone_emb=nn.Embedding(symbols__num_tones,B);nn.init.normal_(A.tone_emb.weight,_D,B**-.5);A.language_emb=nn.Embedding(symbols__num_languages,B);nn.init.normal_(A.language_emb.weight,_D,B**-.5);A.bert_proj=nn.Conv1d(1024,B,1);A.encoder=Encoder(B,G,F,E,D,C,gin_channels=A.gin_channels);A.proj=nn.Conv1d(B,H*2,1)
	def forward(A,x,x_lengths,tone,language,bert,g=_A):x=(A.emb(x)+A.tone_emb(tone)+A.language_emb(language)+A.bert_proj(bert).transpose(1,2))*math.sqrt(A.hidden_channels);x=torch.transpose(x,1,-1);B=torch.unsqueeze(sequence_mask(x_lengths,x.size(2)),1).to(x.dtype);x=A.encoder.forward(x*B,B,g=g);C=A.proj(x)*B;D,E=torch.split(C,A.out_channels,dim=1);return x,D,E,B
class Generator(torch.nn.Module):
	def __init__(A,initial_channel,resblock,resblock_kernel_sizes,resblock_dilation_sizes,upsample_rates,upsample_initial_channel,upsample_kernel_sizes,gin_channels=0):
		F=gin_channels;G=upsample_rates;H=resblock_kernel_sizes;E=resblock;B=upsample_initial_channel;super(Generator,A).__init__();A.num_kernels=len(H);A.num_upsamples=len(G);A.conv_pre=Conv1d(initial_channel,B,7,1,padding=3);E=ResBlock1 if E=='1'else ResBlock2;A.ups=nn.ModuleList()
		for(C,(I,D))in enumerate(zip(G,upsample_kernel_sizes)):A.ups.append(weight_norm(ConvTranspose1d(B//2**C,B//2**(C+1),D,I,padding=(D-I)//2)))
		A.resblocks=nn.ModuleList()
		for C in range(len(A.ups)):
			J=B//2**(C+1)
			for(L,(D,K))in enumerate(zip(H,resblock_dilation_sizes)):A.resblocks.append(E(J,D,K))
		A.conv_post=Conv1d(J,1,7,1,padding=3,bias=_B);A.ups.apply(commons__init_weights)
		if F!=0:A.cond=nn.Conv1d(F,B,1)
	def forward(A,x,g=_A):
		x=A.conv_pre(x)
		if g is not _A:x=x+A.cond(g)
		for C in range(A.num_upsamples):
			x=F.leaky_relu(x,modules__LRELU_SLOPE);x=A.ups[C](x);B=_A
			for D in range(A.num_kernels):
				if B is _A:B=A.resblocks[C*A.num_kernels+D](x)
				else:B+=A.resblocks[C*A.num_kernels+D](x)
			x=B/A.num_kernels
		x=F.leaky_relu(x);x=A.conv_post(x);x=torch.tanh(x);return x
	def remove_weight_norm(B):
		print('Removing weight norm...')
		for A in B.ups:remove_weight_norm(A)
		for A in B.resblocks:A.remove_weight_norm()
class SynthesizerTrn(nn.Module):
	def __init__(A,n_vocab,spec_channels,segment_size,inter_channels,hidden_channels,filter_channels,n_heads,n_layers,kernel_size,p_dropout,resblock,resblock_kernel_sizes,resblock_dilation_sizes,upsample_rates,upsample_initial_channel,upsample_kernel_sizes,n_speakers=256,gin_channels=256,use_sdp=_E,n_flow_layer=4,n_layers_trans_flow=3,flow_share_parameter=_B,use_transformer_flow=_E,**D):
		I=n_layers_trans_flow;J=n_speakers;K=upsample_kernel_sizes;L=upsample_initial_channel;M=upsample_rates;N=resblock_dilation_sizes;O=resblock_kernel_sizes;P=resblock;Q=kernel_size;R=n_layers;S=n_vocab;F=p_dropout;G=n_heads;H=filter_channels;E=inter_channels;C=hidden_channels;B=gin_channels;super().__init__();A.n_vocab=S;A.spec_channels=spec_channels;A.inter_channels=E;A.hidden_channels=C;A.filter_channels=H;A.n_heads=G;A.n_layers=R;A.kernel_size=Q;A.p_dropout=F;A.resblock=P;A.resblock_kernel_sizes=O;A.resblock_dilation_sizes=N;A.upsample_rates=M;A.upsample_initial_channel=L;A.upsample_kernel_sizes=K;A.segment_size=segment_size;A.n_speakers=J;A.gin_channels=B;A.n_layers_trans_flow=I;A.use_spk_conditioned_encoder=D.get('use_spk_conditioned_encoder',_E);A.use_sdp=use_sdp;A.use_noise_scaled_mas=D.get('use_noise_scaled_mas',_B);A.mas_noise_scale_initial=D.get('mas_noise_scale_initial',.01);A.noise_scale_delta=D.get('noise_scale_delta',2e-06);A.current_mas_noise_scale=A.mas_noise_scale_initial
		if A.use_spk_conditioned_encoder and B>0:A.enc_gin_channels=B
		A.enc_p=TextEncoder(S,E,C,H,G,R,Q,F,gin_channels=A.enc_gin_channels);A.dec=Generator(E,P,O,N,M,L,K,gin_channels=B)
		if use_transformer_flow:A.flow=TransformerCouplingBlock(E,C,H,G,I,5,F,n_flow_layer,gin_channels=B,share_parameter=flow_share_parameter)
		else:print("Don't All use transformer_flow.");exit(1)
		A.sdp=StochasticDurationPredictor(C,192,3,.5,4,gin_channels=B);A.dp=DurationPredictor(C,256,3,.5,gin_channels=B);A.emb_g=nn.Embedding(J,B)
	def infer(B,x,x_lengths,sid,tone,language,bert,noise_scale=.333,length_scale=1,noise_scale_w=.333,sdp_ratio=0):
		J=sdp_ratio
		if B.n_speakers>0:C=B.emb_g(sid).unsqueeze(-1)
		else:0
		x,E,G,D=B.enc_p.forward(x,x_lengths,tone,language,bert,g=C);K=torch.ceil(torch.exp(B.sdp.forward(x,D,g=C,reverse=_E,noise_scale=noise_scale_w)*J+B.dp.forward(x,D,g=C)*(1-J))*D*length_scale);O=torch.clamp_min(torch.sum(K,[1,2]),1).long();H=torch.unsqueeze(sequence_mask(O,_A),1).to(D.dtype);I=torch.unsqueeze(D,2)*torch.unsqueeze(H,-1);L,U,M,N=I.shape;P=torch.cumsum(K,-1);Q=P.view(L*N);A=sequence_mask(Q,M).to(I.dtype);A=A.view(L,N,M);A=A-F.pad(A,commons__convert_pad_shape([[0,0],[1,0],[0,0]]))[:,:-1];A=A.unsqueeze(1).transpose(2,3)*I;E=torch.matmul(A.squeeze(1),E.transpose(1,2)).transpose(1,2);G=torch.matmul(A.squeeze(1),G.transpose(1,2)).transpose(1,2);R=E.contiguous()+torch.randn_like(E.contiguous())*torch.exp(G)*noise_scale;S=B.flow.forward(R,H,g=C,reverse=_E);T=B.dec.forward(S*H,g=C);return T
utils__MATPLOTLIB_FLAG=_B
def utils__load_checkpoint(checkpoint_path,model,optimizer=_A,skip_optimizer=_B):
	K='module';L='params';M=skip_optimizer;N=checkpoint_path;H='optimizer';E='param_groups';B=optimizer;A=model;assert os.path.isfile(N);C=torch.load(N,map_location=_DEVICE);P=C['iteration'];Q=C['learning_rate']
	if B is not _A and not M and C[H]is not _A:B.load_state_dict(C[H])
	elif B is _A and not M:F=B.state_dict();R=F[E][0][L];F[E]=C[H][E];F[E][0][L]=R;B.load_state_dict(F)
	I=C['model']
	if hasattr(A,K):O=A.module.state_dict()
	else:O=A.state_dict()
	G={}
	for(D,J)in O.items():
		try:G[D]=I[D];assert I[D].shape==J.shape,(I[D].shape,J.shape)
		except:print('%s is not in the checkpoint'%D);G[D]=J
	if hasattr(A,K):A.module.load_state_dict(G,strict=_B)
	else:A.load_state_dict(G,strict=_B)
	return A,B,Q,P
class HParams:
	def __init__(B,**C):
		for(D,A)in C.items():
			if type(A)==dict:A=HParams(**A)
			B[D]=A
	def keys(A):return A.__dict__.keys()
	def items(A):return A.__dict__.items()
	def values(A):return A.__dict__.values()
	def __len__(A):return len(A.__dict__)
	def __getitem__(A,key):return getattr(A,key)
	def __setitem__(A,key,value):return setattr(A,key,value)
	def __contains__(A,key):return key in A.__dict__
	def __repr__(A):return A.__dict__.__repr__()
transforms__DEFAULT_MIN_BIN_WIDTH=.001
transforms__DEFAULT_MIN_BIN_HEIGHT=.001
transforms__DEFAULT_MIN_DERIVATIVE=.001
def transforms__piecewise_rational_quadratic_transform(inputs,unnormalized_widths,unnormalized_heights,unnormalized_derivatives,inverse=_B,tails=_A,tail_bound=_F,min_bin_width=transforms__DEFAULT_MIN_BIN_WIDTH,min_bin_height=transforms__DEFAULT_MIN_BIN_HEIGHT,min_derivative=transforms__DEFAULT_MIN_DERIVATIVE):
	A=tails
	if A is _A:B=transforms__rational_quadratic_spline;C={}
	else:B=transforms__unconstrained_rational_quadratic_spline;C={'tails':A,'tail_bound':tail_bound}
	D,E=B(inputs=inputs,unnormalized_widths=unnormalized_widths,unnormalized_heights=unnormalized_heights,unnormalized_derivatives=unnormalized_derivatives,inverse=inverse,min_bin_width=min_bin_width,min_bin_height=min_bin_height,min_derivative=min_derivative,**C);return D,E
def transforms__searchsorted(bin_locations,inputs,eps=1e-06):A=bin_locations;A[...,-1]+=eps;return torch.sum(inputs[...,_A]>=A,dim=-1)-1
def transforms__unconstrained_rational_quadratic_spline(inputs,unnormalized_widths,unnormalized_heights,unnormalized_derivatives,inverse=_B,tails=_H,tail_bound=_F,min_bin_width=transforms__DEFAULT_MIN_BIN_WIDTH,min_bin_height=transforms__DEFAULT_MIN_BIN_HEIGHT,min_derivative=transforms__DEFAULT_MIN_DERIVATIVE):
	I=min_derivative;J=tails;D=unnormalized_derivatives;B=tail_bound;C=inputs;A=(C>=-B)&(C<=B);E=~A;G=torch.zeros_like(C);H=torch.zeros_like(C)
	if J==_H:D=F.pad(D,pad=(1,1));K=np.log(np.exp(1-I)-1);D[...,0]=K;D[...,-1]=K;G[E]=C[E];H[E]=0
	else:raise RuntimeError('{} tails are not implemented.'.format(J))
	G[A],H[A]=transforms__rational_quadratic_spline(inputs=C[A],unnormalized_widths=unnormalized_widths[A,:],unnormalized_heights=unnormalized_heights[A,:],unnormalized_derivatives=D[A,:],inverse=inverse,left=-B,right=B,bottom=-B,top=B,min_bin_width=min_bin_width,min_bin_height=min_bin_height,min_derivative=I);return G,H
def transforms__rational_quadratic_spline(inputs,unnormalized_widths,unnormalized_heights,unnormalized_derivatives,inverse=_B,left=_D,right=_F,bottom=_D,top=_F,min_bin_width=transforms__DEFAULT_MIN_BIN_WIDTH,min_bin_height=transforms__DEFAULT_MIN_BIN_HEIGHT,min_derivative=transforms__DEFAULT_MIN_DERIVATIVE):
	Z='constant';a=inverse;b=unnormalized_widths;R=min_bin_height;S=min_bin_width;T=bottom;U=right;N=left;E=inputs
	if torch.min(E)<N or torch.max(E)>U:raise ValueError('Input to a transform is not within its domain')
	O=b.shape[-1]
	if S*O>_F:raise ValueError('Minimal bin width too large for the number of bins')
	if R*O>_F:raise ValueError('Minimal bin height too large for the number of bins')
	H=F.softmax(b,dim=-1);H=S+(1-S*O)*H;B=torch.cumsum(H,dim=-1);B=F.pad(B,pad=(1,0),mode=Z,value=_D);B=(U-N)*B+N;B[...,0]=N;B[...,-1]=U;H=B[...,1:]-B[...,:-1];c=min_derivative+F.softplus(unnormalized_derivatives);I=F.softmax(unnormalized_heights,dim=-1);I=R+(1-R*O)*I;C=torch.cumsum(I,dim=-1);C=F.pad(C,pad=(1,0),mode=Z,value=_D);C=(top-T)*C+T;C[...,0]=T;C[...,-1]=top;I=C[...,1:]-C[...,:-1]
	if a:G=transforms__searchsorted(C,E)[...,_A]
	else:G=transforms__searchsorted(B,E)[...,_A]
	d=B.gather(-1,G)[...,0];e=H.gather(-1,G)[...,0];P=C.gather(-1,G)[...,0];i=I/H;A=i.gather(-1,G)[...,0];D=c.gather(-1,G)[...,0];J=c[...,1:].gather(-1,G)[...,0];V=I.gather(-1,G)[...,0]
	if a:j=(E-P)*(D+J-2*A)+V*(A-D);f=V*D-(E-P)*(D+J-2*A);g=-A*(E-P);h=f.pow(2)-4*j*g;assert(h>=0).all();L=2*g/(-f-torch.sqrt(h));W=L*e+d;K=L*(1-L);Q=A+(D+J-2*A)*K;X=A.pow(2)*(J*L.pow(2)+2*A*K+D*(1-L).pow(2));Y=torch.log(X)-2*torch.log(Q);return W,-Y
	else:M=(E-d)/e;K=M*(1-M);k=V*(A*M.pow(2)+D*K);Q=A+(D+J-2*A)*K;W=P+k/Q;X=A.pow(2)*(J*M.pow(2)+2*A*K+D*(1-M).pow(2));Y=torch.log(X)-2*torch.log(Q);return W,Y
class Encoder(nn.Module):
	def __init__(A,hidden_channels,filter_channels,n_heads,n_layers,kernel_size=1,p_dropout=_D,window_size=4,isflow=_E,**C):
		E='cond_layer_idx';F='gin_channels';G=window_size;H=kernel_size;I=n_heads;J=filter_channels;D=p_dropout;B=hidden_channels;super().__init__();A.hidden_channels=B;A.filter_channels=J;A.n_heads=I;A.n_layers=n_layers;A.kernel_size=H;A.p_dropout=D;A.window_size=G;A.cond_layer_idx=A.n_layers
		if F in C:
			A.gin_channels=C[F]
			if A.gin_channels!=0:A.spk_emb_linear=nn.Linear(A.gin_channels,A.hidden_channels);A.cond_layer_idx=C[E]if E in C else 2;assert A.cond_layer_idx<A.n_layers,'cond_layer_idx should be less than n_layers'
		A.drop=nn.Dropout(D);A.attn_layers=nn.ModuleList();A.norm_layers_1=nn.ModuleList();A.ffn_layers=nn.ModuleList();A.norm_layers_2=nn.ModuleList()
		for K in range(A.n_layers):A.attn_layers.append(MultiHeadAttention(B,B,I,p_dropout=D,window_size=G));A.norm_layers_1.append(LayerNorm(B));A.ffn_layers.append(FFN(B,B,J,H,p_dropout=D));A.norm_layers_2.append(LayerNorm(B))
	def forward(A,x,x_mask,g=_A):
		C=x_mask;E=C.unsqueeze(2)*C.unsqueeze(-1);x=x*C
		for D in range(A.n_layers):
			if D==A.cond_layer_idx and g is not _A:g=A.spk_emb_linear(g.transpose(1,2));g=g.transpose(1,2);x=x+g;x=x*C
			B=A.attn_layers[D](x,x,E);B=A.drop(B);x=A.norm_layers_1[D](x+B);B=A.ffn_layers[D](x,C);B=A.drop(B);x=A.norm_layers_2[D](x+B)
		x=x*C;return x
class MultiHeadAttention(nn.Module):
	def __init__(A,channels,out_channels,n_heads,p_dropout=_D,window_size=_A,heads_share=_E,block_length=_A,proximal_bias=_B,proximal_init=_B):
		E=proximal_init;F=heads_share;G=p_dropout;H=out_channels;C=window_size;D=n_heads;B=channels;super().__init__();assert B%D==0;A.channels=B;A.out_channels=H;A.n_heads=D;A.p_dropout=G;A.window_size=C;A.heads_share=F;A.block_length=block_length;A.proximal_bias=proximal_bias;A.proximal_init=E;A.attn=_A;A.k_channels=B//D;A.conv_q=nn.Conv1d(B,B,1);A.conv_k=nn.Conv1d(B,B,1);A.conv_v=nn.Conv1d(B,B,1);A.conv_o=nn.Conv1d(B,H,1);A.drop=nn.Dropout(G)
		if C is not _A:I=1 if F else D;J=A.k_channels**-.5;A.emb_rel_k=nn.Parameter(torch.randn(I,C*2+1,A.k_channels)*J);A.emb_rel_v=nn.Parameter(torch.randn(I,C*2+1,A.k_channels)*J)
		nn.init.xavier_uniform_(A.conv_q.weight);nn.init.xavier_uniform_(A.conv_k.weight);nn.init.xavier_uniform_(A.conv_v.weight)
		if E:A.conv_k.weight.copy_(A.conv_q.weight);A.conv_k.bias.copy_(A.conv_q.bias)
	def forward(A,x,c,attn_mask=_A):B=A.conv_q(x);C=A.conv_k(c);D=A.conv_v(c);x,A.attn=A.attention(B,C,D,mask=attn_mask);x=A.conv_o(x);return x
	def attention(A,query,key,value,mask=_A):
		K=value;I=key;D=query;J,L,C,E=*I.size(),D.size(2);D=D.view(J,A.n_heads,A.k_channels,E).transpose(2,3);I=I.view(J,A.n_heads,A.k_channels,C).transpose(2,3);K=K.view(J,A.n_heads,A.k_channels,C).transpose(2,3);B=torch.matmul(D/math.sqrt(A.k_channels),I.transpose(-2,-1))
		if A.window_size is not _A:assert C==E,'Relative attention is only available for self-attention.';M=A._get_relative_embeddings(A.emb_rel_k,C);N=A._matmul_with_relative_keys(D/math.sqrt(A.k_channels),M);O=A._relative_position_to_absolute_position(N);B=B+O
		if A.proximal_bias:assert C==E,'Proximal bias is only available for self-attention.';B=B+A._attention_bias_proximal(C).to(device=_DEVICE,dtype=B.dtype)
		if mask is not _A:
			B=B.masked_fill(mask==0,-1e4)
			if A.block_length is not _A:assert C==E,'Local attention is only available for self-attention.';P=torch.ones_like(B).triu(-A.block_length).tril(A.block_length);B=B.masked_fill(P==0,-1e4)
		G=F.softmax(B,dim=-1);G=A.drop(G);H=torch.matmul(G,K)
		if A.window_size is not _A:Q=A._absolute_position_to_relative_position(G);R=A._get_relative_embeddings(A.emb_rel_v,C);H=H+A._matmul_with_relative_values(Q,R)
		H=H.transpose(2,3).contiguous().view(J,L,E);return H,G
	def _matmul_with_relative_values(B,x,y):A=torch.matmul(x,y.unsqueeze(0));return A
	def _matmul_with_relative_keys(B,x,y):A=torch.matmul(x,y.unsqueeze(0).transpose(-2,-1));return A
	def _get_relative_embeddings(A,relative_embeddings,length):
		D=relative_embeddings;B=length;J=2*A.window_size+1;C=max(B-(A.window_size+1),0);E=max(A.window_size+1-B,0);H=E+2*B-1
		if C>0:G=F.pad(D,commons__convert_pad_shape([[0,0],[C,C],[0,0]]))
		else:G=D
		I=G[:,E:H];return I
	def _relative_position_to_absolute_position(G,x):C,D,A,H=x.size();x=F.pad(x,commons__convert_pad_shape([[0,0],[0,0],[0,0],[0,1]]));B=x.view([C,D,A*2*A]);B=F.pad(B,commons__convert_pad_shape([[0,0],[0,0],[0,A-1]]));E=B.view([C,D,A+1,2*A-1])[:,:,:A,A-1:];return E
	def _absolute_position_to_relative_position(G,x):C,D,A,H=x.size();x=F.pad(x,commons__convert_pad_shape([[0,0],[0,0],[0,0],[0,A-1]]));B=x.view([C,D,A**2+A*(A-1)]);B=F.pad(B,commons__convert_pad_shape([[0,0],[0,0],[A,0]]));E=B.view([C,D,A,2*A])[:,:,:,1:];return E
	def _attention_bias_proximal(C,length):A=torch.arange(length,dtype=torch.float32);B=torch.unsqueeze(A,0)-torch.unsqueeze(A,1);return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(B)),0),0)
class FFN(nn.Module):
	def __init__(A,in_channels,out_channels,filter_channels,kernel_size,p_dropout=_D,activation=_A,causal=_B):
		D=causal;E=p_dropout;F=out_channels;G=in_channels;B=kernel_size;C=filter_channels;super().__init__();A.in_channels=G;A.out_channels=F;A.filter_channels=C;A.kernel_size=B;A.p_dropout=E;A.activation=activation;A.causal=D
		if D:A.padding=A._causal_padding
		else:A.padding=A._same_padding
		A.conv_1=nn.Conv1d(G,C,B);A.conv_2=nn.Conv1d(C,F,B);A.drop=nn.Dropout(E)
	def forward(A,x,x_mask):
		B=x_mask;x=A.conv_1(A.padding(x*B))
		if A.activation=='gelu':x=x*torch.sigmoid(1.702*x)
		else:x=torch.relu(x)
		x=A.drop(x);x=A.conv_2(A.padding(x*B));return x*B
	def _causal_padding(A,x):
		if A.kernel_size==1:return x
		B=A.kernel_size-1;C=0;D=[[0,0],[0,0],[B,C]];x=F.pad(x,commons__convert_pad_shape(D));return x
	def _same_padding(A,x):
		if A.kernel_size==1:return x
		B=(A.kernel_size-1)//2;C=A.kernel_size//2;D=[[0,0],[0,0],[B,C]];x=F.pad(x,commons__convert_pad_shape(D));return x
def commons__init_weights(m,mean=_D,std=.01):
	A=m.__class__.__name__
	if A.find('Conv')!=-1:m.weight.data.normal_(mean,std)
def commons__get_padding(kernel_size,dilation=1):A=dilation;return int((kernel_size*A-A)/2)
def commons__convert_pad_shape(pad_shape):A=pad_shape;B=A[::-1];A=[B for A in B for B in A];return A
def commons__intersperse(lst,item):A=[item]*(len(lst)*2+1);A[1::2]=lst;return A
def sequence_mask(length,max_length=_A):
	A=max_length;B=length
	if A is _A:A=B.max()
	C=torch.arange(A,dtype=B.dtype,device=_DEVICE);return C.unsqueeze(0)<B.unsqueeze(1)
chinese_bert__dirnow=os.path.dirname(__file__)
chinese_bert__device=torch.device(_DEVICE)
chinese_bert__model_dir=chinese_bert__dirnow+'/mods/chinese-roberta-wwm-ext-large'
chinese_bert__tokenizer=AutoTokenizer.from_pretrained(chinese_bert__model_dir)
chinese_bert__model=AutoModelForMaskedLM.from_pretrained(chinese_bert__model_dir).to(_DEVICE,dtype=torch.float32)
def chinese_bert__get_bert_feature(text,word2ph):
	D=word2ph;B=chinese_bert__tokenizer(text,return_tensors='pt')
	for A in B:B[A]=B[A].to(chinese_bert__device)
	C=chinese_bert__model(**B,output_hidden_states=_E);C=torch.cat(C['hidden_states'][-3:-2],-1)[0].to(_DEVICE);assert len(D)==len(text)+2;E=D;F=[]
	for A in range(len(E)):G=C[A].repeat(E[A],1);F.append(G)
	return torch.cat(F,dim=0).T
symbols__punctuation=['!','?','…',',','.',_C,'-']
symbols__pu_symbols=symbols__punctuation+['SP','UNK']
symbols__pad='_'
symbols__zh_symbols=['E','En','a','ai','an','ang','ao','b','c','ch','d','e','ei','en','eng','er','f','g','h','i','i0','ia','ian','iang','iao','ie','in','ing','iong','ir','iu','j','k','l','m','n','o','ong','ou','p','q','r','s','sh','t','u','ua','uai','uan','uang','ui','un','uo','v','van','ve','vn','w','x','y','z','zh','AA','EE','OO']
symbols__num_zh_tones=6
symbols__ja_symbols=['I','N','U','a','b','by','ch','cl','d','dy','e','f','g','gy','h','hy','i','j','k','ky','m','my','n','ny','o','p','py','r','ry','s','sh','t','ts','u','V','w','y','z']
symbols__num_ja_tones=1
symbols__en_symbols=['aa','ae','ah','ao','aw','ay','b','ch','d','dh','eh','er','ey','f','g','hh','ih','iy','jh','k','l','m','n','ng','ow','oy','p','r','s','sh','t','th','uh','uw','V','w','y','z','zh']
symbols__num_en_tones=4
symbols__normal_symbols=sorted(set(symbols__zh_symbols+symbols__ja_symbols+symbols__en_symbols))
symbols__symbols=[symbols__pad]+symbols__normal_symbols+symbols__pu_symbols
symbols__sil_phonemes_ids=[symbols__symbols.index(A)for A in symbols__pu_symbols]
symbols__num_tones=symbols__num_zh_tones+symbols__num_ja_tones+symbols__num_en_tones
symbols__language_id_map={_G:0,'JA':1,'EN':2}
symbols__num_languages=len(symbols__language_id_map.keys())
symbols__language_tone_start_map={_G:0,'JA':symbols__num_zh_tones,'EN':symbols__num_zh_tones+symbols__num_ja_tones}
chinese__current_file_path=os.path.dirname(__file__)
chinese__pinyin_to_symbol_map={A.split('\t')[0]:A.strip().split('\t')[1]for A in open(os.path.join(chinese__current_file_path,'opencpop-strict.txt')).readlines()}
chinese__rep_map={'：':',','；':',','，':',','。':'.','！':'!','？':'?','\n':'.','·':',','、':',','...':'…','$':'.','“':_C,'”':_C,'‘':_C,'’':_C,'（':_C,'）':_C,'(':_C,')':_C,'《':_C,'》':_C,'【':_C,'】':_C,'[':_C,']':_C,'—':'-','～':'-','~':'-','「':_C,'」':_C}
def cleaner__clean_text(text,_):
	F='';C=text;Y=re.findall('\\d+(?:\\.?\\d+)?',C)
	for R in Y:C=C.replace(R,cn2an.an2cn(R),1)
	C=C.replace('嗯','恩').replace('呣','母');M=re.compile('|'.join(re.escape(A)for A in chinese__rep_map.keys()));C=M.sub(lambda x:chinese__rep_map[x.group()],C);N=re.sub('[^\\u4e00-\\u9fa5'+F.join(symbols__punctuation)+']+',F,C);M='(?<=[{0}])\\s*'.format(F.join(symbols__punctuation));Z=[A for A in re.split(M,N)if A.strip()!=F];G=[];H=[];D=[]
	for I in Z:
		I=re.sub('[a-zA-Z]+',F,I);O=psg.lcut(I);J=[];K=[];O=O
		for(S,a)in O:
			if a=='eng':continue
			T=[];U=[];b=lazy_pinyin(S,neutral_tone_with_five=_E,style=Style.INITIALS);c=lazy_pinyin(S,neutral_tone_with_five=_E,style=Style.FINALS_TONE3)
			for(B,E)in zip(b,c):T.append(B);U.append(E)
			J.append(T);K.append(U)
		J=sum(J,[]);K=sum(K,[])
		for(B,E)in zip(J,K):
			d=B+E
			if B==E:assert B in symbols__punctuation;L=[B];P='0';D.append(1)
			else:
				Q=E[:-1];P=E[-1];A=B+Q;assert P in'12345'
				if B:
					V={'uei':'ui','iou':'iu','uen':'un'}
					if Q in V.keys():A=B+V[Q]
				else:
					W={'ing':'ying','i':'yi','in':'yin','u':'wu'}
					if A in W.keys():A=W[A]
					else:
						X={'v':'yu','e':'e','i':'y','u':'w'}
						if A[0]in X.keys():A=X[A[0]]+A[1:]
				assert A in chinese__pinyin_to_symbol_map.keys(),(A,I,d);L=chinese__pinyin_to_symbol_map[A].split(' ');D.append(len(L))
			G+=L;H+=[int(P)]*len(L)
	assert sum(D)==len(G);assert len(D)==len(N);G=['_']+G+['_'];H=[0]+H+[0];D=[1]+D+[1];return N,G,H,D
text__lang_bert_func_map={_G:chinese_bert__get_bert_feature}
text___symbol_to_id={B:A for(A,B)in enumerate(symbols__symbols)}
def text__cleaned_text_to_sequence(cleaned_text,tones,language):B=language;A=tones;C=[text___symbol_to_id[A]for A in cleaned_text];D=symbols__language_tone_start_map[B];A=[A+D for A in A];E=symbols__language_id_map[B];F=[E for A in C];return C,A,F
dirnow=os.path.dirname(__file__)
sys.path.append(dirnow)
def get_text(text,language_str,hps):
	E=language_str;H,A,B,C=cleaner__clean_text(text,E);A,B,D=text__cleaned_text_to_sequence(A,B,E)
	if hps.data.add_blank:
		A=commons__intersperse(A,0);B=commons__intersperse(B,0);D=commons__intersperse(D,0)
		for F in range(len(C)):C[F]=C[F]*2
		C[0]+=1
	G=text__lang_bert_func_map[E](H,C);del C;assert G.shape[-1]==len(A);A=torch.LongTensor(A);B=torch.LongTensor(B);D=torch.LongTensor(D);return G,A,B,D
def infer(text,sid,sdp_ratio=0,noise_scale=.667,noise_scale_w=.8,length_scale=1):global net_g;A,D,B,C=get_text(text,_G,hps);E=D.to(_DEVICE).unsqueeze(0);B=B.to(_DEVICE).unsqueeze(0);C=C.to(_DEVICE).unsqueeze(0);A=A.unsqueeze(0);print(E,B,C,A.shape);F=torch.LongTensor([D.size(0)]).to(_DEVICE);del D;G=torch.LongTensor([hps.data.spk2id[sid]]).to(_DEVICE);H=net_g.infer(E,F,G,B,C,A,sdp_ratio=sdp_ratio,noise_scale=noise_scale,noise_scale_w=noise_scale_w,length_scale=length_scale)[0,0].data.cpu().numpy();del E,B,C,A,F,G;return H
model_dir=dirnow+'/mods/hoyoTTS'
config_save_path=os.path.join(model_dir,'config.json')
with open(config_save_path,'r',encoding='utf-8')as f:data=f.read()
config=json.loads(data)
hparams=HParams(**config)
hparams.model_dir=model_dir
hps=hparams
net_g=SynthesizerTrn(len(symbols__symbols),hps.data.filter_length//2+1,hps.train.segment_size//hps.data.hop_length,n_speakers=hps.data.n_speakers,**hps.model)
net_g.eval()
utils__load_checkpoint(f"{model_dir}/G_78000.pth",net_g,_A,skip_optimizer=_E)
net_g.to(_DEVICE)
speaker_ids=hps.data.spk2id
speakers=list(speaker_ids.keys())
def np2wav(nparray,framerate=44100):
	C=nparray;B=np.int16(C/np.max(np.abs(C))*32767);D=io.BytesIO()
	with wave.open(D,'wb')as A:A.setnchannels(1);A.setsampwidth(2);A.setframerate(framerate);B=B.astype(np.int16);A.writeframes(B.tobytes())
	return D.getvalue()
print("infer first time.")
open("./w.wav", "wb").write(np2wav(infer("你好!", "可莉")))
print("finish.")

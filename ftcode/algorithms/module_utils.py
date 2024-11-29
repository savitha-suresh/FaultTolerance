import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import repeat


class CriticMLP(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n, hidden_size):
        super(CriticMLP, self).__init__()
        self.obs_shape_n = obs_shape_n
        self.action_shape_n = action_shape_n
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n + obs_shape_n, hidden_size)
        self.linear_c2 = nn.Linear(hidden_size, hidden_size)
        self.linear_c = nn.Linear(hidden_size, 1)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        x_cat = self.LReLU(self.linear_c1(
            torch.cat([obs_input[:, 0: self.obs_shape_n], action_input[:, 0: self.action_shape_n]], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value


class ActorMLP(nn.Module):
    def __init__(self, num_inputs, action_size, hidden_size):
        super(ActorMLP, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, hidden_size)
        self.linear_a2 = nn.Linear(hidden_size, hidden_size)
        self.linear_a = nn.Linear(hidden_size, action_size)

        self.reset_parameters()
        self.train()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, input, model_original_out=False):
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out:   return model_out, policy  # for model_out criterion
        return policy


class ScaledDotProductAttention(nn.Module):

    def forward(self, query, key, value, mask=None):
        dk = query.size()[-1]
        scores = query.matmul(key.transpose(-2, -1)) / math.sqrt(dk)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = F.softmax(scores, dim=-1)
        return attention.matmul(value)


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 in_features,
                 head_num,
                 bias=True,
                 activation=F.relu):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        if in_features % head_num != 0:
            raise ValueError('`in_features`({}) should be divisible by `head_num`({})'.format(in_features, head_num))
        self.in_features = in_features
        self.head_num = head_num
        self.activation = activation
        self.bias = bias
        self.linear_q = nn.Linear(in_features, in_features, bias)
        self.linear_k = nn.Linear(in_features, in_features, bias)
        self.linear_v = nn.Linear(in_features, in_features, bias)
        self.linear_o = nn.Linear(in_features, in_features, bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_q.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear_k.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear_v.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear_o.weight, mean=0, std=0.1)

    def forward(self, q, k, v, mask=None):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        y = ScaledDotProductAttention()(q, k, v, mask)
        y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.activation is not None:
            y = self.activation(y)
        return y

    def scores(self, q, k, v):
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        dk = q.size()[-1]
        scores = q.matmul(k.transpose(-2, -1)) / math.sqrt(dk)
        attentions = F.softmax(scores, dim=-1)
        return attentions

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return torch.tril(torch.ones(seq_len, seq_len)).view(1, seq_len, seq_len).repeat(batch_size, 1, 1)

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return x.reshape(batch_size, seq_len, self.head_num, sub_dim) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size * self.head_num, seq_len, sub_dim)

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return x.reshape(batch_size, self.head_num, seq_len, in_feature) \
            .permute(0, 2, 1, 3) \
            .reshape(batch_size, seq_len, out_dim)

    def extra_repr(self):
        return 'in_features={}, head_num={}, bias={}, activation={}'.format(
            self.in_features, self.head_num, self.bias, self.activation,
        )


class Coder(nn.Module):
    def __init__(self, input_shape, output_shape):
        super(Coder, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(input_shape, max(input_shape, output_shape))
        self.linear_c2 = nn.Linear(max(input_shape, output_shape), output_shape)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.linear_c1.weight, mean=0, std=0.1)
        nn.init.normal_(self.linear_c2.weight, mean=0, std=0.1)

    def forward(self, x):
        x = self.LReLU(self.linear_c1(x))
        x = self.linear_c2(x)
        return x


class CriticAttention(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(CriticAttention, self).__init__()
        self.agents_n = len(obs_shape_n)
        self.features_n = args.critic_features_num
        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        for i in range(self.agents_n):
            self.encoder.append(Coder(obs_shape_n[i] + action_shape_n[i], self.features_n).to(args.device))
            self.decoder.append(Coder(self.features_n, 1).to(args.device))

        self.attention = MultiHeadAttention(in_features=self.features_n, head_num=1)
        self.obs_size = []
        self.action_size = []
        head_o, head_a, end_o, end_a = 0, 0, 0, 0
        for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
            end_o = end_o + obs_shape
            end_a = end_a + action_shape
            range_o = (head_o, end_o)
            range_a = (head_a, end_a)
            self.obs_size.append(range_o)
            self.action_size.append(range_a)
            head_o = end_o
            head_a = end_a

    def forward(self, obs_input, action_input):
        f_ = []
        for i in range(self.agents_n):
            t = self.encoder[i](torch.cat([obs_input[:, self.obs_size[i][0]:self.obs_size[i][1]], action_input[:, self.action_size[i][0]:self.action_size[i][1]]], dim=1))
            f_.append(t)
        f = torch.cat(f_, dim=1).reshape(-1, self.agents_n, self.features_n)

        values = self.attention(f, f, f)
        out = []
        for i in range(self.agents_n):
            out.append(self.decoder[i](values[:, i]))
        return torch.cat(out, dim=1)

    def attn_mat(self, obs_input, action_input):
        f_ = []
        for i in range(self.agents_n):
            t = self.encoder[i](torch.cat([obs_input[:, self.obs_size[i][0]:self.obs_size[i][1]], action_input[:, self.action_size[i][0]:self.action_size[i][1]]], dim=1))
            f_.append(t)
        f = torch.cat(f_, dim=1).reshape(-1, self.agents_n, self.features_n)

        return self.attention.scores(f, f, f)


class ActorAttention(nn.Module):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(ActorAttention, self).__init__()
        self.features_n = args.actor_features_num
        self.obs_size = args.actor_obs_size
        self.encoder = nn.ModuleList([Coder((item[1] - item[0]), self.features_n).to(args.device) for item in self.obs_size])

        self.attention = MultiHeadAttention(in_features=self.features_n, head_num=args.head_num)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.features_n), requires_grad=False)

        self.linear_head = nn.Linear(self.features_n, 5)

    def forward(self, obs_input, model_original_out=False):
        f_ = []
        if obs_input.ndim == 1:
            obs_input = obs_input.unsqueeze(0)
        for i in range(len(self.obs_size)):
            f_.append(self.encoder[i](obs_input[:, self.obs_size[i][0]:self.obs_size[i][1]]))

        f = torch.cat(f_, dim=1).reshape(-1, len(self.obs_size), self.features_n)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=f.shape[0]).clone()
        f = torch.cat((cls_tokens, f), dim=1)

        x = self.attention(f, f, f)
        x = x[:, 0]
        model_out = self.linear_head(x).squeeze(0)

        u = torch.rand_like(model_out)
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out:  return model_out, policy # for model_out criterion

        return policy

    def attn_mat(self, obs_input):
        f_ = []
        if obs_input.ndim == 1:
            obs_input = obs_input.unsqueeze(0)
        for i in range(len(self.obs_size)):
            f_.append(self.encoder[i](obs_input[:, self.obs_size[i][0]:self.obs_size[i][1]]))
        x = torch.cat(f_, dim=1).reshape(-1, len(self.obs_size), self.features_n)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0]).clone()
        x = torch.cat((cls_tokens, x), dim=1)
        return self.attention.scores(x, x, x)


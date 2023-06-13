"""
Various helper network modules
"""
import torch
import torch.nn as nn
import torch.distributions as distributions

class RealNVP(distributions.Transform):
    def __init__(self, input_size, hidden_size, device):
        super(RealNVP, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        # Initialize networks
        self.scale = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size//2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size//2),
                nn.Tanh()
            ),
            nn.Sequential(
                nn.Linear(input_size//2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size//2),
                nn.Tanh()
            )
        ]).to(device)
        self.translation = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size//2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size//2)
            ),
            nn.Sequential(
                nn.Linear(input_size//2, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, input_size//2)
            )
        ]).to(device)

    def forward(self, x):
        # Apply transformation to first half of input tensor
        if self.input_size == 0:
            x_even = x
        else:
            x_even = x[:, 1:]
        x_a, x_b = x_even[:, :self.input_size//2], x_even[:, self.input_size//2:]
        scale_1 = self.scale[0](x_a)
        translation_1 = self.translation[0](x_a)
        y_b = (x_b * torch.exp(scale_1)) + translation_1
        y_a = x_a

        # Apply transformation to second half of transformed input tensor
        scale_2 = self.scale[1](y_b)
        translation_2 = self.translation[1](y_b)
        y_a = (y_a * torch.exp(scale_2)) + translation_2

        # Combine transformed and untransformed half of input tensor
        y = torch.zeros_like(x, device=x.device)
        if self.input_size == 0:
            y[:, :self.input_size//2] = y_a
            y[:, self.input_size//2:] = y_b
        else:
            y[:, 0] = x[:, 0]
            y[:, 1:self.input_size//2+1] = y_a
            y[:, self.input_size//2+1:] = y_b

        # Compute log determinant of Jacobian
        log_det_J = torch.sum(scale_1, dim=1) + torch.sum(scale_2, dim=1)

        return y, log_det_J

    def backward(self, y):
        # Apply inverse transformation to second half of input tensor
        if self.input_size == 0:
            y_even = y
        else:
            y_even = y[:, 1:]
        y_a, y_b = y_even[:, :self.input_size//2], y_even[:, self.input_size//2:]
        
        scale_2 = self.scale[1](y_b)
        translation_2 = self.translation[1](y_b)
        y_a = (y_a - translation_2) * torch.exp(-scale_2)

        # Apply inverse transformation to first half of transformed input tensor
        x_a = y_a
        scale_1 = self.scale[0](x_a)
        translation_1 = self.translation[0](x_a)
        x_b = (y_b - translation_1) * torch.exp(-scale_1)

        # Combine transformed and untransformed half of input tensor
        x = torch.zeros_like(y, device=y.device)
        if self.input_size == 0:
            x[:, :self.input_size//2] = x_a
            x[:, self.input_size//2:] = x_b
        else:
            x[:, 0] = y[:, 0]
            x[:, 1:self.input_size//2+1] = x_a
            x[:, self.input_size//2+1:] = x_b
        
        # Compute log determinant of Jacobian
        log_det_J = -torch.sum(scale_1, dim=1) - torch.sum(scale_2, dim=1)

        return x, log_det_J

    def log_abs_det_jacobian(self, x, y):
        _, log_det_J = self.realnvp.forward(x)
        return log_det_J



class TransformedDistribution(distributions.Distribution):
    def __init__(self, input_size, hidden_size, device):
        super(TransformedDistribution, self).__init__(validate_args=False)
        
        self.realnvp_transform = RealNVP(input_size, hidden_size, device)
        self.gaussian = distributions.Normal(torch.zeros(input_size).to(device), torch.ones(input_size).to(device))

    def sample(self, sample_shape=(1,)):
        x = self.gaussian.sample(sample_shape)
        y, _ = self.realnvp_transform.forward(x)
        return y

    def log_prob(self, value):
        x, log_det_J = self.realnvp_transform.backward(value)
        log_prob = self.gaussian.log_prob(x).sum(dim=1) + log_det_J
        return torch.reshape(log_prob, (-1, 1))

class ParamMultivariateNormal(nn.Module):
    def __init__(self, dim, device):
        super(ParamMultivariateNormal, self).__init__()

        self.dim = dim
        self.device = device
        self.raw_L = nn.Parameter(torch.eye(dim, device=device))

        rows, cols = torch.tril_indices(self.dim, self.dim, device=device)
        self.L_mask = torch.zeros_like(self.raw_L, device=device)
        self.L_mask[rows, cols] = 1

    def sample(self, sample_shape=(1,)):

        L = torch.mul(self.raw_L, self.L_mask)
        covariance = torch.matmul(L, L.t())
        gaussian = distributions.MultivariateNormal(torch.zeros(self.dim).to(self.device), covariance.to(self.device))
        u = gaussian.sample(sample_shape)
        return u

    def log_prob(self, value):
        L = torch.mul(self.raw_L, self.L_mask)
        covariance = torch.matmul(L, L.t())
        gaussian = distributions.MultivariateNormal(torch.zeros(self.dim).to(self.device), covariance.to(self.device))
        log_prob = gaussian.log_prob(value)
        return torch.reshape(log_prob, (-1, 1))

class CustomSpectralLinear(nn.Module):
    def __init__(self, in_features, out_features, a_dim=2, bias=True, n_power_iterations=1):
        super(CustomSpectralLinear, self).__init__()
        self.s_linear = nn.utils.parametrizations.spectral_norm(nn.Linear(in_features-a_dim, out_features, bias), n_power_iterations=n_power_iterations)
        self.a_linear = nn.Linear(a_dim, out_features, bias=False)
        self.a_dim = a_dim

    def forward(self, x):

        # pass the action columns through the a_liner layer
        a = self.a_linear(x[:, :self.a_dim])
        s = self.s_linear(x[:, self.a_dim:])
        return a + s

class MultiplyConstantLayer(nn.Module):
    def __init__(self, constant):
        super(MultiplyConstantLayer, self).__init__()
        self.constant = constant

    def forward(self, x):
        return x * self.constant.to(x.device)

class NeuralRegressor(nn.Module):
    """
    neural network with nin inputs, nout outputs, nh hidden units per layer, nl layers, and leaky relu activation functions
    if lipschitz is not None, layers include spectral normalization
    if positive is True, the last activation function is a softplus
    """
    def __init__(self, nin, nout, nl, nh, lipschitz=None, positive=False, a_dim=2, n_power_iterations=1):
        super().__init__()
        
        if lipschitz is not None:
            multiplier = torch.pow(torch.tensor([lipschitz]), 1/(nl+1))
        
        if nl==0:
            if lipschitz is None:
                layers = [nn.Linear(nin, nout)]
            else:
                layers = [CustomSpectralLinear(nin, nout, a_dim=a_dim, n_power_iterations=n_power_iterations), MultiplyConstantLayer(multiplier)]
            if positive:
                layers.append(nn.Softplus())
        elif nl>0:
            if lipschitz is None:
                layers = [nn.Linear(nin, nh), nn.Tanh()]
            else:
                layers = [CustomSpectralLinear(nin, nh, a_dim=a_dim, n_power_iterations=n_power_iterations), MultiplyConstantLayer(multiplier), nn.Tanh()]
            for _ in range(nl-1):
                if lipschitz is None:
                    layers += [nn.Linear(nh, nh), nn.Tanh()]
                else:
                    layers += [nn.utils.parametrizations.spectral_norm(nn.Linear(nh, nh), n_power_iterations=n_power_iterations), MultiplyConstantLayer(multiplier), nn.Tanh()]
            if lipschitz is None:
                layers += [nn.Linear(nh, nout)]
            else:
                layers += [nn.utils.parametrizations.spectral_norm(nn.Linear(nh, nout), n_power_iterations=n_power_iterations), MultiplyConstantLayer(multiplier)]
            
            if positive:
                layers.append(nn.Softplus())

        self.lipschitz = lipschitz
        self.net = nn.Sequential(*layers)

    def forward(self, x):
            return self.net(x)

class SCM(nn.Module):
    """
    A single affine transformation SCM with a normal prior 
    """
    def __init__(self, s_dim, nl, nh, a_dim, c_dim, lipschitz_loc=None, lipschitz_scale=None, prior_type='gaussian', device='cpu', n_power_iterations=1):
        super().__init__()
        # initialize the NN models
        self.a_dim = a_dim
        self.s_dim_actionable = s_dim - c_dim    # the first three features are not actionable in MIMIC (gender, re_admission, and age)
        self.location_model = NeuralRegressor(a_dim + s_dim, self.s_dim_actionable, nl, nh, lipschitz=lipschitz_loc, a_dim=a_dim, n_power_iterations=n_power_iterations)
        self.scale_model = NeuralRegressor(a_dim + s_dim, self.s_dim_actionable, nl, nh, lipschitz=lipschitz_scale, positive=True, a_dim=a_dim, n_power_iterations=n_power_iterations)

        # initialize the prior distribution to a zero mean isotropic Gaussian
        if prior_type == 'gaussian':
            self.prior = torch.distributions.Normal(torch.zeros(self.s_dim_actionable).to(device), torch.ones(self.s_dim_actionable).to(device))
        elif prior_type == 'laplace':
            self.prior = torch.distributions.Laplace(torch.zeros(self.s_dim_actionable).to(device), torch.ones(self.s_dim_actionable).to(device))
        elif prior_type == 'multigaussian':
            self.prior = ParamMultivariateNormal(self.s_dim_actionable, device)

    def forward(self, s, a, u):
        # compute the location and scale of the distribution
        s_prime = torch.zeros_like(s, device=s.device)
        s_prime[:, :-self.s_dim_actionable] = s[:, :-self.s_dim_actionable]
        concat_input = torch.cat([a, s], dim=1)
        location = self.location_model(concat_input)
        scale = self.scale_model(concat_input)

        s_prime[:,-self.s_dim_actionable:] = scale * u + location
        return s_prime

    def backward(self, s, a, s_prime):
        # compute the location and scale of the distribution
        u = torch.zeros(s.shape[0], self.s_dim_actionable, device=s.device)
        concat_input = torch.cat([a, s], dim=1)
        location = self.location_model(concat_input)
        scale = self.scale_model(concat_input)
        
        u = (s_prime[:, -self.s_dim_actionable:] - location) / scale
        log_det =  - torch.sum(torch.log(scale), dim=1)
        return u, log_det

    def sample(self, s, a):
        u = self.prior.sample((s.shape[0], ))
        s_prime = self.forward(s, a, u)
        
        return s_prime

    def log_likelihood(self, s, a, s_prime):
        
        u, log_det = self.backward(s, a, s_prime)
        prior_logprob = self.prior.log_prob(u).sum(dim=1)
        return prior_logprob+log_det

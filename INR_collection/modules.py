import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ImplicitMLPLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                omega_0=1, w_norm=False, activation="relu", omega_uniform=False,
                film_conditioning=False, concat_conditioning=0,
                init_method={"weights": 'basic', "bias": "zero"}):
        super().__init__()

        self.omega_0 = nn.Parameter(torch.ones(out_features, requires_grad=False)*omega_0,requires_grad=False)

        if omega_uniform:
            omegas = torch.sort(torch.rand(out_features, requires_grad=False)*omega_0/in_features)
            self.omega_0 = (nn.Parameter(omegas[0],requires_grad=False))

        self.in_features = in_features
        self.out_features = out_features
        self.film_conditioning = film_conditioning
        self.concat_conditioning = concat_conditioning
        if concat_conditioning:
            self.in_features += concat_conditioning 

        self.linear = nn.Linear(self.in_features, out_features, bias=bias)

        if activation == "relu":
            self.activation = nn.LeakyReLU(negative_slope=0.02)
        if activation == "sine":
            self.activation = torch.sin
        if activation == "sigmoid":
            self.activation = nn.Sigmoid()
        if activation == "tanh":
            self.activation = nn.Tanh()
        if activation == "none":
            self.activation = self.return_input

        with torch.no_grad():
            self.init_weights(init_method)
            if w_norm:
                self.linear = torch.nn.utils.weight_norm(self.linear)

    def return_input(self, input):
        return input

    def init_weights(self, init_method):
        with torch.no_grad():
            if init_method["weights"] == "basic":
                # Values taken from IM-NET
                nn.init.normal_(self.linear.weight, mean=0.0, std=0.02)
            if init_method["weights"] == "kaiming_in":
                nn.init.kaiming_normal_(self.linear.weight, mode='fan_in', nonlinearity='leaky_relu', a=0.02)
                with torch.no_grad():
                    self.linear.weight /= self.omega_0
            if init_method["weights"] == "siren":
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0.abs().mean(), 
                                            np.sqrt(6 / self.in_features) / self.omega_0.abs().mean())
            if init_method["weights"] == "siren_omega":
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / init_method["omega"], 
                                            np.sqrt(6 / self.in_features) / init_method["omega"])
            if init_method["weights"] == "siren_first":
                self.linear.weight.uniform_(-1 / self.in_features, 
                                            1 / self.in_features)   
            if init_method["weights"] == "none":
                pass
            
            if init_method["bias"] == "zero":
                nn.init.constant_(self.linear.bias, 0)
            if init_method["bias"] == "polar":
                self.linear.bias.uniform_(0, 2*np.pi)
            if init_method["bias"] == "none":
                pass
    
    def forward(self, layer_input, z=None, gamma=None, beta=None, delta=None, progress=None):
        if self.concat_conditioning:
            if z.shape[1] !=  layer_input.shape[1]: 
                z = z.repeat(1, layer_input.shape[1], 1)
            layer_input = torch.cat((layer_input, z), dim=-1)

        if self.film_conditioning:
            self.feat_multiplier = gamma[:, :,  :self.out_features] + self.omega_0
            self.feat_bias = beta[:, :, :self.out_features]

        else:
            self.feat_multiplier = self.omega_0
            self.feat_bias = 0

        output = self.activation((self.feat_multiplier * self.linear(layer_input)) + self.feat_bias)
        if delta is not None:
            output = output * delta
        return output

class SIREN(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                num_layers=3, num_hidden=256, 
                first_omega_0=30, hidden_omega_0=30):
        super().__init__()
        self.net = []
        self.net.append(ImplicitMLPLayer(in_features, num_hidden, bias=True,
                        omega_0=first_omega_0, w_norm=False, activation="sine", 
                        film_conditioning=False, concat_conditioning=0,
                        init_method={"weights": 'siren_first', "bias": "polar"}))
        for i in range(num_layers-1):
            self.net.append(ImplicitMLPLayer(num_hidden, num_hidden, bias=True,
                            omega_0=hidden_omega_0, w_norm=False, activation="sine", 
                            film_conditioning=False, concat_conditioning=0,
                            init_method={"weights": 'siren', "bias": "polar"}))
        self.net.append(ImplicitMLPLayer(num_hidden, out_features, bias=True,
                omega_0=1, w_norm=False, activation="none", 
                film_conditioning=False, concat_conditioning=0,
                init_method={"weights": 'siren_omega', "omega":30, "bias": "none"}))
        self.net = nn.Sequential(*self.net)

    def forward(self, x):
        return self.net(x)

class PiGANMappingNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, depth=3):
        super().__init__()
        layers = []

        if depth == 0:
            layers.extend([nn.Linear(dim_in, dim_out)]) 
        else:
            layers.extend([nn.Linear(dim_in, dim_hidden), nn.LeakyReLU(0.2, inplace=True)]) 
            nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)
            for i in range(depth-1):
                layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU(0.2, inplace=True)]) 
                nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)
            layers.extend([nn.Linear(dim_hidden, dim_out)]) 
            nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)

        self.net = nn.Sequential(*layers)
        
        with torch.no_grad():
            self.net[-1].weight *= 0.25

    def forward(self, x):
        x = self.net(x)
        gamma, beta = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
        return gamma, beta

class ConcatMappingNetwork(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, depth=3):
        super().__init__()
        layers = []
        layers.extend([nn.Linear(dim_in, dim_hidden), nn.LeakyReLU(0.2, inplace=True)]) 
        nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)
        for i in range(depth-1):
            layers.extend([nn.Linear(dim_hidden, dim_hidden), nn.LeakyReLU(0.2, inplace=True)]) 
            nn.init.kaiming_normal_(layers[-2].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)
        layers.extend([nn.Linear(dim_hidden, dim_out)]) 
        nn.init.kaiming_normal_(layers[-1].weight, mode='fan_in', nonlinearity='leaky_relu', a = 0.2)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

#=== Decoders =================================================================
"""General forward input is coordinates, latent vector"""
class IMNET(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                num_layers=5, initial_hidden=2048, z_size=256, coord_multi=900,
                siren_first=False, siren_hidden=False, siren_final=False,
                init_method={"weights": 'basic', "bias": "zero"}):
        super().__init__()
        self.coord_multi = coord_multi
        self.net = []

        if siren_first:
            self.net.append(ImplicitMLPLayer(1, initial_hidden, activation="sine", omega_0=30,
                                            concat_conditioning=z_size, init_method={"weights": 'siren_first', "bias": "polar"}))
        else:
            self.net.append(ImplicitMLPLayer(1, initial_hidden, activation="relu",
                                        concat_conditioning=z_size, init_method=init_method))
        
        for i, layer in enumerate(range(num_layers-1)):
            if siren_hidden:
                self.net.append(ImplicitMLPLayer(initial_hidden//(2**i), initial_hidden//(2**(i+1)), activation="sine", omega_0=30,
                                                concat_conditioning=z_size + 1, init_method={"weights": 'siren', "bias": "polar"}))
            else:
                self.net.append(ImplicitMLPLayer(initial_hidden//(2**i), initial_hidden//(2**(i+1)), activation="relu",
                                                concat_conditioning=z_size + 1, init_method=init_method))
        
        if siren_final:
            self.net.append(ImplicitMLPLayer(initial_hidden//(2**(i+1)), out_features,
                                            concat_conditioning=0, init_method={"weights": 'siren_omega', "omega":30, "bias": "none"},
                                            activation="none"))
        else:
            self.net.append(ImplicitMLPLayer(initial_hidden//(2**(i+1)), out_features,
                                            concat_conditioning=0, init_method=init_method,
                                            activation="tanh"))
    
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coordinates, z):
        coordinates = coordinates * self.coord_multi
        coordinates = coordinates.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        z = z.repeat(1, coordinates.shape[1], 1)
        output = self.net[0](coordinates, z=z)
        for layer in self.net[1:]:
            if layer.concat_conditioning:
                output = layer(output, z=torch.cat((z, coordinates), dim=-1))
            else:
                output = layer(output)
        return output

class piGAN(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256, num_hidden_INR=256, 
                z_size=256, first_omega_0=600, hidden_omega_0=30):
        super().__init__()
        self.mapping_net = PiGANMappingNetwork(z_size, num_hidden_mapping, num_hidden_mapping*2, depth = num_mapping_layers)
        self.net = []
        self.net.append(ImplicitMLPLayer(in_features, num_hidden_INR, bias=True,
                        omega_0=first_omega_0, w_norm=True, activation="sine", 
                        film_conditioning=True, concat_conditioning=0,
                        init_method={"weights": 'siren_first', "bias": "polar"}))
        for i in range(num_INR_layers-1):
            self.net.append(ImplicitMLPLayer(num_hidden_INR, num_hidden_INR, bias=True,
                            omega_0=hidden_omega_0, w_norm=True, activation="sine", 
                            film_conditioning=True, concat_conditioning=0,
                            init_method={"weights": 'siren', "bias": "polar"}))
        self.net.append(ImplicitMLPLayer(num_hidden_INR, out_features, bias=True,
                omega_0=1, w_norm=True, activation="none", 
                film_conditioning=False, concat_conditioning=0,
                init_method={"weights": 'siren_omega', "omega":30, "bias": "none"}))
        self.net = nn.Sequential(*self.net)

    def forward(self, coordinates, z):
        gamma, beta = self.mapping_net(z)
        output = self.net[0](coordinates, gamma=gamma, beta=beta)
        for layer in self.net[1:]:
            if layer.film_conditioning:
                output = layer(output, gamma=gamma, beta=beta)
            else:
                output = layer(output)
        return output

class SIREN_prog(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                num_layers=3, num_hidden=256, 
                first_omega_0=30, hidden_omega_0=30, total_its=2000, num_groups=8):
        super().__init__()
        self.net = []
        self.net.append(ImplicitMLPLayer(in_features, num_hidden, bias=True,
                        omega_0=first_omega_0, w_norm=True, activation="sine", 
                        film_conditioning=False, concat_conditioning=0, omega_uniform=True,
                        init_method={"weights": 'siren_first', "bias": "polar"}))
        for i in range(num_layers-1):
            self.net.append(ImplicitMLPLayer(num_hidden, num_hidden, bias=True,
                            omega_0=hidden_omega_0, w_norm=False, activation="sine", 
                            film_conditioning=False, concat_conditioning=0,
                            init_method={"weights": 'siren', "bias": "polar"}))
        self.net.append(ImplicitMLPLayer(num_hidden, out_features, bias=True,
                omega_0=1, w_norm=False, activation="none", 
                film_conditioning=False, concat_conditioning=0,
                init_method={"weights": 'siren_omega', "omega":30, "bias": "none"}))
        self.net = nn.Sequential(*self.net)
        
        self.total_its = total_its
        self.num_groups = num_groups
        self.switch_layer = nn.Parameter(torch.zeros(num_hidden), requires_grad=False)
        self.groups = [(i*(num_hidden//num_groups), (i+1)*(num_hidden//num_groups)) for i in range(num_groups)]
        self.buffer_time = (total_its // 2) // ((2 * num_groups))
        self.buffer_group = 0
        self.iteration = 0
        self.state = "buffering"
        print(f"Prog pi-gan with omega: {first_omega_0}, {hidden_omega_0}")

    def switch_state(self):
        if self.buffer_group == self.num_groups-1:
            self.state = "rest"
        elif self.state == "buffering":
            self.state = "rest"
        elif self.state == "rest":
            self.buffer_group += 1
            self.state = "buffering"
            self.plot_moment = True
        print(f"switched to state: {self.state}")

    def step(self):
        if self.state == "buffering":
            index_start, index_end = self.groups[self.buffer_group]
            self.switch_layer[index_start:index_end] = self.switch_layer[index_start:index_end] + (1 / self.buffer_time)
            print(self.switch_layer.mean())
        self.iteration += 1 
        if not (self.iteration % self.buffer_time):
            if self.iteration < (self.total_its // 2):
                self.switch_state()
            else:
                self.state = "rest"
                print(f"state: {self.state}")
        else:
            self.plot_moment = False

    def forward(self, x):
        self.step()
        return self.net(x)

class piGAN_prog(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256, num_hidden_INR=256, 
                z_size=256, first_omega_0=3000, hidden_omega_0=30, total_epochs=5000, num_groups=256):
        super().__init__()
        self.mapping_net = PiGANMappingNetwork(z_size, num_hidden_mapping, num_hidden_mapping*2, depth = num_mapping_layers)
        self.net = []
        self.net.append(ImplicitMLPLayer(in_features, num_hidden_INR, bias=True,
                        omega_0=first_omega_0, w_norm=True, activation="sine", 
                        film_conditioning=True, concat_conditioning=0, omega_uniform=True,
                        init_method={"weights": 'siren_first', "bias": "polar"}))
        for i in range(num_INR_layers-1):
            self.net.append(ImplicitMLPLayer(num_hidden_INR, num_hidden_INR, bias=True,
                            omega_0=hidden_omega_0, w_norm=True, activation="sine", 
                            film_conditioning=True, concat_conditioning=0, omega_uniform=False,
                            init_method={"weights": 'siren', "bias": "polar"}))
        self.net.append(ImplicitMLPLayer(num_hidden_INR, out_features, bias=True,
                omega_0=1, w_norm=True, activation="none", 
                film_conditioning=False, concat_conditioning=0,
                init_method={"weights": 'siren_omega', "omega":30, "bias": "none"}))
        self.net = nn.Sequential(*self.net)

        self.total_epochs = total_epochs
        self.num_groups = num_groups
        self.switch_layer = nn.Parameter(torch.zeros(num_hidden_INR), requires_grad=False)
        self.groups = [(i*(num_hidden_INR//num_groups), (i+1)*(num_hidden_INR//num_groups)) for i in range(num_groups)]
        self.buffer_time = (total_epochs // 2) // ((2 * num_groups))
        self.buffer_group = 0
        self.epoch = 0
        self.state = "buffering"
        self.plot_moment=True
        
        print(f"Prog pi-gan with omega: {first_omega_0}, {hidden_omega_0}")
    # def buffer_step(self):

    def switch_state(self):
        if self.buffer_group == self.num_groups-1:
            self.state = "rest"
        elif self.state == "buffering":
            self.state = "rest"
        elif self.state == "rest":
            self.buffer_group += 1
            self.state = "buffering"
            self.plot_moment = True
        print(f"switched to state: {self.state}")

    def step(self):
        if self.state == "buffering":
            index_start, index_end = self.groups[self.buffer_group]
            self.switch_layer[index_start:index_end] = self.switch_layer[index_start:index_end] + (1 / self.buffer_time)
            print(self.switch_layer.mean())
        self.epoch += 1 
        if not (self.epoch % self.buffer_time):
            if self.epoch < (self.total_epochs // 2):
                self.switch_state()
            else:
                self.state = "rest"
                print(f"state: {self.state}")
        else:
            self.plot_moment = False
    
    def forward(self, coordinates, z, progress=None):
        gamma, beta = self.mapping_net(z)
        # output = self.net[0](coordinates, gamma=gamma, beta=beta, progress=progress)
        output = self.net[0](coordinates, gamma=gamma, beta=beta) * self.switch_layer
        for layer in self.net[1:]:
            if layer.film_conditioning:
                # output = layer(output, gamma=gamma, beta=beta, progress=progress)
                output = layer(output, gamma=gamma, beta=beta)
            else:
                output = layer(output)
        return output


class piGAN_custom(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                num_mapping_layers=3, num_INR_layers=8, num_hidden_mapping=256, num_hidden_INR=256, 
                z_size=256, first_omega_0=600, hidden_omega_0=30, 
                activations= ["sine", "sine", "none"], # "sine", "relu", "none"
                conditioning_method = "film", # "concat", "film", "both"
                conditioning_location = "all", # "all, middle"
                network_shape= "consistent", # "consistent", "shrinking"
                ):
        super().__init__()
        init_methods = []
        for i, activation in enumerate(activations):
            if i == 0:
                init_methods.append({
                    "sine": {"weights": 'siren_first', "bias": "polar"},
                    "relu": {"weights": 'kaiming_in', "bias": "zero"},
                    "none": {"weights": 'siren_omega', "omega":30, "bias": "none"}
                }[activation])
            else:
                init_methods.append({
                    "sine": {"weights": 'siren', "bias": "polar"},
                    "relu": {"weights": 'kaiming_in', "bias": "zero"},
                    "none": {"weights": 'siren_omega', "omega":30, "bias": "none"}
                }[activation])

        print(f"activations: {activations}")
        print(f"init_methods: {init_methods}")

        if conditioning_method  == "film":
            self.film_conditioning = True
            self.concat_conditioning_all = 0 
            self.concat_conditioning_middle = 0 
            self.concat_conditioning_first = 0 
        elif conditioning_method  == "concat":
            if conditioning_location == "middle":
                self.film_conditioning = False
                self.concat_conditioning_all = 0 
                self.concat_conditioning_middle = 1          
                self.concat_conditioning_first = 1          
            elif conditioning_location  == "all":
                self.film_conditioning = False
                self.concat_conditioning_all = 1 
                self.concat_conditioning_middle = 0          
                self.concat_conditioning_first = 1          

        self.network_shape = network_shape

        if self.film_conditioning:
            self.film_mapping_net = PiGANMappingNetwork(z_size,
                                                    num_hidden_mapping, 
                                                    num_hidden_INR*2,
                                                    depth = num_mapping_layers)
        if self.concat_conditioning_first:
            self.concat_mapping_net = ConcatMappingNetwork(z_size,
                                                    num_hidden_mapping, 
                                                    z_size,
                                                    depth = num_mapping_layers)

        self.net = []
        self.net.append(ImplicitMLPLayer(in_features, num_hidden_INR, bias=True,
                        omega_0=first_omega_0, w_norm=False, activation=activations[0], 
                        film_conditioning=self.film_conditioning, concat_conditioning=z_size*self.concat_conditioning_first,
                        init_method=init_methods[0]))
        
        last_hidden = num_hidden_INR
        for i in range(num_INR_layers-1):
            if self.network_shape == "shrinking":
                num_hidden_INR = num_hidden_INR // 2

            if self.concat_conditioning_middle and i == 3:
                self.net.append(ImplicitMLPLayer(last_hidden, num_hidden_INR, bias=True,
                                omega_0=hidden_omega_0, w_norm=False, activation=activations[1], 
                                film_conditioning=self.film_conditioning, concat_conditioning=z_size+1,
                                init_method=init_methods[1]))
            elif self.concat_conditioning_all:
                self.net.append(ImplicitMLPLayer(last_hidden, num_hidden_INR, bias=True,
                                omega_0=hidden_omega_0, w_norm=False, activation=activations[1], 
                                film_conditioning=self.film_conditioning, concat_conditioning=z_size+1,
                                init_method=init_methods[1]))
            else:
                self.net.append(ImplicitMLPLayer(last_hidden, num_hidden_INR, bias=True,
                                omega_0=hidden_omega_0, w_norm=False, activation=activations[1], 
                                film_conditioning=self.film_conditioning, concat_conditioning=0,
                                init_method=init_methods[1]))

            last_hidden = num_hidden_INR

        self.net.append(ImplicitMLPLayer(last_hidden, out_features, bias=True,
                omega_0=1, w_norm=False, activation=activations[2], 
                film_conditioning=False, concat_conditioning=0,
                init_method=init_methods[2]))
        self.net = nn.Sequential(*self.net)

        for i, layer in enumerate(self.net):
            if layer.film_conditioning:
                print(f"layer {i}: Film conditioned")
            if layer.concat_conditioning:
                print(f"layer {i}: concat conditioned")

    def forward(self, coordinates, z):
        gamma, beta, concat = None, None, None
        coordinates = coordinates.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        
        if self.film_conditioning:
            gamma, beta = self.film_mapping_net(z)
        elif self.concat_conditioning_first:
            concat = self.concat_mapping_net(z)
            concat = concat.repeat(1, coordinates.shape[1], 1)
            
        output = self.net[0](coordinates, gamma=gamma, beta=beta, z=concat)
        for layer in self.net[1:]:
            if layer.film_conditioning:
                output = layer(output, gamma=gamma, beta=beta)
            elif layer.concat_conditioning:
                output = layer(output, z=torch.cat((concat, coordinates), dim=-1))
            else:
                output = layer(output)
        return output

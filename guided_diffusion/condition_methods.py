from abc import ABC, abstractmethod
import torch
import numpy as np
from torch.fft import fftn, ifftn, fftshift, ifftshift
from torch.nn import functional as F
from pdb import set_trace


__CONDITIONING_METHOD__ = {}

def register_conditioning_method(name: str):
    def wrapper(cls):
        if __CONDITIONING_METHOD__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __CONDITIONING_METHOD__[name] = cls
        return cls
    return wrapper

def get_conditioning_method(name: str, operator, noiser, **kwargs):
    if __CONDITIONING_METHOD__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined!")
    return __CONDITIONING_METHOD__[name](operator=operator, noiser=noiser, **kwargs)

    
class ConditioningMethod(ABC):
    def __init__(self, operator, noiser, **kwargs):
        self.operator = operator
        self.noiser = noiser

    def project(self, data, noisy_measurement, **kwargs):
        return self.operator.project(data=data, measurement=noisy_measurement, **kwargs)
    
    def grad_and_value(self, x_prev, x_0_hat, measurement, t, **kwargs):
        if self.noiser.__name__ == 'gaussian':
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
        elif self.noiser.__name__ == 'poisson':
            Ax = self.operator.forward(x_0_hat, **kwargs)
            difference = measurement-Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        else:
            raise NotImplementedError
             
        return norm_grad, norm
   
    @abstractmethod
    def conditioning(self, x_t, measurement, noisy_measurement=None, **kwargs):
        pass
    
@register_conditioning_method(name='vanilla')
class Identity(ConditioningMethod):
    # just pass the input without conditioning
    def conditioning(self, x_t):
        return x_t
    
@register_conditioning_method(name='projection')
class Projection(ConditioningMethod):
    def conditioning(self, x_t, noisy_measurement, **kwargs):
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement)
        return x_t


@register_conditioning_method(name='mcg')
class ManifoldConstraintGradient(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
        
    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        # posterior sampling
        norm_grad, norm, _ = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, **kwargs)
        x_t -= norm_grad * self.scale
        
        # projection
        x_t = self.project(data=x_t, noisy_measurement=noisy_measurement, **kwargs)
        return x_t, norm


@register_conditioning_method(name='score_sde')
class ScoreSDE(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)
    
    def init_mags(self, measurement):
       pass

    def init_kernel_mag(self, kernel, img_size):
        pass

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, noisy_measurement, **kwargs):
        difference = noisy_measurement - self.operator.forward(x_prev)
        norm = torch.linalg.norm(difference)
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm
        
@register_conditioning_method(name='dps')
class DiffusionPosteriorSampling(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, t, **kwargs):
        norm_grad, norm = self.grad_and_value(x_prev=x_prev, x_0_hat=x_0_hat, measurement=measurement, t=t, **kwargs)
        x_t -= norm_grad * (self.scale)
        return x_t, norm

@register_conditioning_method(name='fgps')
class FGPS(ConditioningMethod):
    def __init__(self, operator, noiser, cond_model, schedule, start_bound, end_bound, T, **kwargs):
        super().__init__(operator, noiser)

        self.scale_schedule = kwargs.get('scale_schedule', None)
        if self.scale_schedule=="fixed":
            self.scale = kwargs.get('scale', 1.0)
        else:
            time_steps = np.linspace(0, T, T)
            annealing_values = 0.5 * (a + b) + 0.5 * (a - b) * np.cos(np.pi * time_steps / T)
            self.scale = torch.tensor(annealing_values).to("cuda")
            
        self.cond_model = cond_model
        if schedule == 'linear':
            self.cutoff_schedule = torch.linspace(start_bound, end_bound, 1000).flip(0)
        elif schedule == 'exponential':
            self.cutoff_schedule = (end_bound - (end_bound - start_bound) * torch.exp(-torch.linspace(0, 5, 1000))).flip(0)
        else:
            raise NotImplementedError("Only linear and exponential schedule is currently supported for FGPS.")

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, t, **kwargs):
        cutoff = self.cutoff_schedule[t.cpu()]
        if self.noiser.__name__ == 'gaussian':
            difference = self.cond_model(measurement, cutoff) \
                    - self.cond_model(self.operator.forward(x_0_hat, **kwargs), cutoff)
            norm = torch.linalg.norm(difference)
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        elif self.noiser.__name__ == 'poisson':
            Ax = self.cond_model(self.operator.forward(x_0_hat, **kwargs), cutoff)
            difference = self.cond_model(measurement, cutoff) - Ax
            norm = torch.linalg.norm(difference) / measurement.abs()
            norm = norm.mean()
            norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        else:
            raise NotImplementedError("Only Gaussian and Poisson noise implemented.")

        if self.scale_schedule=="fixed":
            x_t -= norm_grad * (self.scale)
        else:
            x_t -= norm_grad * (self.scale[t])
        return x_t, norm
        
@register_conditioning_method(name='ps+')
class PosteriorSamplingPlus(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.num_sampling = kwargs.get('num_sampling', 5)
        self.scale = kwargs.get('scale', 1.0)

    def conditioning(self, x_prev, x_t, x_0_hat, measurement, **kwargs):
        norm = 0
        for _ in range(self.num_sampling):
            # TODO: use noiser?
            x_0_hat_noise = x_0_hat + 0.05 * torch.rand_like(x_0_hat)
            difference = measurement - self.operator.forward(x_0_hat_noise)
            norm += torch.linalg.norm(difference) / self.num_sampling
        
        norm_grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
        x_t -= norm_grad * self.scale
        return x_t, norm


# Implement of Diffusion with Spherical Gaussian Constraint(DSG)
@register_conditioning_method(name='DSG')
class DSG(ConditioningMethod):
    def __init__(self, operator, noiser, **kwargs):
        super().__init__(operator, noiser)
        self.interval = kwargs.get('interval', 1.)
        self.guidance_scale = kwargs.get('guidance_scale', 1.)
        print(f'interval: {self.interval}')
        print(f'guidance_scale: {self.guidance_scale}')

    def conditioning(self, x_prev, x_t, x_t_mean, x_0_hat, measurement, idx, **kwargs):
        eps = 1e-8
        if idx % self.interval == 0:
            difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            norm = torch.linalg.norm(difference)
            grad = torch.autograd.grad(outputs=norm, inputs=x_prev)[0]
            grad_norm = torch.linalg.norm(grad, dim=[1, 2, 3])

            b, c, h, w = x_t.shape
            r = torch.sqrt(torch.tensor(c * h * w)) * kwargs.get('sigma_t', 1.)[0, 0, 0, 0]
            guidance_rate = self.guidance_scale

            d_star = -r * grad / (grad_norm + eps)
            d_sample = x_t - x_t_mean
            mix_direction = d_sample + guidance_rate * (d_star - d_sample)
            mix_direction_norm = torch.linalg.norm(mix_direction, dim=[1, 2, 3])
            mix_step = mix_direction / (mix_direction_norm + eps) * r

            return x_t_mean + mix_step, norm

        else:
            # use the code below for print loss in unconditional step and commit it for saving time
            # difference = measurement - self.operator.forward(x_0_hat, **kwargs)
            # norm = torch.linalg.norm(difference)
            # return x_t, norm

            return x_t, torch.zeros(1)

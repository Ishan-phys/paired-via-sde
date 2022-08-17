import torch
from sampling import get_sampler
from models.utils import get_score_fn
from configs.config import CFGS

def get_sde_loss_fn(sde, reduce_mean=True, continuous=True, eps=1e-5):
    
    """Create a loss function for training with arbirary SDEs.
    
    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
        ad-hoc interpolation to take continuous time steps.
        eps: A `float` number. The smallest time step to sample from.
        
    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)
    device = CFGS["device"]
    
    def loss_sc(t,z):  
        """Define a loss function for a particular time and noise level z

        Args:
            t (_type_): _description_
            z (_type_): _description_
        """
        def loss_score(model, img, cond):
            """Evaluates the loss of a score function.

            Args:
                model: a score model
                img: a mini-batch of images
                cond: the conditioning

            Returns:
                evaluated loss
            """
            score_fn = get_score_fn(sde, model, continuous=continuous)
            mean_img, std_img = sde.marginal_prob(img, t)
            perturbed_img = mean_img + std_img[:, None, None, None] * z
            score = score_fn(torch.cat((cond, perturbed_img), dim=1), t)
            losses = torch.square(score * std_img[:, None, None, None] + z)
            losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
            
            return torch.mean(losses)
        
        return loss_score

    def loss_fn(model_xy, model_yx, batch):
        """Compute the loss function.
        
        Args:
            model: A score model.
            batch: A mini-batch of training data.
        
        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """        
        # Load the x and y paired batches of dataset.
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        
        # Sample the noise and set the time variable
        t = torch.rand(x.shape[0], device=device) * (sde.T - eps) + eps
        z = torch.randn_like(x)
        
        # Define the loss of a score function
        loss_score = loss_sc(t,z)
        
        # Calculate the losses. 
        loss_1 = loss_score(model_xy, x, y)
        loss_2 = loss_score(model_yx, y, x)
        
        loss = loss_1 + loss_2
        
        return loss

    return loss_fn
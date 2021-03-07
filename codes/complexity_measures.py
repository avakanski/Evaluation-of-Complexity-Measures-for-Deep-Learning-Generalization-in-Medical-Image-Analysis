# The implementation of complexity measures is from https://github.com/nitarshan/robust-generalization-measures
# It is based on the paper by Dziugaite et el. (2020) "In Search of Robust Measures of Generalization"
# https://arxiv.org/abs/2010.11924

from typing import Dict, List, NamedTuple, Optional, Tuple
from enum import Enum, IntEnum
from copy import deepcopy
from torch import Tensor
from contextlib import contextmanager
import math

import numpy as np
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader

class ComplexityType(Enum):
  # Measures from Fantastic Generalization Measures
  PARAMS = 20
  INVERSE_MARGIN = 22
  LOG_SPEC_INIT_MAIN = 29
  LOG_SPEC_ORIG_MAIN = 30
  LOG_PROD_OF_SPEC_OVER_MARGIN = 31
  LOG_PROD_OF_SPEC = 32
  FRO_OVER_SPEC = 33
  LOG_SUM_OF_SPEC_OVER_MARGIN = 34
  LOG_SUM_OF_SPEC = 35
  LOG_PROD_OF_FRO_OVER_MARGIN = 36
  LOG_PROD_OF_FRO = 37
  LOG_SUM_OF_FRO_OVER_MARGIN = 38
  LOG_SUM_OF_FRO = 39
  FRO_DIST = 40
  DIST_SPEC_INIT = 41
  PARAM_NORM = 42
  PATH_NORM_OVER_MARGIN = 43
  PATH_NORM = 44
  PACBAYES_INIT = 48
  PACBAYES_ORIG = 49
  PACBAYES_FLATNESS = 53
  PACBAYES_MAG_INIT = 56
  PACBAYES_MAG_ORIG = 57
  PACBAYES_MAG_FLATNESS = 61
  # Other Measures
  L2 = 100
  L2_DIST = 101

  @classmethod
  def data_dependent_measures(cls):
    return {
      cls.PACBAYES_ORIG,
      cls.PACBAYES_INIT,
      cls.PACBAYES_MAG_ORIG,
      cls.PACBAYES_MAG_INIT,
      cls.PACBAYES_FLATNESS,
      cls.PACBAYES_MAG_FLATNESS,
      cls.INVERSE_MARGIN,
      cls.LOG_PROD_OF_FRO_OVER_MARGIN,
      cls.LOG_SUM_OF_FRO_OVER_MARGIN,
      cls.LOG_PROD_OF_SPEC_OVER_MARGIN,
      cls.LOG_SUM_OF_SPEC_OVER_MARGIN,
      cls.PATH_NORM_OVER_MARGIN,
    }

  @classmethod
  def acc_dependent_measures(cls):
    return {
      cls.PACBAYES_ORIG,
      cls.PACBAYES_INIT,
      cls.PACBAYES_MAG_ORIG,
      cls.PACBAYES_MAG_INIT,
      cls.PACBAYES_FLATNESS,
      cls.PACBAYES_MAG_FLATNESS,
    }

class EvaluationMetrics(NamedTuple):
  acc: float
  avg_loss: float
  num_correct: int
  num_to_evaluate_on: int
  all_complexities: Dict[ComplexityType, float]

# instance
CT = ComplexityType

# https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py
@torch.no_grad()
def _reparam(model):
  def in_place_reparam(model, prev_layer=None):
    for child in model.children():
      prev_layer = in_place_reparam(child, prev_layer)
      if child._get_name() == 'Conv2d':
        prev_layer = child
      elif child._get_name() == 'BatchNorm2d':
        scale = child.weight / ((child.running_var + child.eps).sqrt())
        prev_layer.bias.copy_( child.bias  + ( scale * (prev_layer.bias - child.running_mean) ) )
        perm = list(reversed(range(prev_layer.weight.dim())))
        prev_layer.weight.copy_((prev_layer.weight.permute(perm) * scale ).permute(perm))
        child.bias.fill_(0)
        child.weight.fill_(1)
        child.running_mean.fill_(0)
        child.running_var.fill_(1)
    return prev_layer
  model = deepcopy(model)
  in_place_reparam(model)
  return model

@contextmanager
def _perturbed_model(
  model,
  sigma: float,
  rng,
  magnitude_eps: Optional[float] = None
):
  device = next(model.parameters()).device
  if magnitude_eps is not None:
    noise = [torch.normal(0,sigma**2 * torch.abs(p) ** 2 + magnitude_eps ** 2, generator=rng) for p in model.parameters()]
  else:
    noise = [torch.normal(0,sigma**2,p.shape, generator=rng).to(device) for p in model.parameters()]
  model = deepcopy(model)
  try:
    [p.add_(n) for p,n in zip(model.parameters(), noise)]
    yield model
  finally:
    [p.sub_(n) for p,n in zip(model.parameters(), noise)]
    del model

# https://drive.google.com/file/d/1_6oUG94d0C3x7x2Vd935a2QqY-OaAWAM/view
@torch.no_grad()
def _pacbayes_sigma(
  model,
  dataloader,
  accuracy: float,
  seed: int,
  magnitude_eps: Optional[float] = None,
  search_depth: int = 15,
  montecarlo_samples: int = 10,
  accuracy_displacement: float = 0.1,
  displacement_tolerance: float = 1e-2,
) -> float:
  lower, upper = 0, 2
  sigma = 1

  BIG_NUMBER = 10348628753
  device = next(model.parameters()).device
  rng = torch.Generator(device=device) if magnitude_eps is not None else torch.Generator()
  rng.manual_seed(BIG_NUMBER + seed)

  for _ in range(search_depth):
    sigma = (lower + upper) / 2
    accuracy_samples = []
    for _ in range(montecarlo_samples):
      with _perturbed_model(model, sigma, rng, magnitude_eps) as p_model:
        loss_estimate = 0
        for data, target in dataloader:
          # data = data.to(device, dtype=torch.float)
          # modified line to the line below to move target to device
          data, target = data.to(device, dtype=torch.float), target.to(device)
          logits = p_model(data)
          pred = logits.data.max(1, keepdim=True)[1]  # get the index of the max logits
          batch_correct = pred.eq(target.data.view_as(pred)).type(torch.FloatTensor).cpu()
          loss_estimate += batch_correct.sum()
        loss_estimate /= len(dataloader.dataset)
        accuracy_samples.append(loss_estimate)
    displacement = abs(np.mean(accuracy_samples) - accuracy)
    if abs(displacement - accuracy_displacement) < displacement_tolerance:
      break
    elif displacement > accuracy_displacement:
      # Too much perturbation
      upper = sigma
    else:
      # Not perturbed enough to reach target displacement
      lower = sigma
  return sigma

@torch.no_grad()
def get_all_measures(model, init_model, dataloader, 
  acc: float,
  seed: int,
  ) -> Dict[CT, float]:
  measures = {}

  model = _reparam(model)
  init_model = _reparam(init_model)

  device = next(model.parameters()).device
  m = len(dataloader.dataset)

  def get_weights_only(model) -> List[Tensor]:
    blacklist = {'bias', 'bn'}
    return [p for name, p in model.named_parameters() if all(x not in name for x in blacklist)]
  
  weights = get_weights_only(model)
  dist_init_weights = [p-q for p,q in zip(weights, get_weights_only(init_model))]
  d = len(weights)

  def get_vec_params(weights: List[Tensor]) -> Tensor:
    return torch.cat([p.view(-1) for p in weights], dim=0)

  w_vec = get_vec_params(weights)
  dist_w_vec = get_vec_params(dist_init_weights)
  num_params = len(w_vec)

  def get_reshaped_weights(weights: List[Tensor]) -> List[Tensor]:
    # If the weight is a tensor (e.g. a 4D Conv2d weight), it will be reshaped to a 2D matrix
    return [p.view(p.shape[0],-1) for p in weights]
  
  reshaped_weights = get_reshaped_weights(weights)
  dist_reshaped_weights = get_reshaped_weights(dist_init_weights)

  # print("Calculate complexity measures\n")
  
  # Vector Norm Measures
  measures[CT.L2] = w_vec.norm(p=2)
  measures[CT.L2_DIST] = dist_w_vec.norm(p=2)
  
  # VC-Dimension Based Measures
  measures[CT.PARAMS] = torch.tensor(num_params) # 20

  # Measures on the output of the network
  def _margin(
    model,
    dataloader: DataLoader
  ) -> Tensor:
    margins = []
    for data, target in dataloader:
      data = data.to(device, dtype=torch.float)
      logits = model(data)
      correct_logit = logits[torch.arange(logits.shape[0]), target].clone()
      logits[torch.arange(logits.shape[0]), target] = float('-inf')
      max_other_logit = logits.data.max(1).values  # get the index of the max logits
      margin = correct_logit - max_other_logit
      margins.append(margin)
    return torch.cat(margins).kthvalue(m // 10)[0]

  margin = _margin(model, dataloader).abs()
  measures[CT.INVERSE_MARGIN] = torch.tensor(1, device=device) / margin ** 2 # 22

  # Norm & Margin-Based Measures
  fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in reshaped_weights])
  spec_norms = torch.cat([p.svd().S.max().unsqueeze(0) ** 2 for p in reshaped_weights])
  dist_fro_norms = torch.cat([p.norm('fro').unsqueeze(0) ** 2 for p in dist_reshaped_weights])
  dist_spec_norms = torch.cat([p.svd().S.max().unsqueeze(0) ** 2 for p in dist_reshaped_weights])

  # Approximate Spectral Norm
  # Note that these use an approximation from [Yoshida and Miyato, 2017]
  # https://arxiv.org/abs/1705.10941 (Section 3.2, Convolutions)
  measures[CT.LOG_PROD_OF_SPEC] = spec_norms.log().sum() # 32
  measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN] = measures[CT.LOG_PROD_OF_SPEC] - 2 * margin.log() # 31
  measures[CT.LOG_SPEC_INIT_MAIN] = measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN] + (dist_fro_norms / spec_norms).sum().log() # 29
  measures[CT.FRO_OVER_SPEC] = (fro_norms / spec_norms).sum() # 33
  measures[CT.LOG_SPEC_ORIG_MAIN] = measures[CT.LOG_PROD_OF_SPEC_OVER_MARGIN] + measures[CT.FRO_OVER_SPEC].log() # 30
  measures[CT.LOG_SUM_OF_SPEC_OVER_MARGIN] = math.log(d) + (1/d) * (measures[CT.LOG_PROD_OF_SPEC] -  2 * margin.log()) # 34
  measures[CT.LOG_SUM_OF_SPEC] = math.log(d) + (1/d) * measures[CT.LOG_PROD_OF_SPEC] # 35

  # Frobenius Norm
  measures[CT.LOG_PROD_OF_FRO] = fro_norms.log().sum() # 37
  measures[CT.LOG_PROD_OF_FRO_OVER_MARGIN] = measures[CT.LOG_PROD_OF_FRO] -  2 * margin.log() # 36
  measures[CT.LOG_SUM_OF_FRO_OVER_MARGIN] = math.log(d) + (1/d) * (measures[CT.LOG_PROD_OF_FRO] -  2 * margin.log()) # 38
  measures[CT.LOG_SUM_OF_FRO] = math.log(d) + (1/d) * measures[CT.LOG_PROD_OF_FRO] # 39

  # Distance to Initialization
  measures[CT.FRO_DIST] = dist_fro_norms.sum() # 40
  measures[CT.DIST_SPEC_INIT] = dist_spec_norms.sum() # 41
  measures[CT.PARAM_NORM] = fro_norms.sum() # 42

  # Path-norm
  # https://github.com/bneyshabur/generalization-bounds/blob/master/measures.py#L98
  dataset_type_D = iter(dataloader).next()[0][0].shape
  def _path_norm(model) -> Tensor:
    model = deepcopy(model)
    model.eval()
    for param in model.parameters():
      if param.requires_grad:
        param.data.pow_(2)
    x = torch.ones([1] + list(dataset_type_D), device=device) # modified line, replaced model.dataset_type.D with dataset_type_D
    x = model(x)
    del model
    return x.sum()
  
  measures[CT.PATH_NORM] = _path_norm(model) # 44
  measures[CT.PATH_NORM_OVER_MARGIN] = measures[CT.PATH_NORM] / margin ** 2 # 43

  # Exact Spectral Norm
  # https://arxiv.org/abs/1805.10408
  # https://github.com/brain-research/conv-sv/blob/master/conv2d_singular_values.py#L52
  def _spectral_norm_fft(kernel: Tensor, input_shape: Tuple[int, int]) -> Tensor:
    # PyTorch conv2d filters use Shape(out,in,kh,kw)
    # [Sedghi 2018] code expects filters of Shape(kh,kw,in,out)
    # Pytorch doesn't support complex FFT and SVD, so we do this in numpy
    np_kernel = np.einsum('oihw->hwio', kernel.data.cpu().numpy())
    transforms = np.fft.fft2(np_kernel, input_shape, axes=[0, 1]) # Shape(ih,iw,in,out)
    singular_values = np.linalg.svd(transforms, compute_uv=False) # Shape(ih,iw,min(in,out))
    spec_norm = singular_values.max()
    return torch.tensor(spec_norm, device=kernel.device)

  # Flatness-based measures
  sigma = _pacbayes_sigma(model, dataloader, acc, seed)
  def _pacbayes_bound(reference_vec: Tensor) -> Tensor:
    return (reference_vec.norm(p=2) ** 2) / (4 * sigma ** 2) + math.log(m / sigma) + 10
  measures[CT.PACBAYES_INIT] = _pacbayes_bound(dist_w_vec) # 48
  measures[CT.PACBAYES_ORIG] = _pacbayes_bound(w_vec) # 49
  measures[CT.PACBAYES_FLATNESS] = torch.tensor(1 / sigma ** 2) # 53
  
  # Magnitude-aware Perturbation Bounds
  mag_eps = 0.001
  mag_sigma = _pacbayes_sigma(model, dataloader, acc, seed, mag_eps)
  omega = num_params
  def _pacbayes_mag_bound(reference_vec: Tensor) -> Tensor:
    numerator = mag_eps ** 2 + (mag_sigma ** 2 + 1) * (reference_vec.norm(p=2)**2) / omega
    denominator = mag_eps ** 2 + mag_sigma ** 2 * dist_w_vec ** 2
    return 1/4 * (numerator / denominator).log().sum() + math.log(m / mag_sigma) + 10
  measures[CT.PACBAYES_MAG_INIT] = _pacbayes_mag_bound(dist_w_vec) # 56
  measures[CT.PACBAYES_MAG_ORIG] = _pacbayes_mag_bound(w_vec) # 57
  measures[CT.PACBAYES_MAG_FLATNESS] = torch.tensor(1 / mag_sigma ** 2) # 61
  
  # Adjust for dataset size
  def adjust_measure(measure: CT, value: float) -> float:
    if measure.name.startswith('LOG_'):
      return 0.5 * (value - np.log(m))
    else:
      return np.sqrt(value / m)
      
  return {k: adjust_measure(k, v.item()) for k, v in measures.items()}   



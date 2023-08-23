# General Imports
import numpy as np
import random

# ML Imports
import torch
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline

class Jitter (object):
    
    def __init__(self, u=0, s=0.03):
        self.u = u
        self.s = s
        
    def __call__(self, sample):
        x = sample

        shp = x.shape
        jit = np.random.normal(self.u, self.s, shp)
        r = x+jit
        return x
    
    
class MagnitudeWarp (object):
    
    def __init__(self, u=1, s=0.2, k=4):
        self.u = u
        self.s = s
        self.k = k
        
    def __call__(self, sample):
        x = sample
        
        # Timepoints 
        t = len(x[0])
        
        # Generate knot time points
        ts = [0]
        xs = np.random.randint(1,t-1, self.k)
        xs.sort()
        for r in xs:
            ts.append(r)
        ts.append(t)

        # Generate Knot values
        L, _ = x.shape
        knot_vals = np.random.normal(self.u,self.s**2,(L,self.k+2))

        # Interpolate spline
        cs = CubicSpline(ts, knot_vals, axis=1)

        # Create range of spline values to return
        ns = np.arange(t)

        r = cs(ns)*x  
        return r
    
class Scaling (object):
    
    def __init__(self, u=1, s=0.2):
        self.u = u
        self.s = s
        
    def __call__(self, sample):
        x = sample
        v = np.random.normal(1, self.s)
        r = x*v
        return r

class WindowSlicing (object):
    
    def __init__(self, reduce_ratio=0.9):
        self.reduce_ratio = reduce_ratio
        
    def window_slice(self, x, reduce_ratio=0.9):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        target_len = np.ceil(reduce_ratio*x.shape[1]).astype(int)
        if target_len >= x.shape[1]:
            return x
        starts = np.random.randint(low=0, high=x.shape[1]-target_len, size=(x.shape[0])).astype(int)
        ends = (target_len + starts).astype(int)

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            ret[i,:] = np.interp(np.linspace(0, target_len, num=x.shape[1]), np.arange(target_len), pat[starts[i]:ends[i]]).T
        return ret
        
    def __call__(self, sample):
        x = sample
        
        r = self.window_slice(x, self.reduce_ratio)
        
        return r

class TimeWarp (object):
    
    def __init__(self, window_ratio=0.1, scales=[0.8, 1.2]):
        self.window_ratio = window_ratio
        self.scales = scales
        
    def window_warp(self, x, window_ratio=0.1, scales=[0.8, 1.2]):
        # https://halshs.archives-ouvertes.fr/halshs-01357973/document
        warp_scales = np.random.choice(scales, x.shape[0])
        warp_size = np.ceil(window_ratio*x.shape[1]).astype(int)
        window_steps = np.arange(warp_size)

        window_starts = np.random.randint(low=1, high=x.shape[1]-warp_size-1, size=(x.shape[0])).astype(int)
        window_ends = (window_starts + warp_size).astype(int)

        ret = np.zeros_like(x)
        for i, pat in enumerate(x):
            start_seg = pat[:window_starts[i]]
            window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i]])
            end_seg = pat[window_ends[i]:]
            warped = np.concatenate((start_seg, window_seg, end_seg))                
            ret[i,:] = np.interp(np.arange(x.shape[1]), np.linspace(0, x.shape[1]-1., num=warped.size), warped).T
        return ret
        
    def __call__(self, sample):
        x = sample
        
        r = self.window_warp(x, self.window_ratio, self.scales)

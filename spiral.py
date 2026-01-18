import numpy as np
from PIL import Image
import scipy as sp

def encode(array_img, anchor_xy=None, eps=0.2, omega=40):
    '''
    Encodes an image with a set of parameters which produces a 
    curve giving the impression of the same image by eye.
    
    The eventual functional form we're aiming for is
    
        X(s) = x(s) + dx(s)*eps*a(s)*sin(w*s).
        Y(s) = y(s) + dy(s)*eps*a(s)*sin(w*s),
        
    where parameter s smoothly interpolates with enough resolution 
    to draw the curve on top of (x(t), y(t)) (anchor_xy).
    
    This function constructs a(t) based on sampling adjacent image intensities 
    near point (x(t), y(t)).
    
    Inputs: 
        array_img: numpy array of uint8 (integers 0 through 255); 
            greyscale image where 0=black
        anchor_xy: Either None, or list-like of shape (2,n) 
            of a trajectory (x(t), y(t)). If None, reasonable 
            choices are made for a trajectory.

    Outputs:
        eps  : float, max amplitude of perturbations
        w    : float, oscillation frequency.
        s    : numpy array of floats, monotone increasing from 0 to 1; 
            expect on the order of length 2*w*n to smoothly 
            sample oscillations.
        a(s) : array of floats, functional parameter on [0,1] 
            associated with the curve. Locally the form is
            
                x(s) + dx*eps*a(s)*sin(w*s).
                y(s) + dy*eps*a(s)*sin(w*s),
                
            where (dx,dy) is a local rate of change for the curve.
    '''
    #####################
    #
    # input checking and setting defaults
    _shape = np.shape(array_img)
    assert len(_shape) == 2 # want grayscale images.
    m,n = _shape
    
    # Default curve is spiral r=k*theta with center at center of image.
    # Choose k and theta such that 10 revolutions happen 
    # to hit the closest image boundary.  
    if anchor_xy is None:
        _y0, _x0 = m/2, n/2
        _t = np.linspace(0, 10*2*np.pi, 10*60)
        _k = min(_x0,_y0)/max(_t)
        if n>m:
            _t = _t - np.pi/2 # begin/halt at bottom of circle instead.
        _r = _k*_t
        anchor_xy = [_x0 + _r*np.cos(_t), _y0 + _r*np.sin(_t)]
    #
    ######################
    #
    x,y = anchor_xy
    t = np.linspace(0,1, len(x))
    ref = max(2*int(omega),1)
    s = np.linspace(0,1, ref*len(x))
    
    a_t = np.zeros(len(t))
    #a_s = np.zeros(len(s))
    for k in range(len(t)):
        # TODO: average neighboring pixels rather than rounding down.
        i,j = int(anchor_xy[0][k]), int(anchor_xy[1][k])
        
        # in-bounds?
        if i>m-1 or j>m-1:
            a_t[k]=0
        else:
            a_t[k] = array_img[i,j]/255
        a_s[ref*k] = a_t[k]
    #
        
    # linearly interpolate on the finer grid.
    #for k in range(len(t)-1):
    #    _th = np.linspace(0,1,ref)
    #    a_s[ref*k:ref*(k+1)] = (1-_th)*a_s[ref*k] + _th*a_s[ref*(k+1)]
    return a_t
#

def resolve(a_t, eps=0.2, omega=40):
    '''
    Given a_t, eps, and omega, computes
        X(s) = x(s) + eps*
    
        eps: float, max amplitude for oscillations. 
        Algorithm works in pixel coordinates; so for tight curves 
        it's recommended 0 < eps < 1 (adjacent pixels are 1 unit apart)
        DEFAULT: 
    omega : float, oscillation frequency for perturbations.
        DEFAULT: 
    '''
    n = len(a_t)
    skip = int(2*omega)
    N = skip*n
    
    a_s = np.zeros(N, dtype=float)
    
    #for k in range(n):
    #    a_s[k*skip:(k+1)*skip] = a_
    pass
    
if __name__=="__main__":
    from matplotlib import pyplot as plt
    
    # TODO: Not clear if this conversion to greyscale works for native CMYK images.
    im = Image.open('Kermit_Drinking_Tea.jpg').convert('L')
    arr = np.array(im)
    
    a_t = encode(arr)
    

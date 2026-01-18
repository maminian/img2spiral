import numpy as np
from PIL import Image

def encode(array_img, anchor_xy=None):
    '''
    Encodes an image with a set of parameters which produces a 
    curve giving the impression of the same image by eye.
    
    The eventual functional form we're aiming for is roughly
    
        X(s) = x(s) + eps*u*sin(omega*s)
        Y(s) = y(s) + eps*v*sin(omega*s)
        
    where parameter s smoothly interpolates with enough resolution 
    to draw the curve on top of (x(t), y(t)) (xy).
    
    This function constructs a(t) based on sampling adjacent image intensities 
    near point (x(t), y(t)).
    
    Inputs: 
        array_img: numpy array of uint8 (integers 0 through 255); 
            greyscale image where 0=black
        anchor_xy: Either None, or list-like of shape (2,n) 
            of a trajectory (x(t), y(t)). If None, reasonable 
            choices are made for a trajectory.

    Outputs:
        xy : passes the input as output; otherwise passes internally
            created default.
        a_t : array of floats, functional parameter on [0,1] 
            associated with the local amplitude of oscillations.
    
    '''
    #####################
    #
    # input checking and setting defaults
    _shape = np.shape(array_img)
    assert len(_shape) == 2 # want grayscale images.
    m,n = _shape
    
    # Default curve is spiral r=k*theta with center at center of image.
    # Choose k and theta such that 30 revolutions happen 
    # to hit the closest image boundary.  
    if anchor_xy is None:
        _y0, _x0 = m/2, n/2
        _t = np.linspace(0, 40*2*np.pi, 40*60)
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
    
    a_t = np.zeros(len(t))
    for k in range(len(t)):
        # TODO: average neighboring pixels rather than clipping/rounding down.
        i,j = int(anchor_xy[0][k]), int(anchor_xy[1][k])
        
        # in-bounds?
        if i>m-1 or j>m-1:
            a_t[k]=0
        else:
            a_t[k] = array_img[i,j]/255
    #
    
    # invert about 0.5 since we want lighter pixels
    # (larger array values) to have lesser oscillation amplitude.
    a_t = 1-a_t
    
    return anchor_xy,a_t
#

def resolve(anchor_xy, a_t, eps=4, omega=8):
    '''
    Given a_t, eps, and omega, computes and outputs roughly
    
        X(s) = x(s) + eps*a(s)*u*sin(omega*s)
        Y(s) = y(s) + eps*a(s)*v*sin(omega*s)
    
    Inputs:
        eps: float, max amplitude for oscillations. 
        Algorithm works in pixel coordinates; so for tight curves 
        it's recommended 0 < eps < 1 (adjacent pixels are 1 unit apart)
        DEFAULT: 
    omega : float, oscillation frequency for perturbations.
        DEFAULT: 
    
    Outputs:
        X_s, Y_s : numpy arrays of floats to substitute into ax.plot (for example).
    '''
    n = len(a_t)
    skip = int(8*omega) # TODO: less ambiguous resolution parameter; non hard-coded
    N = skip*n
    
    a_s = np.zeros(N, dtype=float)
    
    x,y = anchor_xy
    t = np.linspace(0,1,n)
    
    s = np.linspace(0, n*(2*np.pi), N)
    
    # linearly interpolate between consecutive (x(t),y(t))
    x_s,y_s = np.zeros(N, dtype=float),np.zeros(N, dtype=float)
    for k in range(n):
        x_s[k*skip] = x[k]
        y_s[k*skip] = y[k]
        a_s[k*skip] = a_t[k]
    #
    
    _th = np.linspace(0,1,skip)
    for k in range(n-1):
        x_s[k*skip:(k+1)*skip] = (1-_th)*x_s[k*skip] + _th*x_s[(k+1)*skip]
        y_s[k*skip:(k+1)*skip] = (1-_th)*y_s[k*skip] + _th*y_s[(k+1)*skip]
        a_s[k*skip:(k+1)*skip] = (1-_th)*a_s[k*skip] + _th*a_s[(k+1)*skip]
    #
    maxk = ((N-1)*skip)+1
    ##
    
    # Identify parallel and perp directions relative to the curve; 
    # oscillations should be perp to the local velocity.
    # TODO: sort out issues with 0 speed, truncating arrays, etc.
    x_s = x_s[:maxk]
    y_s = y_s[:maxk]
    a_s = a_s[:maxk]
    
    dx = np.diff(x_s)
    dy = np.diff(y_s)
    dx = np.concat([dx, [dx[-1]]])
    dy = np.concat([dy, [dy[-1]]])
    
    speed = np.sqrt(dx**2 + dy**2)
    
    u = -dy/speed
    v = dx/speed
    
    # compute
    X_s = x_s + (eps*a_s)*u*np.sin(omega*s[:maxk])
    Y_s = y_s + (eps*a_s)*v*np.sin(omega*s[:maxk])
    
    return X_s,Y_s
#


if __name__=="__main__":
    from matplotlib import pyplot as plt
    
    # TODO: don't know if this conversion to greyscale works for native CMYK images.
    im = Image.open('Kermit_Drinking_Tea.jpg').convert('L')
    arr = np.array(im)
    
    xy,a_t = encode(arr)
    
    X_s,Y_s = resolve(xy, a_t)
    
    # visualize
    fig,ax = plt.subplots(1,2, figsize=(10,5), constrained_layout=True)
    
    ax[0].matshow(arr, cmap=plt.cm.Greys_r)
    ax[1].plot(Y_s, -X_s, lw=0.5)
    ax[1].set(aspect='equal', xticks=[], yticks=[])
    ax[0].set(xticks=[], yticks=[])
    
    fig.show()
    #fig.savefig('kermie.jpg', bbox_inches='tight')
    

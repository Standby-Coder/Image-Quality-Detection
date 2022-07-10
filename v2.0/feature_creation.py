import feature_utils.laplacian_variance as laplacian_variance
import feature_utils.brenners_measure as brenners_measure
import feature_utils.gradient_energy as gradient_energy
import feature_utils.hist_entropy as hist_entropy
import feature_utils.wav_coeff as wav_coeff
import feature_utils.spatial_freq as spatial_freq
import feature_utils.sqrgrad as sqrgrad

def get_features(train):
    """Variance of Laplacian"""
    train = laplacian_variance.start(train)
    
    """Brenner's Focal Measure"""
    train = brenners_measure.start(train)
    
    """Gradient Energy"""
    train = gradient_energy.start(train)
    
    """Squared Gradient"""
    train = sqrgrad.start(train)
    
    """Histogram Entropy"""
    train = hist_entropy.start(train)
    
    """Spatial Frequency"""
    train = spatial_freq.start(train)
    
    """Sum of Wavelet Coefficients"""
    train = wav_coeff.start(train,"sum")
   
    """Variance of Wavelet Coefficients"""
    train = wav_coeff.start(train,"var")
    
    return train
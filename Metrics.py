import numpy as np
from scipy.signal import convolve2d


def _create_gaussian_window(size, sigma):
    """Create a Gaussian window (equivalent to MATLAB's fspecial('gaussian'))"""
    ax = np.arange(-size // 2 + 1., size // 2 + 1.)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2. * sigma ** 2))
    kernel /= np.sum(kernel)
    return kernel


class Scores:
    def __init__(self, K=(0.01, 0.03), window_size=11, sigma=1.5, L=255):
        self.K1, self.K2 = K
        self.L = L
        self.window = _create_gaussian_window(window_size, sigma)

    def _ssim_index(self, img1, img2):
        """SSIM calculation matching MATLAB's ssim_index logic"""
        if img1.shape != img2.shape:
            raise ValueError("Input images must have the same dimensions")
        if img1.ndim != 2:
            raise ValueError("Only 2D grayscale images are supported")

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        window = self.window
        C1 = (self.K1 * self.L) ** 2
        C2 = (self.K2 * self.L) ** 2

        mu1 = convolve2d(img1, window, mode='valid')
        mu2 = convolve2d(img2, window, mode='valid')
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma12 = convolve2d(img1 * img2, window, mode='valid') - mu1_mu2
        sigma1_sq = convolve2d(img1 * img1, window, mode='valid') - mu1_sq
        sigma2_sq = convolve2d(img2 * img2, window, mode='valid') - mu2_sq

        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2

        ssim_map = (numerator1 * numerator2) / (denominator1 * denominator2)
        return np.mean(ssim_map)

    def compute_ssim(self, X, Y):
        """Compute the mean SSIM between 3D tensors X and Y"""
        if X.shape != Y.shape:
            raise ValueError("The compared tensors must have the same shape")

        I, J = X.shape[0], X.shape[1]
        K = X.size // (I * J)
        X = X.reshape((I, J, K))
        Y = Y.reshape((I, J, K))  # Assume correct shape provided externally

        # Convert to log domain (dB)
        X = 10 * np.log10(np.clip(X, 1e-12, None))
        Y = 10 * np.log10(np.clip(Y, 1e-12, None))

        # Normalize to [0, 255]
        min_val = np.min(Y)
        X += min_val
        Y += min_val
        max_val = np.max(Y)
        X = X / max_val * 255
        Y = Y / max_val * 255

        # Compute SSIM for each slice
        ssim_scores = []
        for k in range(K):
            score = self._ssim_index(X[:, :, k], Y[:, :, k])
            ssim_scores.append(score)

        return np.mean(ssim_scores)

    def compute_nmse(self, X, Y):
        """Compute the Normalized Mean Squared Error between tensors X and Y"""
        if X.shape != Y.shape:
            raise ValueError("The compared tensors must have the same shape")

        numerator = np.linalg.norm(X - Y) ** 2
        denominator = np.linalg.norm(X) ** 2
        return numerator / denominator

import numpy as np
import matplotlib.pyplot as plt


def gen_three_blobs():
    """Generate three blobs with three gaussian distributions in 2D."""
    mu1 = [0, 0]
    sigma1 = np.eye(2)
    
    mu2 = [5, 5]
    sigma2 = np.eye(2)
    
    mu3 = [5, 10]
    sigma3 = np.eye(2)

    data1 = np.random.multivariate_normal(mu1, sigma1, 1000)
    data2 = np.random.multivariate_normal(mu2, sigma2, 1000)
    data3 = np.random.multivariate_normal(mu3, sigma3, 1000)

    return data1, data2, data3

def gen_circles():
    """Generate data according to two concentric circles."""
    
    def noisy_circle(r, theta):
        return np.array([r*np.cos(theta), r*np.sin(theta)]) + \
               np.random.multivariate_normal([0,0], .001*np.eye(2))
    
    thetas = np.linspace(-np.pi, np.pi, num=800)
    data1 = np.array([noisy_circle(1, theta) for theta in thetas])
    data2 = np.array([noisy_circle(.5, theta) for theta in thetas])
    return data1, data2

def gen_snakes():
    """Generate 2D 'snake' kindda data."""
    def noise():
        return np.random.multivariate_normal([0,0], 0.003*np.eye(2))
    thetas = np.linspace(0, np.pi, 800)
    data1 = np.array([np.array([theta, 2*np.sin(theta)])+noise() 
                        for theta in thetas])
    thetas = np.linspace(np.pi, 2*np.pi, 800)
    data2 = np.array([np.array([theta, 2*np.sin(theta)])+noise() 
                        for theta in thetas])
    shift = np.array([[-np.pi/2, .5]]*len(data2))
    return data1, data2 + shift

def plot_data(fname, *data):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(data)))
    for d, c in zip(data, colors):
        xs, ys = d[:,0], d[:,1]
        ax.plot(xs, ys, 'o', color=c)
    ax.set_aspect('equal')
    fig.savefig(fname)
    

if __name__ == '__main__':
    plot_data('blobs.pdf', *gen_three_blobs())
    plot_data('circles.pdf', *gen_circles())
    plot_data('snakes.pdf', *gen_snakes())


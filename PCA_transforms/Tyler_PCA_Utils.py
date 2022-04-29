from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def pca_transform(data, n_components):
    pca = PCA(n_components)
    data = pca.fit_transform(data)
    return data, pca

def pca_inverse_transform(data, pca):
    data = pca.inverse_transform(data)
    return data

def complete_transform(data, n_components):
    # split RGB of data
    print("Starting PCA transform")
    X=data.data
    R,G,B = X[:,:,:,0].reshape(X.shape[0],-1), X[:,:,:,1].reshape(X.shape[0],-1), X[:,:,:,2].reshape(X.shape[0],-1)
    
    # start of edits: 
    # use the split RGB values to create (32*32) x 3 data matrix T where R,G,B are 'variables'
    # then just run T through PCA and see if perform better
    
    # reshape R,G,B down to vectors *not sure if code correct
    R1 = R.reshape([1024,1])
    G1 = G.reshape([1024,1])
    B1 = B.reshape([1024,1])
    T = [R1,G1,B1] # creates 1024 x 3 matrix *?*
    
    # PCA transform
    print(np.isinf(T).any())
    T_proj, T_pca = pca_transform(T, n_components)
    print("T is transformed")
    
    # split back into 32x32 RGB matrices so can re-combine back into image
    r_proj,g_proj,b_proj = T_proj[:,0].reshape([32,32]), T_proj[:,1].reshape([32,32]), T_proj[:,2].reshape([32,32]))
    
    data_proj = np.stack((r_proj, g_proj, b_proj), axis=2)
    # reshape to square images
    data_proj = data_proj.reshape(X.shape[0], np.sqrt(n_components).astype('uint8'), 
                                    np.sqrt(n_components).astype('uint8'), 3)
    # inverse transform
    print("Beginning inverse transform")
    T_inv = pca_inverse_transform(T_proj, T_pca)
    # reshape to square
    print("Inverse transform has finished")
    # **check dimensions of reshape
    r_imgs = np.reshape(T_inv[:,0], [32,32,1])
    g_imgs = np.reshape(T_inv[:,1], [32,32,1])
    b_imgs = np.reshape(T_inv[:,1], [32,32,1])
    # concat
    reconstructed_imgs = np.stack((r_imgs, g_imgs, b_imgs), axis=3)
    
    return reconstructed_imgs, r_pca, g_pca, b_pca, data_proj

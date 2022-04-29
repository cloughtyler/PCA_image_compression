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
    
    # edits: use the split RGB values to create (32*32) x 3 data matrix T where R,G,B are 'variables'
    # then just run T through PCA and see if perform better
    
    T = [R,G,B]
    T = np.transpose(T) # (32*32) x 3 matrix
    
    # PCA transform
    print(np.isinf(T).any())
    T_proj, T_pca = pca_transform(T, n_components)
    print("T is transformed")
    g_proj, g_pca = pca_transform(G, n_components)
    print("Green is transformed")
    b_proj, b_pca = pca_transform(B, n_components)
    print("Blue is transformed")
    # concat
    data_proj = np.stack((r_proj, g_proj, b_proj), axis=2)
    # reshape to square images
    data_proj = data_proj.reshape(X.shape[0], np.sqrt(n_components).astype('uint8'), 
                                    np.sqrt(n_components).astype('uint8'), 3)
    # inverse transform
    print("Beginning inverse transform")
    r_inv = pca_inverse_transform(r_proj, r_pca)
    g_inv = pca_inverse_transform(g_proj, g_pca)
    b_inv = pca_inverse_transform(b_proj, b_pca)
    # reshape to square
    print("Inverse transform has finished")
    r_imgs = np.reshape(r_inv, (X.shape[0], X.shape[1], X.shape[2]))
    g_imgs = np.reshape(g_inv, (X.shape[0], X.shape[1], X.shape[2]))
    b_imgs = np.reshape(b_inv, (X.shape[0], X.shape[1], X.shape[2]))
    # concat
    reconstructed_imgs = np.stack((r_imgs, g_imgs, b_imgs), axis=3)
    
    return reconstructed_imgs, r_pca, g_pca, b_pca, data_proj

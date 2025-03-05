#%% Works in version  1.26.4
import numpy as np
from pyrsistent import v
from sympy import im
print("NumPy version:", np.__version__)
#%% Works in version 2.2.3
import pandas as pd
print("Pandas version:", pd.__version__)

#%%
hkl_ffc = [(1, 1, 1), (2, 0, 0), (1, 1, 3), (2, 2, 0),
(4,0,0), (3, 1, 1), (2, 2, 2), (3, 3, 1), (4, 2, 0), 
(4, 2, 2),(5, 1, 1), (3, 3, 3), (4, 4, 0), (5, 3, 1),
(4,4,2)]

# Platin lattice fringes
a_Pt = 3.92 # Å
def d(h, k, l, a):
    return round(a/np.sqrt(h**2+k**2+l**2),3)

for h, k, l in hkl_ffc:
    print(f'd({h}{k}{l}) = {d(h, k, l,a_Pt)} Å')

# Au lattice fringes
a_Au = 4.08 # Å

for h, k, l in hkl_ffc:
    print(f'd({h}{k}{l}) = {d(h, k, l,a_Au)} Å')

# Make a dataframe of Pt and Au lattice fringes
df = pd.DataFrame(columns=['hkl', 'd_Pt', 'd_Au'])
for h, k, l in hkl_ffc:
    df = pd.concat([df, pd.DataFrame([{'hkl': f'{h}{k}{l}', 'd_Pt': d(h, k, l, a_Pt), 'd_Au': d(h, k, l, a_Au)}])], ignore_index=True)
df  
# new column with the inverse of the d-values for Pt and Au
df['1/d_Pt'] = 1/df['d_Pt']
df['1/d_Au'] = 1/df['d_Au']

# Save the dataframe as a latex table
with open('lattice_fringes.tex', 'w') as f:
    f.write(df.to_latex(index=False
                        , column_format='|c|c|c|c|c|'
                        , escape=False
                        , caption='Lattice fringes for Pt and Au'
                        , label='tab:lattice_fringes'))

df
#%%
import os 
import hyperspy.api as hs # Works in version 1.7.3
import skimage as ski
import matplotlib.pyplot as plt
print(hs.__version__)
print(ski.__version__)
#%%
folder = os.getcwd()
#only tiff files
filenames = [f for f in os.listdir(folder) if f.endswith('.tif')]
file_path = [os.path.join(folder, f) for f in filenames]

# Load the images 
image_au = hs.load('AuC_1000kx_backup.tif')
image_au.plot()
im_au = image_au.data


#%%
image_pt = hs.load('Pt_particle_800kx_backup.tif')
image_pt.plot()
im_pt = image_pt.data

# %%
def compute_fft(array):
    fft = np.fft.fft2(array) #2d fft
    fft = np.fft.fftshift(fft) # shift the zero frequency component to the center

    pfft = ski.filters.gaussian(abs(fft), sigma=1)
    pfft_log = np.log(pfft)
    pfft_log-=pfft_log.min() #normalize
    pfft_log/=pfft_log.max()
    #enhance the contrast
    filtered_fft = pfft_log**1.5

    return filtered_fft

# %%
# plt.imshow(im_au, origin='upper',cmap='gray')
# plt.imshow(compute_fft(im_au), origin='upper',cmap='gray')


# %% Au 
fig, ax = plt.subplots(figsize =(7,7))
ax.imshow(compute_fft(im_au), origin='upper',cmap='gray')
for i in range(len(df['hkl'])-1):
    
    center = (im_au.shape[0]/2,im_au.shape[0]/2)
    radius = 10*image_au.axes_manager[0].scale*im_au.shape[0]/df['d_Au'][i] # in 1/nm

    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius*np.cos(theta)
    y = center[1] + radius*np.sin(theta)
    ax.plot(x, y, 'blue', lw=0.5)
    
    if i % 2 == 0:
        text_x = center[0] + radius*np.cos(np.pi/4)
        text_y = center[1] + radius*np.sin(np.pi/4)
    else:
        text_x = center[0] + radius*np.cos(np.pi/2)
        text_y = center[1] + radius*np.sin(np.pi/2)
    ax.text(text_x, text_y, '<'+df['hkl'][i]+'>', color='blue', fontsize=8)
# last one in red
g = len(df['hkl'])-1
center = (im_au.shape[0]/2,im_au.shape[0]/2)
radius = 10*image_au.axes_manager[0].scale*im_au.shape[0]/df['d_Au'][g] # in 1/nm

theta = np.linspace(0, 2*np.pi, 100)
x = center[0] + radius*np.cos(theta)
y = center[1] + radius*np.sin(theta)
ax.plot(x, y, 'r', lw=0.5)
text_x = center[0] + radius*np.cos(np.pi/4)
text_y = center[1] + radius*np.sin(np.pi/4)
ax.text(text_x, text_y, '<'+df['hkl'][g]+'>',
    color='r', fontsize=8)

l= 700

plt.xlim(im_au.shape[0]/2-l,im_au.shape[0]/2+l)
plt.ylim(im_au.shape[0]/2-l,im_au.shape[0]/2+l)
plt.tight_layout()
plt.axis('off')
plt.title('FFT of the AuC image')

plt.savefig('FFT_AuC_annotations.png', dpi=300)


df['1/d_Au']
df
# %% Pt
plt.imshow(compute_fft(im_pt), origin='upper',cmap='gray')
l= 700
plt.xlim(im_pt.shape[0]/2-l,im_pt.shape[0]/2+l)
plt.ylim(im_pt.shape[0]/2-l,im_pt.shape[0]/2+l)
plt.axis('off')

# %%
# divide the image in 4 quadrants and only plot the lower left quadrant
im_pt_11 = im_pt[im_pt.shape[0]//2:, :im_pt.shape[1]//2]


fig, (ax1,ax2) = plt.subplots(1,2,figsize =(8,4))
ax1.imshow(im_pt_11, origin='upper',cmap='gray')
ax1.axis('off')

ax2.imshow(compute_fft(im_pt_11), origin='upper',cmap='gray')
l= 500
ax2.set_xlim(im_pt_11.shape[0]/2-l,im_pt_11.shape[0]/2+l)
ax2.set_ylim(im_pt_11.shape[0]/2-l,im_pt_11.shape[0]/2+l)
plt.axis('off')
plt.tight_layout()

# divide the image in 4 quadrants and only plot the upper right quadrant
im_pt_22 = im_pt[:im_pt.shape[0]//2, im_pt.shape[1]//2:]

fig, (ax1,ax2) = plt.subplots(1,2,figsize =(8,4))
ax1.imshow(im_pt_22, origin='upper',cmap='gray')
ax1.axis('off')

ax2.imshow(compute_fft(im_pt_22), origin='upper',cmap='gray')
l= 500
ax2.set_xlim(im_pt_22.shape[0]/2-l,im_pt_22.shape[0]/2+l)
ax2.set_ylim(im_pt_22.shape[0]/2-l,im_pt_22.shape[0]/2+l)
plt.axis('off')
plt.tight_layout()

# %%
fig, ax = plt.subplots(figsize =(7,7))
ax.imshow(compute_fft(im_pt), origin='upper',cmap='gray')
for i in range(5):
    
    center = (im_pt.shape[0]/2,im_pt.shape[0]/2)
    radius = 10*image_pt.axes_manager[0].scale*im_pt.shape[0]/df['d_Pt'][i] # in 1/nm

    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius*np.cos(theta)
    y = center[1] + radius*np.sin(theta)
    ax.plot(x, y, 'blue', lw=0.5)
    
    if i % 2 == 0:
        text_x = center[0] + radius*np.cos(np.pi/4)
        text_y = center[1] + radius*np.sin(np.pi/4)
    else:
        text_x = center[0] + radius*np.cos(np.pi/2)
        text_y = center[1] + radius*np.sin(np.pi/2)
    ax.text(text_x, text_y, '<'+df['hkl'][i]+'>', color='blue', fontsize=8)
# last one in red
# g = len(df['hkl'])-1
# center = (im_pt.shape[0]/2,im_pt.shape[0]/2)
# radius = 10*image_pt.axes_manager[0].scale*im_pt.shape[0]/df['d_Pt'][g] # in 1/nm

# theta = np.linspace(0, 2*np.pi, 100)
# x = center[0] + radius*np.cos(theta)
# y = center[1] + radius*np.sin(theta)
# ax.plot(x, y, 'r', lw=0.5)
# text_x = center[0] + radius*np.cos(np.pi/4)
# text_y = center[1] + radius*np.sin(np.pi/4)
# ax.text(text_x, text_y, '<'+df['hkl'][g]+'>',
#     color='r', fontsize=8)

l= 700

plt.xlim(im_pt.shape[0]/2-l,im_pt.shape[0]/2+l)
plt.ylim(im_pt.shape[0]/2-l,im_pt.shape[0]/2+l)
plt.tight_layout()
plt.axis('off')
plt.title('FFT of the Pt image')

plt.savefig('FFT_Pt_annotations.png', dpi=300)


df['1/d_Pt']
df
# %%
fig, ax = plt.subplots(figsize =(7,7))
ax.imshow(compute_fft(im_pt_11), origin='upper',cmap='gray')
for i in range(5):
    
    center = (im_pt_11.shape[0]/2,im_pt_11.shape[0]/2)
    radius = 10*image_pt.axes_manager[0].scale*im_pt_11.shape[0]/df['d_Pt'][i] # in 1/nm

    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius*np.cos(theta)
    y = center[1] + radius*np.sin(theta)
    ax.plot(x, y, 'blue', lw=0.5)
    
    if i % 2 == 0:
        text_x = center[0] + radius*np.cos(np.pi/4)
        text_y = center[1] + radius*np.sin(np.pi/4)
    else:
        text_x = center[0] + radius*np.cos(np.pi/2)
        text_y = center[1] + radius*np.sin(np.pi/2)
    ax.text(text_x, text_y, '<'+df['hkl'][i]+'>', color='blue', fontsize=8)
# last one in red
# g = len(df['hkl'])-1
# center = (im_pt_11.shape[0]/2,im_pt_11.shape[0]/2)
# radius = 10*image_pt.axes_manager[0].scale*im_pt_11.shape[0]/df['d_Pt'][g] # in 1/nm

# theta = np.linspace(0, 2*np.pi, 100)
# x = center[0] + radius*np.cos(theta)
# y = center[1] + radius*np.sin(theta)
# ax.plot(x, y, 'r', lw=0.5)
# text_x = center[0] + radius*np.cos(np.pi/4)
# text_y = center[1] + radius*np.sin(np.pi/4)
# ax.text(text_x, text_y, '<'+df['hkl'][g]+'>',
#     color='r', fontsize=8)

l= 700/2    

plt.xlim(im_pt_11.shape[0]/2-l,im_pt_11.shape[0]/2+l)
plt.ylim(im_pt_11.shape[0]/2-l,im_pt_11.shape[0]/2+l)
plt.tight_layout()
plt.axis('off')
plt.title('FFT of the Pt image')

plt.savefig('FFT_Pt_annotations.png', dpi=300)


df['1/d_Pt']
df
# %% calculate the zone axis
v111 = np.array([1,1,1])
v200 = np.array([2,0,0])
#cros product
zone_axis = np.cross(v111,v200)
zone_axis


# %%

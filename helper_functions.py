import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt

def extract_passive_phases(v):
    is_previous_v_zero = False
    starting_points = []
    action_points = []
    for i in np.arange(0,len(v)):
        if v[i]==0:
            if not is_previous_v_zero:
                is_previous_v_zero = True
                starting_points += [i]
            elif (i==len(v)-1):
                action_points += [i]            
        elif (is_previous_v_zero):
            action_points += [i]
            is_previous_v_zero = False
    # This is needed if the recording ends in the passive control phase
    if len(action_points) < len(starting_points):
        starting_points = starting_points[:-1]
    return starting_points, action_points

def loadmat(filename):
    '''
    see stackoverflow for possible tweaks to scipy's loadmat
    https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    
    return data

def differentiate(t, x):
    x = np.append(x[0]*np.ones(3), np.append(x, x[-1]*np.ones(3)))
    
    timestep = np.median(np.diff(t))
    t = np.append(t[0]-np.arange(1,4)*timestep, np.append(t, t[-1]+np.arange(1,4)*timestep))

    # smooth noise-robust differentiators, see: 
    # http://www.holoborodko.com/pavel/numerical-methods/ \
    # numerical-derivative/smooth-low-noise-differentiators/#noiserobust_2
    v = (1*(x[6:]-x[:-6])/((t[6:]-t[:-6])/6) + 
         4*(x[5:-1] - x[1:-5])/((t[5:-1]-t[1:-5])/4) + 
         5*(x[4:-2] - x[2:-4])/((t[4:-2]-t[2:-4])/2))/32
    
    return v

def differentiate_2(t, x):
    x = np.append(x[0]*np.ones(3), np.append(x, x[-1]*np.ones(3)))
    timestep = np.median(np.diff(t))
    t = np.append(t[0]-np.arange(1,4)*timestep, np.append(t, t[-1]+np.arange(1,4)*timestep))
    a = (x[:-6] + 2*x[1:-5] - x[2:-4] - 4*x[3:-3] - x[4:-2] + 2*x[5:-1]+x[6:])\
            /(16*timestep*timestep)
    return a

def plot_pdf(data, var, ax=None, ls='-', bins=20):
    subjects = data.subject.unique().astype(int)
    if ax is None:
        fig, ax = plt.subplots()
    
    for subject in subjects:
        x = data.loc[data.subject==subject, var]
        subj_hist, subj_bins = np.histogram(x, bins=bins, normed=True)
        subj_hist[subj_hist<4/len(x)] = np.nan
        ax.plot((subj_bins[1:] + subj_bins[:-1])/2, subj_hist, label=subject, ls=ls, alpha=0.8)
    
    x = data.loc[:, var]
    mean_hist, mean_bins = np.histogram(x, bins=bins, normed=True)
    # only plot bins with 5 or more data points
    mean_hist[mean_hist<4/len(x)] = np.nan
    ax.plot((mean_bins[1:] + mean_bins[:-1])/2, mean_hist, ls=ls, alpha=1, color='k')
    
    ax.set_ylabel('pdf')
    ax.set_yscale('log')
    ax.set_ylim((0.005, 25))

    return ax

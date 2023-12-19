#Final edit: 20230727 16:06
#import required packages

#For file reading
import pandas as pd
import os

#For calibration 
import numpy as np
from scipy.interpolate import interp1d
from scipy import signal,linalg, fft as sp_fft 
from scipy.signal import find_peaks, butter 
from math import factorial

#For progress bar 
from tqdm import tqdm, tnrange, tqdm_notebook
#For plotting
from matplotlib import pyplot as plt
from matplotlib import gridspec

#Plot formatting
from matplotlib import rc,rcParams
from pylab import *

##############################################################################################################
#Set the ticks and label of figure
params = {        
         'axes.spines.right': False,
         'axes.spines.top': False,
         'axes.spines.left': True,
         'axes.spines.bottom': True,
          
         "font.weight" : "bold",
         "axes.linewidth" : 1.5,
         "axes.labelweight" : "bold",
         "axes.titleweight" : "bold"}

plt.rcParams.update(params)

##############################################################################################################
'''
To-do: 

Moving kernel for smoothing 
running average  / Gaussian kernel 
canonical lowpass filter 

shading of stim timing + duration 
add on stim identification results 
'''
##############################################################################################################

#Basic calibration

#Binning signals 
def binning_seg(data,sprt,bsz,min_snr,base_ave = None, base_sd = None, norm_meth = [False,False]):
    '''
    Parameters:
    ----------
    sprt: Sampling rate (Hz)
    bsz: Bin size (Frames)
    min_snr: Cutoff value for detecting responsive event (Z-score or normalized value)
    norm_meth: Vector with two Boolean values  
                First: restandardize the bin base on formula: F - base_ave / base_sd
                Second: Restandardize the bin base on formula: F - base_ave / base_ave

    Return:
    -------
    tt: Total length of signal (s)
    min_snr: Cutoff value for detecting responsive event (Z-score or normalized value)
    cnt: Array of binning results from original signal
    ext_seg: Array of time points with active event 
    int_seg: Array of time points with silent event 

    '''
    bsz = int(bsz) #Make sure binning unit is integer and > 1 frame  
    tt = max(data[:,0])

    #Perform binning 
    cnt = np.zeros(int(tt/(bsz/sprt)))#Total bin number
    for i in range(len(cnt)):
        tc = 0
        for j, t in enumerate(data[:,0]):  #Creating steps based on time 
            if t >= i*bsz/sprt and t < (i+1)*bsz/sprt:
                cnt[i] += data[j,1]
                tc+=1
        cnt[i] /= tc #Average signal
    if norm_meth[1]:
        base_sd = base_ave #dF/F formula (F-F0)/F0
    if norm_meth[0]:
        cnt -= base_ave
        cnt /= base_sd  #Averaged Z score  
    #Calibrate responsive segments  
    ext_seg = np.array([t for t, v in enumerate(cnt) if v >= min_snr])*(bsz/sprt)
    int_seg = np.array([t for t, v in enumerate(cnt) if v <= -min_snr])*(bsz/sprt)

    return tt, min_snr, np.array(cnt, dtype = 'float'), ext_seg, int_seg

#Normalization
def normalize(trace, dev, denominator): 
    """
    Parameters:
    ----------
    trace: 1D Array of calcium trace
    dev: value for the whole trace to be substracted with 
    denominator:  value for the whole trace to be divided with after substraction

    Return:
    -------
    1D array of normalized signal 
    """
   
    return np.array((trace - dev)/denominator, dtype = 'float')

#Trace segmentation & standardization 
def standardize_extraction(signal,baseline_period= None,baseline_cutoff = 1, absolute = False):
    """
    Parameters:
    ----------
    signal: 1D Array of calcium trace
    baseline_period: A two-element vector with index of the start/end points of baseline period, default set as None (All signal considered as baseline)
    baseline_cutoff: ratio from 0 to 1, determine the range of signals included as baseline range (data points below the assigned cutoff after normalized would be extracted, concatenated to a continuous baseline signal session)
    absolute: Requirement for adjusting baseline_cutoff based on the min/max of input signal  

    Return:
    -------
    Whole signal transformed into Z-score 
    baseline_sig: Partial signal from original trace with value below cutoff, defined as baseline period 
    ave: Average value of baseline period 
    sd: Standard deviation of baseline period 
    """

    baseline_sig = signal
    cutoff = baseline_cutoff if absolute else (np.nanmax(signal)- np.nanmin(signal))*baseline_cutoff + np.nanmin(signal)

    if len(baseline_period) != 0:#Baseline period defined 
        baseline_sig = signal[baseline_period[0]:baseline_period[1]]
        ave, sd = np.nanmean(baseline_sig), np.nanstd(baseline_sig)
    else:
        baseline_sig = signal[np.where(signal < cutoff)[0][()]]
        ave, sd = np.nanmean(baseline_sig), np.nanstd(baseline_sig)

    return normalize(signal, ave, sd), baseline_sig, ave, sd 

#Special type of low-pass filter 
def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """
    Parameters:
    ----------
    y : array_like, shape (N,)
      the values of the time history of the signal.
    window_size : int
      the length of the window. Must be an odd integer number.
    order : int
      the order of the polynomial used in the filtering.
      Must be less then `window_size` - 1.
    deriv: int
      the order of the derivative to compute (default = 0 means only smoothing)
    
    Return:
    -------
    ys : ndarray, shape (N)
      the smoothed signal (or it's n-th derivative).
    
    Notes
    -----
    Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at 0
    """
    
    try:
        window_size = abs(int(window_size))
        order = abs(int(order))
    except ValueError:#, msg
        raise ValueError("window_size and order have to be of type int")
        if window_size % 2 != 1 or window_size < 1:
            raise TypeError("window_size size must be a positive odd number")
        if window_size < order + 2:
            raise TypeError("window_size is too small for the polynomials order")

    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

# Upsampling & downsampling
def resample(x, num, t=None, axis=0, window=None, domain='time'):
    """
    Parameters:
    ---------
    x : array_like
        The data to be resampled.
    num : int
        The number of samples in the resampled signal.
    t : array_like, optional
        If `t` is given, it is assumed to be the equally spaced sample
        positions associated with the signal data in `x`.
    axis : int, optional
        The axis of `x` that is resampled.  Default is 0.
    window : array_like, callable, string, float, or tuple, optional
        Specifies the window applied to the signal in the Fourier
        domain.  See below for details.
    domain : string, optional
        A string indicating the domain of the input `x`:
        ``time`` Consider the input `x` as time-domain (Default),
        ``freq`` Consider the input `x` as frequency-domain.

    Return:
    -------
    resampled_x or (resampled_x, resampled_t)
        Either the resampled array, or, if `t` was given, a tuple
        containing the resampled array and the corresponding resampled
        positions.
    """
    if domain not in ('time', 'freq'):
        raise ValueError("Acceptable domain flags are 'time' or"
                         " 'freq', not domain={}".format(domain))
    x = np.asarray(x)
    Nx = x.shape[axis]

    # Check if we can use faster real FFT
    real_input = np.isrealobj(x)

    if domain == 'time':
        # Forward transform
        if real_input:
            X = sp_fft.rfft(x, axis=axis)
        else:  # Full complex FFT
            X = sp_fft.fft(x, axis=axis)
    else:  # domain == 'freq'
        X = x

    # Apply window to spectrum
    if window is not None:
        if callable(window):
            W = window(sp_fft.fftfreq(Nx))
        elif isinstance(window, np.ndarray):
            if window.shape != (Nx,):
                raise ValueError('window must have the same length as data')
            W = window
        else:
            W = sp_fft.ifftshift(signal.get_window(window, Nx))

        newshape_W = [1] * x.ndim
        newshape_W[axis] = X.shape[axis]
        if real_input:
            # Fold the window back on itself to mimic complex behavior
            W_real = W.copy()
            W_real[1:] += W_real[-1:0:-1]
            W_real[1:] *= 0.5
            X *= W_real[:newshape_W[axis]].reshape(newshape_W)
        else:
            X *= W.reshape(newshape_W)

    # Copy each half of the original spectrum to the output spectrum, either
    # truncating high frequences (downsampling) or zero-padding them
    # (upsampling)

    # Placeholder array for output spectrum
    newshape = list(x.shape)
    if real_input:
        newshape[axis] = num // 2 + 1
    else:
        newshape[axis] = num
    Y = np.zeros(newshape, X.dtype)

    # Copy positive frequency components (and Nyquist, if present)
    N = min(num, Nx)
    nyq = N // 2 + 1  # Slice index that includes Nyquist if present
    sl = [slice(None)] * x.ndim
    sl[axis] = slice(0, nyq)
    Y[tuple(sl)] = X[tuple(sl)]
    if not real_input:
        # Copy negative frequency components
        if N > 2:  # (slice expression doesn't collapse to empty array)
            sl[axis] = slice(nyq - N, None)
            Y[tuple(sl)] = X[tuple(sl)]

    # Split/join Nyquist component(s) if present
    # So far we have set Y[+N/2]=X[+N/2]
    if N % 2 == 0:
        if num < Nx:  # downsampling
            if real_input:
                sl[axis] = slice(N//2, N//2 + 1)
                Y[tuple(sl)] *= 2.
            else:
                # select the component of Y at frequency +N/2,
                # add the component of X at -N/2
                sl[axis] = slice(-N//2, -N//2 + 1)
                Y[tuple(sl)] += X[tuple(sl)]
        elif Nx < num:  # upsampling
            # select the component at frequency +N/2 and halve it
            sl[axis] = slice(N//2, N//2 + 1)
            Y[tuple(sl)] *= 0.5
            if not real_input:
                temp = Y[tuple(sl)]
                # set the component at -N/2 equal to the component at +N/2
                sl[axis] = slice(num-N//2, num-N//2 + 1)
                Y[tuple(sl)] = temp

    # Inverse transform
    if real_input:
        y = sp_fft.irfft(Y, num, axis=axis)
    else:
        y = sp_fft.ifft(Y, axis=axis, overwrite_x=True)

    y *= (float(num) / float(Nx))

    if t is None:
        return y
    else:
        new_t = np.arange(0, num) * (t[1] - t[0]) * Nx / float(num) + t[0]
        return y, new_t

# Low-pass filter
def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    #y = lfilter(b, a, data)
    y = signal.filtfilt(b, a, data)
    return y

# AUC calibration (linear trapezoid method)
def AUC_cal(trace, dt, tt):
    """
    Parameters:
    ---------
    trace: 1D array of signal 
    dt: time interval between two adjacent points (s)
    tt: total length of trace (s)

    Return:
    ---------
    scalar implying the area-under-curve of input trace for unit of time (1 s)
    """
    sig_sum = [(trace[i] + trace[i+1])/2 for i in range(len(trace)-1)]
    return (sum(sig_sum)*dt)/tt
##############################################################################################################
#Other functions for determining categorical info 

#For neural classification based on binning results
def neuron_res(ext_seg,inh_seg,cutoff):
    """
    Parameters:
    ---------
    ext_sig: scalar implying the amount of excited bins 
    inh_sig: scalar implying the amount of inhibited bins 
    cutoff: minimum amount of responsive bin to define reponsiveness

    Return:
    ---------
    classification result 
    """
    return ['EXT' if ext_seg>= cutoff else 'INH' if inh_seg >= cutoff else 'NR'][0]

# Determing frame corresponded to specific timing 
def start_end_index(time, start, pre,post, decimal = 0):
    """
    Parameters:
    ---------
    time: 1D array of time values 
    start: intended start time point for signal extraction 
    pre: lenght of time inducded before intended timing 
    post: length of time included after intended timing

    Return:
    ---------
    index representing the start & end index for range extracted from original time array  
    """
    return np.array([np.nanmin(np.where(np.round(time,decimal)== round(start-pre,decimal))[0]),\
         np.nanmax(np.where(np.round(time,decimal)== round(start+post-1,decimal))[0])])


# Interactive script for manual typing in stimulus name & corresponded timing
def stim_time_input():
    '''
    A interactive function for manual typing in stimulus name & corresponded timing

    Return:
    ---------
    1. Array of introduced stimulus 
    2. Array of corresponding timings for every stimulus onset

    Note: 
        Called out whenever automatic stimulus dectection from time table file is unavailable. 
        Stimulus and corresponding onset timing should be sequentially and manually typed in.  
    '''
    #Generate stimulus list & corresponding time points 
    stim , n = [], 1
    while True: 
        sti = input(f'Please type in stimulus #{n}')
        if  sti == '':
            print('Stimulus assignment done!')
            break
        else:
            stim.append(sti) 
            n +=1
    return np.array(stim) , np.array([input(f'Introducing time point of {sti} (sec)') for i , sti in enumerate(stim)],dtype = 'int')

# Automatically identifying stimulus & stimulus timing
def stim_time_search(data,file_n,str_col = 1):

    '''
    Parameters:
    ---------
    data: File of time table (array or table)
    file_n: File name (String) 
    str_col: Integer, the "index" where the column with the timing of first stimulus
    Return:
    ---------
    stim: Array of introduced stimulus 
    stim_time: Array of corresponding timings for every stimulus onset

    Note: 
    Filename in timetable shall be exaclty the same as that of preprocessed file with 'Proc ' prefix and '.csv' suffix removed
    '''
    
    # Find out the column with file name
    file_name = np.where(data[0,:] == 'File name')[0]
    # transform the string of minute into float of second, x is the str
    MintoSec = lambda x: float(x.split(':')[0])*60+float(x.split(':')[-1])

    temp_file = file_n.removesuffix('.csv')
    temp_file = temp_file.split('Proc ')[-1] #If the data is preprocessed - Remove the label in file name 

    match_row = np.where(data[:,file_name] == temp_file)[0]
    # print(match_row)
    # stim_index = np.where(np.isnan(a[match_row,4:]))
    if len(match_row) ==0:
        print('No corresponded stimulus time points detected, switch to manual mode')
        stim , stim_time = stim_time_input()
    else: 
        stim_index = [i for i ,v in enumerate(data[match_row,str_col:][0]) if not pd.isna(v)]
        stim = data[0,str_col:][stim_index]
        stim_time = [float(data[match_row,str_col:][0][ind]) for ind in stim_index]

    for i ,s in enumerate(stim):
        print(f'Stim #{i}: {s}; Time: {stim_time[i]} s')
        
    return stim, stim_time

# Script for signal preprocessing
# %%
#Whole trace preprocessing 
def ca_preprocess(file_info, resamp_info, lp_filter_info, standardize_info, preproc_plot):
    '''
    Parameters:
    ---------
    file_info: dictionary of information for files to be processed 
    resamp_info: dictionary of information for signal resampling 
    lp_filter_info: dictionary containing information for filter 
    standardize_info: dictionary containing information for Z-score signal conversion 
    preproc_plot: assigned parameters for plotting out preprocessed results 
    '''
    #Search for files with assigned format 
    file_list = [_ for _ in os.listdir(file_info['file path']) if _.endswith(file_info['file format'])]
    print(file_list)
    #Directory for output saving 
    newpath = os.path.join(file_info['file path'],'Preprocessed')
    if not os.path.isdir(newpath): #Create folder if it does not exist 
        os.makedirs(newpath)

    for ii in tnrange(len(file_list), desc = 'All files'): #Loop through all files 
        #Move to directory with file to process
        os.chdir(file_info['file path'])

        file_n =  file_list[ii]
        #Call out file reading functions based on file format
        if file_info['file format'] == '.csv': #csv file
            data = pd.read_csv(file_n, header=None,encoding = 'latin', engine='c').values
        # for i,file_n in enumerate(files):
        elif file_info['file format'] == '.xlsx': #excel file
            data = pd.read_excel(file_n, header=None,encoding = 'latin').values
        elif file_info['file format'] == '.npz': #Numpy array file 
            with np.load(file_n) as d:
                data = np.array([d[x] for x in d])
                data = np.reshape(data,[data.shape[1],data.shape[2]]) #Reshape into 2-D array
        else:
            print(f'Specified file format ({file_info["file format"]}) not supported') #Drop out of loop if unsupported file type was included 
            break
        print(f'File in process: {file_n}')

        #Data preparation
        if file_info['recording type'] == 'photometry': #one dimension 
            signal = np.array(data[1:,],dtype = 'float')
            cellID = [''] #adding sham ID to match output table format 
        elif file_info['recording type'] == 'miniscope': #Multi-units calcium imaging
            #Remove rejected or undecided traces 
            accept_indx = np.concatenate([np.array([0]),np.where(data[1,:] == ' accepted')[0]]) #Indexes of time column & accepted units
            cellID = np.array(data[0,accept_indx[1:]],dtype = 'str')
            signal = np.array(data[2:,accept_indx],dtype = 'float')# time at first column, calcium signals followed  
        else: 
            print('Not valid specification of recording type')
            continue#break

        #Detect discontinuity in time and correct them
        dt = signal[1,0]-signal[0,0]
        dev = signal[1:,0]-signal[:-1,0]# [signal[i,0]-signal[i-1,0] for i in range(1,len(signal[:,0]))] 
        discontinuous_points = np.where(dev > dt*100)[0]
        if discontinuous_points.size == 0:
            print('No break point detected')
        else:
            print(f'Seg index: {discontinuous_points+1}')

        for i,t in enumerate(discontinuous_points):
            if i+1 < 1: #Discontinuous points before the last one 
                print(signal[t+1,0] - signal[t,0])
                signal[t+1:discontinuous_points[i+1],0] -= (signal[t+1,0] - signal[t,0])
            else:
                signal[t+1:,0] -= (signal[t+1,0] - signal[t,0])

        #Interpolation 
        if resamp_info['perform resampling']: #if perform resampling is specified as "True"
            signal_new = np.linspace(min(signal[:,0]),max(signal[:,0]),new_n := int(signal.shape[0]*resamp_info['resampling factor'])) #Resample the "Time"
            # signal_new = resample(signal[:,1], , t=signal[:,0])[1]#Time column 
            for i in tnrange(1,signal.shape[1], desc= f'Resample {file_n} progress'):
                signal_new = np.c_[signal_new, resample(signal[:,i], new_n)]
                if resamp_info['plot results']:
                #Visualize resampling result - For quality check 
                    plt.figure(figsize = (6,2))
                    plt.plot(signal[:,0],signal[:,i],color = 'red',label ='Original')
                    plt.plot(signal_new[:,0],signal_new[:,i],color = 'blue',label = f'Resampled (factor = {resamp_info["resampling factor"]})')
                    plt.title(f'Resample result: {file_n}')
                    plt.xlabel('Time (s)')
                    plt.legend(loc = 1)
                    os.chdir(newpath)
                    [plt.savefig(f'Resample result({file_n.removesuffix(file_info["file format"])} - {cellID[i-1]}).{ff}', dpi=300, transparent=True, bbox_inches='tight') \
                    if preproc_plot['save'] else None for ff in preproc_plot['export format']]
                    plt.show()
        else: 
            signal_new = np.copy(signal)

        #Preprocessing 

        #Defining baseline period
        if standardize_info['baseline length'] == None or standardize_info['baseline length'] == 0:#All signal as global baseline 
            sig_base = [0,signal_new.shape[0]]
        else:#Apply baseline range specified by user 
            sig_base =  start_end_index(signal_new[:,0], standardize_info['baseline start'], 0, standardize_info['baseline length'])

        plot_n = 1 
        for i in tnrange(1,signal_new.shape[1], desc = f"Preprocess {file_n} progress"):# Loop through all cell traces
            #Time of trace
            tr_time = signal_new[:,0]
            #Smooth the signal 
            sm_tr = savitzky_golay(signal_new[:,i], lp_filter_info['window size'], lp_filter_info['polynomial order'])
            smooth_all = np.copy(sm_tr) if i == 1 else np.c_[smooth_all, sm_tr]           
            #Normalize signal 
            norm_tr = normalize(sm_tr, np.nanmin(sm_tr), np.nanmax(sm_tr) -np.nanmin(sm_tr))
            norm_all = np.copy(norm_tr) if i==1 else np.c_[norm_all, norm_tr]
            #Convert to Z-score
            z_tr = standardize_extraction(sm_tr,sig_base,baseline_cutoff=1, absolute=False)[0] 
            z_all = np.copy(z_tr) if i == 1 else np.c_[z_all,z_tr]

            #Visualize processing result
            if preproc_plot['show'] and plot_n <= preproc_plot['max plot number']:
                fig ,axes = plt.subplots(nrows = 1,ncols = 3,figsize = (12,3),sharex = True, sharey=False)
                #Smoothed signal 
                axes[0].plot(tr_time,signal_new[:,i],label ='Original',color = 'cyan',alpha = 0.3)
                axes[0].plot(tr_time,sm_tr,label = 'Denoised',color = 'black')
                axes[0].set(xlabel='Time (s)',title='Smoothing result', ylabel='dF/F (%)')
                axes[0].legend(loc = 'best' ,frameon = False)
                #Normalized trace
                axes[1].plot(tr_time,norm_tr,color = 'black')
                axes[1].set(xlabel='Time (s)',title=f'{file_n.removesuffix(file_info["file format"])} {cellID[i-1]}\nNormalized result', ylabel='Scaled dF/F')
                #Z-score 
                axes[2].plot(tr_time, z_tr , color = 'black')
                axes[2].set(xlabel='Time (s)',ylabel=f'Z-score ($\sigma$)',title='Standardized signal')
                fig.tight_layout()
                os.chdir(newpath)
                [fig.savefig(f'Preprocessing result({file_n.removesuffix(file_info["file format"])} - {cellID[i-1]}).{ff}', dpi=300, transparent=True, bbox_inches='tight') \
                if preproc_plot['save'] else None for ff in preproc_plot['export format']]
                plot_n +=1
        out_meth = np.array(['dFF','Norm','Z-score'])
        keep = file_info['keep scale'] 

        #Remove unwanted signal scale from exported file 
        delete_inx = [np.arange(len(cellID)*(j),len(cellID)*(j+1),1)  for j ,d in enumerate(keep) if not d] 
        total_sig = np.delete(np.c_[smooth_all, norm_all, z_all],np.concatenate(delete_inx),1)\
             if len(delete_inx)>0 else np.c_[smooth_all, norm_all, z_all]
        
        #Preparing exported file 
        out_ID = np.tile(cellID, sum(keep))
        out_meth = np.concatenate([(np.tile(out_meth[j],len(cellID)))  for j , d in enumerate(keep) if d])

        #First column 
        first_col = np.concatenate([['Signal type','Cell ID'],np.array(tr_time,dtype = 'str')])

        #Combine 
        out_table = np.transpose(np.c_[out_meth, out_ID, np.transpose(total_sig)])
        out_table= np.c_[first_col, out_table]
        os.chdir(newpath) 
        file_n2 = file_n.removesuffix(file_info['file format']) #Base name of given file 
        [np.savetxt(f'Sep points {file_n2}.txt', np.insert(discontinuous_points+1,0,0), delimiter ="    ",fmt ='% s') if discontinuous_points.size >0 else None] #Save file for further segmentation
        [np.savetxt(f'Proc {file_n2}{ff}', out_table, delimiter =",",fmt ='% s') for ff in ['.csv']]  #Save preprocessed results in subfolder 
    print('All procedure completed.')

# Script for signal extraction & basic descriptive statistics 
def sig_ext_cal_cmbind(file_info, segmentation_info, standardize_info, binning_info, spike_detection_info, plotting_info):
    '''
    Parameters:
    ---------
    file_info: dictionary of information for files to be processed 
    segmentation_info: dictionary of information for extracting signals of certain timing 
    standardize_info: dictionary containing information for Z-score signal conversion 
    binning_info:  dictionary containing information for binning & unit response determination 
    spike_detection_info, plotting_info): dictionary containing information for detecting spikes in extracted signal 
    plotting_info: assigned parameters for plotting out results
    '''

    file_list = [_ for _ in os.listdir(file_info['file path']) if _.endswith(file_info['file format'])]

    file_name, total_ID, total_stim, total_extseg,\
        total_intseg ,tol_res_type,total_AUC,  total_pre_AUC,\
            total_spike , total_base_ave, total_base_sd\
            = [], [], [], [], [], [], [], [], [], [], []
    
    #Determine signal scale 
    meth = segmentation_info['signal to use']

    #Determine binning scale
    norm_meth = [binning_info['method'] == 'Z-score',binning_info['method'] == 'dFF']
    
    #Directory for saving outputs 
    newpath2 = os.path.join(file_info['file path'] ,'Extracted')
    if not os.path.isdir(newpath2):
        os.makedirs(newpath2)

    total_cnt = 0 #Index of total processed cells/traces 
    for i in tnrange(len(file_list), desc = 'All files'):#For every file 
        file_n = file_list[i]
        os.chdir(file_info['file path'])
        if file_n.startswith(file_info['skip prefix']):
            print(f'{file_n} skipped')
            continue
        else:
            data = pd.read_csv(file_n, header=None, low_memory=False).values
            # proc_file.append(i)
            print(f'File in process: {file_n}')

            if file_info['recording type'] == 'miniscope':
                type_indx = np.where(data[0,:]== meth)[0] 
                cellID = data[1,type_indx]
                signal = np.array(data[2:,type_indx],dtype = 'float')
            elif file_info['recording type'] == 'photometry': 
                signal = np.array(data[2:,np.where(data[0,:]== meth)[0]],dtype = 'float')
                cellID = ['']
            else:
                print('Not valid specification of recording type')
                break
            #Creat all common time column for all cells and all files 
            if total_cnt== 0: #Prevent error occuring from skipped files 
                uni_time_range = start_end_index(np.array(data[2:,0], dtype = 'float'), 0, 0, segmentation_info['pre']+segmentation_info['post']) #Universal time range for all extracted signals
                time = np.linspace(-segmentation_info['pre'],segmentation_info['post'],uni_time_range[1]-uni_time_range[0]) #First and universal time column
                ext_sig, bin_ori = np.copy(time), np.copy(time) #Extract signal table with original signals (initiated with time column)
                bins = np.linspace(1,n:=len(binning_seg(np.c_[time,time],binning_info['sampling rate'],binning_info['bin size'],binning_info['cutoff SNR'])[2]),n)#Binning results 
                baseline_pre_inx = [0,np.nanmin(np.where(np.floor(time) == 0)[0])] #Frame of t = 0s 

            #assign stimuli and corresponded time points 
            if segmentation_info['time table'].size == 0: #Time table = none
                stim, stim_time = stim_time_input()
            else: 
                stim , stim_time = stim_time_search(segmentation_info['time table'],file_n,segmentation_info['first stim column']-1)

            if file_info['recording type'] == 'miniscope':
                fig = plt.figure(figsize = (3*len(stim),0.05*len(cellID)))#tight_layout=True,
                # fig, ax = plt.subplots(nrows = 1,ncols = len(stim), figsize = (12,3*len(stim)))
                fin_ratio =  np.array(np.repeat(4,len(stim)),dtype = 'float')
                fin_ratio[-1] += 0.5 #add space for color bar 
                spec = gridspec.GridSpec(ncols=len(stim), nrows=1,width_ratios=fin_ratio)
            else: #Photometry 
                fig, ax = plt.subplots(1,1,figsize = (10,3))

            for j in tnrange(len(stim), desc = f'{file_n} all stim progress'): #For every stimulus 
                sti = stim[j]
                sti_range = start_end_index(np.array(data[2:,0], dtype = 'float'), stim_time[j],  segmentation_info['pre'],segmentation_info['post'])
                for k in tnrange(signal.shape[1], desc = f"{sti} extr progress",leave = True):#For every cell 
                    #Signal extraction 
                    #Resmpling all signals so that they have the same length 
                    trace = resample(standardize_extraction(signal[:,k],sti_range)[1], len(time)) 
                    ext_sig = np.c_[ext_sig, trace]#Extracted signal
                    total_cnt += 1
                    #Restandardize if local baseline assigned 
                    if  standardize_info['apply local baseline'] and segmentation_info['pre'] > 0:
                        #Convert segment signal to the scale of Z-score based on pre-stimulus signals 
                        Z = standardize_extraction(ext_sig[:,total_cnt],baseline_pre_inx)
                        bin_ori = np.c_[bin_ori, Z[0]]
                        total_base_ave.append(Z[2])
                        total_base_sd.append(Z[3])
                    else:
                        base_range = start_end_index(np.array(data[2:,0], dtype = 'float'), \
                                                standardize_info['global baseline range'][0],0, \
                                                    standardize_info['global baseline range'][1]- standardize_info['global baseline range'][0])
                        base = standardize_extraction(signal[:,k],base_range)
                        total_base_ave.append(base[2])
                        total_base_sd.append(base[3])
                        #Directly combine original signal 
                        bin_ori = np.c_[bin_ori,ext_sig[:,total_cnt]]
                    #Binning 
                    if binning_info['standardized source'] and segmentation_info['pre'] > 0:#Use standardized signal for binning (Only execute when pre-stim baseline exist) 
                        bins_res =  binning_seg(np.c_[time,bin_ori[:,total_cnt]],binning_info['sampling rate'],binning_info['bin size'],binning_info['cutoff SNR'])
                    else:#Use original signal for binning 
                    #If the imported data is presented in Z-score format , restandardization is required in binning step
                        bins_res =binning_seg(np.c_[time,ext_sig[:,total_cnt]],binning_info['sampling rate'],binning_info['bin size'],binning_info['cutoff SNR'],\
                            base_ave = total_base_ave[total_cnt-1], base_sd =  total_base_sd[total_cnt-1],norm_meth=norm_meth)
                    bins = np.c_[bins, bins_res[2]]
                    total_extseg.append(len(bins_res[3]))
                    total_intseg.append(len(bins_res[4]))
                    tol_res_type.append(neuron_res(len(bins_res[3]), len(bins_res[4]), binning_info['response criteria']))

                    #AUC 
                    total_AUC.append(AUC_cal(ext_sig[baseline_pre_inx[1]:,total_cnt], 1/binning_info['sampling rate'], segmentation_info['post'])) #Only AUC after t = 0 would be included 
                    if segmentation_info['pre'] >0 and standardize_info['apply local baseline']:
                        total_pre_AUC.append(AUC_cal(ext_sig[:baseline_pre_inx[1],total_cnt], 1/binning_info['sampling rate'], segmentation_info['pre']))#Pre stimulus AUC
                    else:
                        total_pre_AUC.append(AUC_cal(base[1],1/binning_info['sampling rate'],standardize_info['global baseline range'][1]- standardize_info['global baseline range'][0]))

                    #Spike detection 
                    peaks_x1 , properties_x1 = find_peaks(ext_sig[:,total_cnt], spike_detection_info['prominence'], spike_detection_info['width'])
                    total_spike.append(len(peaks_x1))#Events number 

                print(f"\rSignals associated to stimulus {sti} are extracted and processed.",end = '') 

                if file_info['recording type'] == 'miniscope': #plot heatmap 
                    ax = plt.subplot(spec[j])
                    im =ax.imshow(ext_sig[:,-signal.shape[1]:].T, vmin= plotting_info['heatmap range'][0], vmax = plotting_info['heatmap range'][1], \
                                interpolation = 'none', cmap = plotting_info['color map'],aspect='auto') ##Plot segmentated traces as heatmap 
                    ax.set(title = sti, ylabel = 'cell #' if j==0 else '', xlabel = 'Time from\nstim onset (s)', \
                            yticks = [0, signal.shape[1]-1], yticklabels = [1, signal.shape[1]],
                            xticks = [0,baseline_pre_inx[1], ext_sig.shape[0]-2], \
                            xticklabels = [round(np.nanmin(time)), 0, round(np.nanmax(time))])
                    ax.vlines([baseline_pre_inx[1]], 0 , signal.shape[1]-1, linestyle = 'dotted', color = 'white')#indicating timing of stim onset 
                else: #Photometry: plot traces
                    ax.plot(time, ext_sig[:,-1], label = sti)
            #plotting
            os.chdir(newpath2)   
            if file_info['recording type'] == 'miniscope':  
                # fig.suptitle(f"{file_n.removesuffix(file_info['file format'])} segmentated cell traces", weight='bold')
                fig.colorbar(im, label = f"{file_n.removesuffix(file_info['file format']).removeprefix('Proc')}\n{meth}")
                fig.tight_layout()
                [fig.savefig(f'{file_n.removesuffix(file_info["file format"])} extr signal heatmap.{ff}', dpi=300, transparent=True, bbox_inches='tight') \
                    if plotting_info['save plot'] else None for ff in plotting_info['export format']]

            else: #Photometry 
                ax.set(xlabel='Time from stim on (s)',ylabel= meth,title=file_n,ylim= plotting_info['y range'])
                ax.legend(frameon = False)
                [fig.savefig(f'{file_n.removesuffix(file_info["file format"])} extr traces plot.{ff}', dpi=300, transparent=True, bbox_inches='tight') \
                if plotting_info['save plot'] else None for ff in plotting_info['export format']]

            #Updating total otput file 
            total_ID.append(np.tile(cellID,len(stim)))
            total_stim.append(np.repeat(stim,len(cellID)))
            file_name.append(np.tile(file_n,total_len := len(cellID)*len(stim)))
    #Preparing exported file 
    
    #First column 
    exp_parameters = ['File name','Cell ID','Stimulus',\
                    f'Baseline mean ({"local" if standardize_info["apply local baseline"] else "global"})',\
                    f'Baseline std ({"local" if standardize_info["apply local baseline"] else "global"})',\
                    f'{"Local" if standardize_info["apply local baseline"] else "Global"} base AUC',\
                        'AUC','Spike count',
                    f'Excited bin (cutoff = {binning_info["cutoff SNR"]}x S.D)',\
                    f'Silenced bin (cutoff = -{binning_info["cutoff SNR"]}x S.D)',
                    f'Bin response (>={binning_info["response criteria"]} responsive bins)',\
                    f'Bin SNR ({binning_info["bin size"]/binning_info["sampling rate"]} s; scale = {binning_info["method"]})']
    
    bin_num = np.array(np.linspace(1,n,n),dtype = 'str')
    first_col = np.concatenate([exp_parameters,bin_num,['','Cell ID','Time (s)'],np.array(ext_sig[:,0],dtype = 'str')])
    #Merge 
    out_table = np.array(np.vstack([np.concatenate(file_name), np.concatenate(total_ID), np.concatenate(total_stim),\
    total_base_ave,total_base_sd,total_pre_AUC,total_AUC,total_spike,total_extseg,total_intseg,tol_res_type,np.tile('',len(total_AUC))]),dtype = 'str')
    out_table2 = np.r_[out_table, np.array(bins[:,1:],dtype= 'str')]
    out_table3 = np.r_[out_table2, np.vstack([np.tile('',len(total_AUC)),np.concatenate(total_ID),np.concatenate(total_stim)]), np.array(ext_sig[:,1:],dtype = 'str')]
    out_table_fin= np.c_[first_col, out_table3]

    os.chdir(newpath2) 
    print('Saving file...')
    [np.savetxt(f'{meth} signal extraction & stats{ff}', out_table_fin, delimiter =",",fmt ='% s') for ff in ['.csv']]
    print('File saved, procedure ended.')
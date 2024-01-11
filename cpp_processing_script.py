#Final editing: 20231217 23:56
import numpy as np
import pandas as pd 
from matplotlib import pyplot as plt
import os
import csv
from matplotlib import gridspec
from scipy import fft as sp_fft
from scipy import signal
import itertools
import pickle

np.seterr(divide='ignore',invalid='ignore')

from matplotlib import rc,rcParams
from pylab import *
# Set the ticks and label of figure
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

def resample(x, num, t=None, axis=0, window=None, domain='time'):
    """
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
    Returns
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

def norm(sig):
    """
    Inputs
    -------
    sig : 1-D array
        The data to be normalized.
    Returns
    -------
    normalized input signal
    """
    min_s, max_s = np.array(sig).min() ,np.array(sig).max() 
    return (sig - min_s)/(max_s-min_s)
    
def euclidian_distance(pos1,pos2):
    """
    Inputs
    -------
    pos1, pos2 : 1-D array 
        denoting the coordinates of two points

    Returns
    -------
    The euclidian distance between the two points
    """
    return np.sqrt(sum([(pos1[i]**2 + pos2[i]**2) for i in range(2)]))
    
def standardize_extraction(signal,baseline_period= None,baseline_cutoff = 1, absolute = False):
    """
    Input
        1. signal: 1D Array of calcium trace
        2. baseline_period: A two-element vector with index of the start/end points of baseline period, default set as None (All signal considered as baseline)
        3. baseline_cutoff: ratio from 0 to 1, determine the range of signals included as baseline range (data points below the assigned cutoff after normalized would be extracted, concatenated to a continuous baseline signal session)
        4. absolute: Requirement for adjusting baseline_cutoff based on the min/max of input signal  
    Output
        1. Whole signal transformed into Z-score 
        2. baseline_sig: Partial signal from original trace with value below cutoff, defined as baseline period 
        3. ave: Average value of baseline period 
        4. sd: Standard deviation of baseline period 
    """
    baseline_sig = signal
    cutoff = baseline_cutoff if absolute else (np.max(signal)- np.min(signal))*baseline_cutoff + np.min(signal)

    if not baseline_period == None:#Baseline period defined 
        baseline_sig = signal[baseline_period[0]:baseline_period[1]]
        ave, sd = baseline_sig.mean(), np.std(baseline_sig)
    else:
        baseline_sig = signal[np.where(signal < cutoff)[0][()]]
        ave, sd = baseline_sig.mean(), np.std(baseline_sig)

    return normalize(signal, ave, sd), baseline_sig, ave, sd 

def start_end_index(time, start, pre,post, decimal = 0):
    """
    Inputs
    -------
    time : 1-D array 
        Time series for time
    start : float
        Timing (s) to align as the 0 s of extracted signal
    pre : float
        Lenght of time to include before "start"
    post : float
        Lenght of time to include after "start"
    decimal : integer
        Number of decimal for finding best-fit index of time that match with specified timing

    Returns
    -------
    [Index of start, Index of end]
    """

    return [np.where(np.floor(time*(10**decimal))/10**decimal == np.floor(start-pre))[0].min(),\
         np.where(np.floor(time*(10**decimal))/10**decimal == np.floor(start+post-1))[0].max()]

def normalize(trace, dev, denominator): 
    """
    Inputs
    -------
    trace : 1-D array 
        Time series to be standardized
    dev : float
        The number to subtract from the entire input time series
    denominator : float
        The number for the subtracted signal to divid with
    Returns
    -------
    An array of standardized signal
    """
    return np.array((trace - dev)/denominator, dtype = 'float')

res_det = lambda r,thr: ['EX' if r > thr else 'INH' if r < -thr else 'NS'][0] #Response determination 

def traj_analyze_CPP(data,f_name,gate_range,t_delay,f_rate,\
                     crop_length, pref_side, plot_raw,save_plot,save_table,save_dir):
    """
    Inputs
    -------
    data : 2-D array 
        A table with Time, X position (Must have column name of "Pos. X (mm)"), Y position (Must have column name of "Pos. X (mm)") information included
    f_name : string
        Name of file in process
    gate_range : [x range, y range]
        Specifying the normalized range along horizontal or vertical direction in which border crossing event could be considered valid
    t_delay : float/integer
        Delay of trial initiation in seconds, used for realigning time
    f_rate : integer
        Frame rate of imaging & video capturing (supposed to be synchronized)
    crop_length : float/integer
        Total length of time included for analysis in seconds, data after specified time point will be cropped off
    pref_side : "R" or "L"
        Specifying the side of conditioning, if preferred side is left, x coordinates will be reversed in sign before annotating exploring side
    plot_raw : Boolean
        Whether or not to show the graphs demonstrating preprossesing outcomes, including correction of missing points, specifying exploring chamber, as well as labeling of border crossing events.
    save_plot : Boolean
        Whether or not the plotted graphs should be saved
    save_table : Boolean
        Whether or not to export the trajectory processing output into a .csv file (With corrected time, X/Y position, exploring side, distance to center, crossing timing of two types of events)
    save_dir : r-string
        The directory in which the output will be saved

    Returns
    -------
    Plotting

    output_dat : dictionary
        Trajectory processing output (With corrected time, X/Y position, exploring side, distance to center, crossing timing of two types of events)

    Notes
    ------
    Called internally by function "traj_trace_pair_process"
    """
    ####
    #Add alternative input file types (ezTrack or Toxtrac)
    if '_LocationOutput' in f_name: #ezTrack file export format
        time = np.array(data[1:, np.where(data[:,0] == 'Frame')[0][0]], dtype = 'float') - t_delay / f_rate 
        crop_p = crop_length * f_rate
        #Extract X Y time series, normalized to the range of -1,1, and remove data out of range
        X = (norm(np.array(data[1:,np.where(data[0,:] == 'X')[0]],dtype = 'float').flatten())-.5)[:crop_p]*2 
        Y = (norm(np.array(data[1:,np.where(data[0,:] == 'Y')[0]],dtype = 'float').flatten())-.5)[:crop_p]*2 
        time = time[:crop_p]
    else: #toxtrac 
        time = np.array(data[1:,0],dtype = 'float')-t_delay #Realign time 
        crop_p = np.where(time < crop_length)[0].max() 
        #Extract X Y time series, normalized to the range of -1,1, and remove data out of range
        X = (norm(np.array(data[1:,np.where(data[0,:] == 'Pos. X (mm)')[0]],dtype = 'float').flatten())-.5)[:crop_p]*2 
        Y = (norm(np.array(data[1:,np.where(data[0,:] == 'Pos. Y (mm)')[0]],dtype = 'float').flatten())-.5)[:crop_p]*2
        time = time[:crop_p]

    #Find out discontinuous time points 
    dt = min(time[1:]-time[:-1])
    dis_indx = []
    for j,t in enumerate(time[1:]):
        if t-time[j]> 2*dt: #Any two data point with interval more than 2 times of the average interval will be considered as a gap and annotated
            dis_indx.append(j)
            
    if plot_raw:
    #Plot raw trajectories
        f,a = plt.subplots(1,3,figsize = (20,3))
        for j , p in enumerate([X,Y]):
            a[j].plot(time,p,'-k',alpha = 0.6)
            a[j].hlines(0,0,max(time),color = 'red',linestyles = 'dotted')
            [a[j].fill_between([time[k],time[k+1]],-1,1,color = 'pink', alpha = 0.5) for k in dis_indx]
            a[j].fill_between([0],0,0,color = 'pink', alpha = 0.5,label = 'Not tracked') #Specifying not tracked timing 
            a[j].set(ylabel = f"Normalized {['X','Y'][j]} position",
                    xlabel = "Time (s)")
            a[j].legend(loc = 3)

        a[2].plot(X,Y,color = 'black', alpha = 0.6)
        a[2].set(xlabel='X', ylabel = 'Y',xticks = [], yticks = [])
        a[2].vlines(0,-.5,.5,color = 'red')
        a[2].spines['right'].set_visible(True)
        a[2].spines['top'].set_visible(True)

        f.suptitle(f'{f_name.split("cpp.txt")[0]}')
        os.chdir(save_dir)
        [f.savefig(f'{f_name.split("cpp.txt")[0]} raw normalized trajectory.png', dpi = 300) if save_plot else None]

    time_new , X_new, Y_new = np.copy(time), np.copy(X), np.copy(Y)
    inj_accum = 0
    for indx in dis_indx: #Run through all recorded gaps' start timing
        #Fill up time gaps with continuous linear sequence
        time_new = np.insert(time_new,indx+inj_accum+1,add:=np.arange(time[indx],time[indx+1],dt)[:])
        #Fill up missing points with dicernable gap value (Nan)
        X_new = np.insert(X_new,indx+inj_accum+1,np.repeat(np.nan,len(add)))
        Y_new = np.insert(Y_new,indx+inj_accum+1,np.repeat(np.nan,len(add)))
        inj_accum += len(add)
        
    #detect chamber transition 
    # Acceptable y ,x range
    y_in_range = np.abs(Y_new) < gate_range[1]
    x_in_range = np.abs(X_new) < gate_range[0]
    both_match = np.logical_and(x_in_range, y_in_range) #Timing in which animal locates in the border range
    #Potential border crossing event
    #reverse crossing definition base on preferred side 
    pref_factor = (int(pref_side == 'R')-0.5)*2 #Right: 1, Left: -1
    X_new *= pref_factor
    #Whether the animal position is in the right chamber (1) or left chamber (0)
    x_side = np.array([np.nan if np.isnan(x) else 1 if x > 0 else 0 for x in X_new])
    #Reverse the 1, -1 correspondence based on preferred chamber side 
    episode = (x_side[1:] - x_side[:-1])
    cond_enter = np.where(np.logical_and(episode == 1 ,both_match[:-1]))[0]
    uncond_enter = np.where(np.logical_and(episode == -1,both_match[:-1]))[0]
    #Calibrate the distance of animal to the center of exploring space (in arbitrary units)
    dist = [euclidian_distance([x,y],[0,0]) for x,y in zip(X_new,Y_new)]
    
    if plot_raw:
        plt.figure(figsize = (15,5))
        spec = gridspec.GridSpec(ncols=4, nrows=2, width_ratios=[2,2,1,1])
        # Top left panel, plot out X coordinates with exploring side, corssing events labeled
        ax = plt.subplot(spec[0,:2])
        ax.plot(time_new,X_new,'-k')
        ax.set(xlim=(0,max(time_new)),
                     xlabel= 'Time (s)',
                     ylabel='Normaized X position')

        ax.vlines(time_new[cond_enter]-1,1,1.3,alpha = 1,color = 'black',label = 'Cond. enter') #Enter conditioned chamber 
        ax.vlines(time_new[uncond_enter]-1,1.3,1.6,alpha = 1,color = 'red', label = 'Cond. exit') #Enter unconditioned chamber
        ax.vlines(time_new[np.where(X_new > 0)[0]],1,-1,alpha = 0.04,color = 'cyan') #Animal in right
        ax.vlines(time_new[np.where(X_new < 0)[0]],1,-1,alpha = 0.04,color = 'pink') #Animal in left
        ax.fill_between([0],0,0,color = 'cyan',label = 'Right chamber')
        ax.fill_between([0],0,0,color = 'pink',label = 'Left chamber')
        ax.hlines(0,0,max(time_new),color = 'black',linestyles='dashed')
        ax.legend(loc = 3)
        # Left panel, plot out the 2D trajectory with positions of crossing event occurrences labeled
        ax1 = plt.subplot(spec[:,-2:])
        #Trajectory
        ax1.plot(X_new,Y_new,alpha = 0.5)
        ax1.set(ylabel= 'Normaized Y position', xlabel ='Normaized X position')  
        #Crossing events 
        ax1.scatter(X_new[cond_enter],Y_new[cond_enter],c = 'red',label='Cond. enter')
        ax1.scatter(X_new[uncond_enter],Y_new[uncond_enter],c = 'green',label = 'Cond. exit')
        ax1.legend(loc=2)
        ax1.spines['right'].set_visible(True)
        ax1.spines['top'].set_visible(True)

        #Left bottom panel, Distance to center
        ax2 = plt.subplot(spec[1,:2])
        ax2.plot(time_new, dist)
        ax2.set(xlim =(0,max(time_new)),xlabel='Time (s)',ylabel='Distance to center')


        plt.suptitle(f'{f_name.split("cpp.txt")[0]}\nConditioned side: {pref_side}')
        plt.tight_layout()
        os.chdir(save_dir)
        [plt.savefig(f'{f_name.split("cpp.txt")[0]} Labeled trajectory.png', dpi = 300) if save_plot else None]
    
    #Prepare for data saving 
    output_dat = {
    "Time (s)": time_new,
    "X position":[np.concatenate(X_new) if X_new.ndim != 1 else X_new][0],
    "Y position": [np.concatenate(Y_new) if Y_new.ndim != 1 else Y_new][0],
    "Side (U:-1 C:1)":  [np.concatenate((x_side-0.5)*2)*pref_factor if x_side.ndim != 1 else (x_side-0.5)*2*pref_factor][0],
    "Distance to center": np.array(dist),
    "Cond. exit Event timing (s)": np.array(uncond_enter),
    "Cond. enter Event timing (s)": np.array(cond_enter)
    }

    if save_table:
        os.chdir(save_dir)
        output_df = output_dat.copy()
        #Fill up empty space for creating data frame
        output_df["Cond. exit Event timing (s)"] = np.insert(np.array(time_new[uncond_enter],dtype = 'str'),len(uncond_enter),np.repeat(' ',len(time_new)- len(uncond_enter)))
        output_df["Cond. enter Event timing (s)"] =  np.insert(np.array(time_new[cond_enter],dtype = 'str'),len(cond_enter),np.repeat(' ',len(time_new)- len(cond_enter)))

        output_df = pd.DataFrame(output_df)
        output_df.to_csv(f'{f_name.split("cpp.txt")[0]} output tables - {pref_side} formalin.csv')
        
    return(output_dat)

def traj_trace_pair_process(calciumpath, trajpath, tracking_source, formalin_side, t_pre, t_post, sprt, thr, \
                            start_delay, Total_duration, cross_boundary, \
                            exclude_animal, raw_plot, save_raw_plot, save_table, \
                            save_dir, dictionary_name):
    """
    Inputs
    -------
    calciumpath : r-string
        Directory at which "preprocessed" calcium signals are saved (Shall be .csv file)
    trajpath : r-string
        Directory at which tracked trajectory are saved (in .csv or .txt format)
    tracking source : 'Toxtrac' or 'ezTrack'
        Specify the software used for tracking
    formalin_side : dictionary
        Specifying the formalin conditioned side for every animal ('R' or 'L')
    t_pre : float
        Length of time to include before chamber crossing event
    t_post : float
        Length of time to include after chamber crossing event
    sprt : integer
        Sampling rate or frame rate of calcium imaging or behavioral video capturing (Hz)
    thr : float
        Minimum mean difference in activity after crossing event to identify cross-repsonsive cells
    start_delay : float/integer
        Delay of trial initiation in seconds, used for realigning time
    Total_duration : float/integer
        Total length of time included for analysis in seconds, data after specified time point will be cropped off
    cross_boundary : [x range, y range]
        Specifying the normalized range along horizontal or vertical direction in which border crossing event could be considered valid
    exclude_animal : list
        List of animal IDs to be excluded from analysis
    raw_plot : Boolean
        Whether or not to show the graphs demonstrating preprossesing outcomes, including correction of missing points, specifying exploring chamber, as well as labeling of border crossing events.
    save_plot : Boolean
        Whether or not the plotted graphs should be saved
    save_table : Boolean
        Whether or not to export the "trajectory processing" output into a .csv file (With corrected time, X/Y position, exploring side, distance to center, crossing timing of two types of events)
    save_dir : r-string
        The directory in which the output will be saved
    dictionary_name : string
        The name for the exported dictionary (.pkl file) with CPP metadata 

    Returns
    -------
    Plotting

    Export Dictionary with CPP metadata
    """
    #Data preperation 
    #Calcium single
    os.chdir(calciumpath)#Move to directory with calcium traces
    file_list_Ca = [_ for _ in os.listdir(calciumpath) if _.endswith('.csv')]

    data_Ca = pd.read_csv(file_list_Ca[0],header = None, low_memory = False).values

    cell_ID = data_Ca[np.where(data_Ca[:,0] == 'Cell ID')[0].min(),1:]
    file_name = data_Ca[np.where(data_Ca[:,0] == 'File name')[0],1:][0]
    #Determine mouse ID & chornic pain stage from file name
    state = np.array([f_n.split(' ')[-1].removesuffix('.csv') for f_n in file_name])
    mice =  np.array([file_name[i].split(' ')[1] for i in range(len(file_name))],dtype = 'str')

    signal_start_index = np.where(data_Ca[:,0] == 'Time (s)')[0][0]+1
    Ca_time = np.array(data_Ca[signal_start_index:,0],dtype = 'float')
    signal =  np.array(data_Ca[signal_start_index:,1:],dtype = 'float')

    trace_meta = {
        'Cell ID':[],
        'State':[],
        'Mouse':[],
        'Cross type':[],
        'Trace time':[],
        'Extracted trace mean':[],
        'Extracted trace sem':[],
        'Mean response': [],
        'Response type': [],
        'Cham':[],
        'Cham mean':[]
    }

    #Trajectory processing
    if tracking_source == 'Toxtrac':
        file_list_traj = [_ for _ in os.listdir(trajpath) if _.endswith('.txt')]
        
    elif tracking_source == 'ezTrack':
        file_list_traj = [_ for _ in os.listdir(trajpath) if _.endswith('.csv')]

    for i, traj_file in enumerate(np.sort(file_list_traj)):
        os.chdir(trajpath)#move to diractory with tracked trajectory
        print(f"File in process: {traj_file}")

        if tracking_source == 'Toxtrac':
            data = []
            #read in txt file 
            with open(traj_file, newline = '') as f:                                                                                          
                f_reader = csv.reader(f, delimiter='\t')
                for l in f_reader:
                    data.append(l)
            data = np.array(data)
            traj_state = traj_file.split("cpp.txt")[0].split(' ')[-2]
            traj_mice = traj_file.split("cpp.txt")[0].split(' ')[0]

        elif tracking_source == 'ezTrack':
            #read in csv file
            data = pd.read_csv(traj_file,header = None, low_memory = False).values
            traj_state = traj_file.split("cpp_LocationOutput.csv")[0].split(' ')[-2]
            traj_mice = traj_file.split("cpp_LocationOutput.csv")[0].split(' ')[0]
        
        #Trajectory processing & crossing events searching
        if traj_mice in exclude_animal:
            print('Animal excluded')
            continue # skip this file

        traj_metadata = traj_analyze_CPP(data,traj_file,cross_boundary,start_delay,sprt,
                                         Total_duration,formalin_side[traj_mice],raw_plot,
                                         save_raw_plot, save_table, save_dir)
        
        time, X,Y , cond_ent, cond_ext , dist_border, side = traj_metadata['Time (s)'],traj_metadata['X position'],traj_metadata['Y position'],\
        traj_metadata['Cond. enter Event timing (s)'],traj_metadata['Cond. exit Event timing (s)'], traj_metadata['Distance to center'],  traj_metadata['Side (U:-1 C:1)']

        match_cell = np.where(np.logical_and(mice == traj_mice, state == traj_state))[0]
    #-----------------------------------------------------
        #Determine indexes of cells from the same trial & animal
        # match_cell = np.where(np.logical_and(mice == traj_mice, state == f_state_match[traj_state]))[0]
        
        #Create common time scale, stay unchanged after initial definition 
        com_t = [time[start_end_index(time, t_pre,t_pre,t_post)[0]:start_end_index(time, t_pre,t_pre,t_post)[1]]-t_pre\
                if i == 0 else com_t][0]
        trace_meta['Trace time'] = com_t
    #-----------------------------------------------------
        for cell in match_cell:
            trace = resample(signal[:,cell],len(time)) #Resample to the same length
            #Crossing events 
            cross_type = np.concatenate([np.repeat('ENT',len(cond_ent)),np.repeat('EXT',len(cond_ext))])
            cross_trace,rm_indx = [],[]
            for k,t in enumerate(np.concatenate([time[cond_ent], time[cond_ext]])):
                if t <= Ca_time.max() - t_post and t >= t_pre: #Exclude events with insufficient time window for cropping
                    cross_timing = start_end_index(time, t,t_pre,t_post) # 10 s before to 10 s after event
                    cross_trace.append(resample(trace[cross_timing[0]:cross_timing[1]], len(com_t))) #Resample for matching signal length 
                else:
                    rm_indx.append(k) #Recording indexes of removal 
            cross_trace = np.array(cross_trace)
            #***** Verification required *****
            cross_type = [np.delete(cross_type,np.array(rm_indx)) if len(rm_indx)> 0 else cross_type][0]#Update list of event type - removing recorded indexes 
            #Standardization by pre_event baseline 
            trace_Z = np.array([standardize_extraction(cross_trace[c,:],[0,sprt*t_pre])[0] for c in range(cross_trace.shape[0])])        
            ave_diff = np.mean(trace_Z[:,sprt*t_pre:],axis = 1) - np.mean(trace_Z[:,:sprt*t_pre],axis = 1) #Response - baseline in Z-score

            mean_res = np.array([ave_diff[np.where(cross_type == dir)[0]].mean() for dir in ['ENT','EXT']])
            res_type = [res_det(r,thr) for r in mean_res] #Excited, inhibited or no response
    #-----------------------------------------------------
            trace_meta['Cell ID'].append(np.repeat(cell_ID[cell],2))
            trace_meta['State'].append(np.repeat(traj_state,2))
            trace_meta['Mouse'].append(np.repeat(traj_mice,2))
            trace_meta['Cross type'].append(['ENT','EXT'])
            trace_meta['Extracted trace mean'].append(np.array([trace_Z[np.where(cross_type == cross)[0],:].mean(axis = 0) for cross in ['ENT','EXT']]))
            trace_meta['Extracted trace sem'].append(np.array([trace_Z[np.where(cross_type == cross)[0],:].std(axis = 0)/np.sqrt(len(match_cell)) for cross in ['ENT','EXT']]))
            trace_meta['Mean response'].append(mean_res)
            trace_meta['Response type'].append(res_type)
            trace_meta['Cham'].append(['Cond','Uncond']) #side
            trace_meta['Cham mean'].append([trace[np.where(X>0)[0]].mean(),trace[np.where(X<0)[0]].mean()]) #Chamber-averaged value

            #  Plotting out averaged traces of responsive cells
            if (np.abs(mean_res) > thr).sum() >= 1 and raw_plot: #Either crossing event shows large enough response, and plotting is allowed
                fig,ax = plt.subplots(1,2)
                for t,cross in enumerate(['ENT','EXT']):
                    mt_trace = trace_Z[np.where(cross_type == cross)[0],:]
                    ax[t].plot(com_t,ave:=np.mean(mt_trace ,axis = 0))
                    ax[t].fill_between(com_t,\
                                    ave+np.std(mt_trace,axis = 0)/np.sqrt(len(match_cell)),\
                                    ave-np.std(mt_trace,axis = 0)/np.sqrt(len(match_cell)),\
                                    alpha = 0.3)

                    ax[t].hlines([0],-t_pre,t_post,linestyle = 'dashed', color = 'black')
                    ax[t].hlines([-thr,thr],-t_pre,t_post,linestyle = 'dashed', color = 'silver')
                    ax[t].vlines([0],trace_Z.min(),trace_Z.max(),linestyle = 'dashed', color = 'red')
                    ax[t].set(xlim = (-t_pre,t_post),
                            ylim = (trace_Z.min(),trace_Z.max()), #Restrain plot scale
                        xlabel = 'Time from crossing',
                        ylabel = 'Z-score',
                        title = f"Conditioned chamber \n{['enter','exit'][t]}")
        #                 ax[t].hist(ave_diff[np.where(cross_type == cross)[0]],orientation="horizontal")
                fig.suptitle(f'{mice[cell]} - {state[cell]} - {cell_ID[cell]}',weight = 'bold')
                fig.tight_layout()

    #Data format convertion for saving
    for key,dat in trace_meta.items():
        trace_meta[key] = [np.concatenate(dat) if key != 'Trace time' else dat][0]  #transform into array

    #Save processed results as a dictionary 
    os.chdir(save_dir)#Save dictionary at desired directory 
    with open(f'{dictionary_name }.pkl', 'wb') as f:
        pickle.dump(trace_meta, f)

#Functions for plotting
################################################################
def shade_trace_plot(trajpath,formalin_side, file_list_traj, Ca_time, \
                     signal, mice, state, start_delay, Total_duration, cross_boundary, \
                        shade_color,raw_plot, save_raw_plot, save_table, save_dir):
    '''
    Inputs
    -------
    trajpath : r-string
        Directory with trajectory files
    formalin_side : dictionary
        Specifying the formalin conditioned side for every animal ('R' or 'L')
    file_list_traj : list or 1-D array
        List of trajectory files to process
    Ca_time : 1-D array
        Time series of Ca signal time 
    signal : 2-D array
        CPP calcium traces of "all neurons" of "all animals" from "all trials"
    mice : 1-D array
        Corresponding mouse ID of every calcium trace in `signal`
    state : 1-D array
        Corresponding CP state of every calcium trace in `signal`
    start_delay : float/integer
        Delay of trial initiation in seconds, used for realigning time
    Total_duration : float/integer
        Total length of time included for analysis in seconds, data after specified time point will be cropped off
    cross_boundary : [x range, y range]
        Specifying the normalized range along horizontal or vertical direction in which border crossing event could be considered valid
    shade_color : [color for uncnditioned side, color for conditioned side]
    raw_plot : Boolean
        Whether or not to show the graphs demonstrating preprossesing outcomes, including correction of missing points, specifying exploring chamber, as well as labeling of border crossing events.

    Returns
    -------
    plotting 
    '''
    # Determine trajectory file format 
    toxtrac = file_list_traj[0].endswith('.txt')

    for traj_file in np.sort(file_list_traj):
        os.chdir(trajpath)
        #Read in trajectory file & determine trial information (mouse ID, CP state etc.)
        if toxtrac:
            data = []
            #read in txt file 
            with open(traj_file, newline = '') as f:                                                                                          
                f_reader = csv.reader(f, delimiter='\t')
                for l in f_reader:
                    data.append(l)
            data = np.array(data)
            traj_state = traj_file.split("cpp.txt")[0].split(' ')[-2]
            traj_mice = traj_file.split("cpp.txt")[0].split(' ')[0]
        else: #File exported from ezTrack
            #read in csv file
            data = pd.read_csv(traj_file,header = None, low_memory = False).values
            traj_state = traj_file.split("cpp_LocationOutput.csv")[0].split(' ')[-2]
            traj_mice = traj_file.split("cpp_LocationOutput.csv")[0].split(' ')[0]

        #Generate trajectory / calcium trace paired dataset 
        traj_metadata = traj_analyze_CPP(data,traj_file,cross_boundary,start_delay,20,\
                                         Total_duration,formalin_side[traj_mice],\
                                            raw_plot,save_raw_plot,save_table,save_dir)
        time, side = traj_metadata['Time (s)'], traj_metadata['Side (U:-1 C:1)'] # Extract time & exploring position 

        match_cell = np.where(np.logical_and(mice == traj_mice, state == traj_state))[0] # Index of cell in current trial 
        #Plot out marked traces
        fig,ax = plt.subplots(1,1,figsize = (10,len(match_cell)*0.2))

        #Mark exploring side of every time point with color
        [ax.vlines(time[np.where(side == s)[0]] ,0 ,len(match_cell), alpha = 0.02, color = shade_color[k]) for k,s in enumerate([-1,1])] 
        #Plot out calcium traces
        [ax.plot(Ca_time,signal[:,match_cell[n]]+1*n,'-k') for n in range(len(match_cell))]
        ax.set(title = f'{traj_file.split("cpp.txt")[0]} - {formalin_side[traj_mice]}-conditioned',
            xlim = (0,600),
            ylim = (None,len(match_cell)),
            ylabel = 'Cell #',
            xlabel = 'Time (s)',
            yticks =[0,len(match_cell)],
            yticklabels = np.array([0,len(match_cell)]))
        ax.fill_between([0],0,0,color = 'red',label = 'Conditioned chamber')
        ax.fill_between([0],0,0,color = 'lime',label = 'Unconditioned chamber')
        ax.legend(loc = 3)

        os.chdir(save_dir)
        fig.savefig(f'{traj_file.split("cpp.txt")[0]} marked traces.png', dpi = 600)

def triplot(x,y,cutoff,text,prefix,filename, group_col,histogram_color, delt_type, save_plot,save_dir):
    '''
    Inputs
    -------
    x : 1-D array
        Variable #1
    y : 1-D array
        Variable #2
    cutoff : Float
        Threshold for identifying responsive neurons 
    text : [string, string]
        Text label for x and y
    prefix : string
        Prefix of figure title
    filename : string
        Name of the exported figure
    group_col : list with lenght of 3
        Color (in pie chart) denoting x-pref, y-pref, or no pref cells
    histogram_color : string
        Filled color for histogram
    delt_type : 'diff' or 'pref'
        way of determining cell classification index, this could be either 'diff' or 'pref'
    save_plot : Boolean
        Whether or not the plotted graphs should be saved
    save_dir : r-string
        The directory in which the output will be saved
    Returns
    -------
    plotting 
    cell_res : 1-D array
        Response of every songle cell
    '''
    cell_res = []
    c_list = np.array([-cutoff,0,cutoff]) 
    lx = np.linspace(0,top := max([np.max(x),np.max(y)]),1000) 
    #Calculate the y value 
    ly = lambda x, c: x+c

    f = plt.figure()
    f, (a, b, c) = plt.subplots(nrows=1, ncols=3,constrained_layout=True,figsize=(15,5))
    delt_pool = []
    for i, (pre,post) in enumerate(zip(x, y)):
        # calibrate delta 
        delta = [(pre-post) if delt_type == 'diff' \
                 else (pre-post)/(pre+post) if delt_type  == 'pref' \
                    else None][0]
        delt_pool.append(delta)

        #Plot out data point with color label
        a.scatter(x[i],y[i],s = 10,c= group_col[[0 if delta >= cutoff \
                                                 else 1 if delta < -cutoff \
                                                    else 2 if abs(delta) < cutoff else None][0]])
        cell_res.append(['INH' if delta>= cutoff \
                         else 'EXT' if delta <= -cutoff \
                            else 'NS' if abs(delta) < cutoff else None][0])
        
    a.plot(lx,ly(lx,c_list[1]),color = 'blue',alpha = 0.5) # y = x

    if delt_type == 'diff':
        [a.plot(lx,ly(lx,c),'--',color = 'blue', alpha = 0.2) if j !=1 \
         else None for j,c in enumerate(c_list)] #Plot out decision boundary
    a.axis('equal')
    a.set(xlim=(0,top+0.01),
          ylim=(0,top+0.01),
          xlabel = f"{text[0]} response",
          ylabel = f"{text[1]} response")
    
    [inh, ext, ns] = [sum(np.array([i == re for i in cell_res])) for re in ['INH','EXT','NS']]

    # Pie chart of responsive cells 
    labels = [f'{text[0]} pref.', f'{text[1]} pref.', 'No pref.']
    sizes = [inh,ext,ns]
    piecolors = group_col
    b.pie(sizes, colors = piecolors, labels=labels, autopct='%1.1f%%',\
          wedgeprops={'width':0.6,'alpha' : 0.5,'linewidth':3,'edgecolor':'w'},
        shadow=False, startangle=90)
    b.text(0, 0, f'N = {len(cell_res)}', ha='center', va='center', fontsize=10)
    b.legend(loc = 'lower center',frameon = False,ncol = 3,bbox_to_anchor = (0.5,-.05))

    #Histogram of diff index (difference or preferecne score)
    hist_bar = c.hist(delt_pool, 20, color = histogram_color,density = False)
    c.set(ylabel = 'Cell count',
          xlabel = f'{["Preference score" if delt_type == "pref" else "Inter-side difference"][0]}',
          xlim = [(-1,1) if delt_type == "pref" else (-.5,.5)][0])
    [c.set_xticks([-1,0,1],[f'-1\nUncond. only', '0\nBoth', f'1\nCond. only']) if delt_type == "pref" else None]
    c.vlines(np.nanmean(delt_pool),0,hist_bar[0].max(),color= 'red', linestyles = 'dashed',label = 'Pop Mean')
    c.legend(frameon = False)

    # b.set_title()
    f.suptitle(f'{prefix} Unit classification ({["Preference score" if delt_type == "pref" else "Inter-side difference"][0]} cutoff: $\pm{round(cutoff,3)}$)',weight='bold')
    # f.subplots_adjust(wspace=0.2)
    os.chdir(save_dir)
    [f.savefig(f"{filename}.png", dpi = 600, bbox_inches = 'tight') if save_plot else None]

    return np.array(cell_res)

def pool_response_class_pie(trace_meta, state_list, pie_color_rough, pie_color_detail, plot_name, save_plot,save_dir):
    '''
    Inputs
    -------
    trace_meta : dictionary
        Metadata for trajectory analysis
    state_list : 1-D array
        List of CP states included in analysis
    pie_color_rough : list
        Color for [Both responsive, Cond. enter only, Con.d exit only, None responsive] 
    pie_color_detail : list with lenght of 8
        Color for detailed response combination
    plot_name : string
        Name of the exported figure
    save_plot : Boolean
        Whether or not the plotted graphs should be saved
    save_dir : r-string
        The directory in which the output will be saved

    Returns
    -------
    plotting 
    '''
    # Plot- donut chart for pooled results
    fig, ax = plt.subplots(ncols = len(state_list),nrows = 2,figsize = (15,10))
    for i, s in enumerate(state_list):
        #index of enter response & ext response
        [ent_indx , ext_indx] = [np.where(np.logical_and(trace_meta['State'] == s, trace_meta['Cross type'] == cr))[0] \
            for cr in np.sort(np.unique(trace_meta['Cross type']))] #find cell responses with the same cross event, animal, and trial

        cross_res, res_match = [], []
        for ra , rb in itertools.product(['EX','INH','NS'], repeat  = 2): 
            cross_res.append(f"in_{ra}_out_{rb}") 
            res_match.append(np.where(np.logical_and(trace_meta['Response type'][ent_indx] == ra,\
                trace_meta['Response type'][ext_indx] == rb))[0])
        #Remove response combination with no matched cells
        keep_indx = np.where(np.logical_xor(np.zeros(len(cross_res)) == 0,  np.array([r.size for r in res_match]) == 0))[0] 
        #In accordance, determine the indexes of matched cells
        res_match_del = np.array(res_match,dtype = 'object')[keep_indx] 
        cross_res_del = np.array(cross_res)[keep_indx]
        # response type conversion : EXT, INH = True, NS = False
        res_rend = np.array([[re.split('_')[1] != 'NS', re.split('_')[3] != 'NS']\
            for re in cross_res_del])#Responding or not
        # Count the number of cells with certain response combinations
        res_count = [np.concatenate(res_match_del[np.logical_and(res_rend[:,0] == k[0], res_rend[:,1]== k[1])]).size \
                    for k in itertools.product([True,False],repeat = 2)]
        
        # res_count = [np.concatenate(res_match_del[np.logical_and(res_rend[:,0] == k[0], res_rend[:,1]== k[1])]).size \
        #             for k in itertools.product([True,False],repeat = 2)]
        # res_count = [np.logical_and(res_rend[:,0] == k[0], res_rend[:,1]== k[1]).sum() \
        #             for k in itertools.product([True,False],repeat = 2)] 
        
        #plotting - pie (donut) charts
        ax[0,i].pie(res_count, labels = res_count, colors = pie_color_rough, 
            wedgeprops={'width':0.6,'alpha' : 0.5,'linewidth':3,'edgecolor':'w'}, 
            shadow=False, startangle=90)
        ax[0,i].text(0, 0, f'N = {len(ent_indx)}', ha='center', va='center', fontsize=10)
        ax[0,i].legend([f"{r}: {round(cnt/len(ent_indx)*100,1)}%" for r,cnt in zip(['Both','Enter only','Exit only','None'],res_count)],
                            bbox_to_anchor=(0.9,0), # Legend position 
                            loc='upper right', 
                            ncol = 2,
                            frameon = False, 
                            fancybox=True) 
        ax[0,i].set_title(f"{s} - Responsiveness")
        #Quantifying detailed response types 
        ax[1,i].pie(frac:=[len(r) for k,r in enumerate(res_match_del) if (res_rend[k] == [False,False]).sum() <2], 
            labels = frac, colors = pie_color_detail[keep_indx[:-1]],
            wedgeprops={'width':0.6,'alpha' : 0.8,'linewidth':3,'edgecolor':'w'}, 
            shadow=False, startangle=90)
        ax[1,i].text(0, 0, f'N = {len(ent_indx) - res_count[-1]}', ha='center', va='center', fontsize=10)
        ax[1,i].set_title(f"{s} - detailed response")
        ax[1,i].legend([f"Exit {r.split('_')[3].lower()}-Enter {r.split('_')[1].lower()}: {round(cnt/len(ent_indx)*100,1)}%" for r,cnt in zip(cross_res_del[:-1],frac)],
                        bbox_to_anchor=(1,0), # Legend position 
                        loc='upper right', 
                        ncol = 2,
                        frameon = False, 
                        fancybox=True) 
    fig.suptitle(plot_name,weight= 'bold')
    fig.tight_layout()
    os.chdir(save_dir)
    [fig.savefig(f'{plot_name}.png', dpi = 600, bbox_inches = 'tight') if save_plot else None]

def individual_response_class_pie(trace_meta, mice, state_list, pie_color_rough, plot_name, save_plot,save_dir):
    '''
    Inputs
    -------
    trace_meta : dictionary
        Metadata for trajectory analysis
    state_list : 1-D array
        List of CP states included in analysis
    pie_color_rough : list
        Color for [Both responsive, Cond. enter only, Con.d exit only, None responsive] 
    plot_name : string
        Name of the exported figure
    save_plot : Boolean
        Whether or not the plotted graphs should be saved
    save_dir : r-string
        The directory in which the output will be saved

    Returns
    -------
    plotting 
    '''
    # Plot- donut chart for individual results
    for m in np.unique(mice): #Run through individual mouse
        fig, ax = plt.subplots(ncols = len(state_list),nrows = 1,figsize = (15,5))
        for i, s in enumerate(state_list): 
            [ent_indx , ext_indx] = [np.where(np.logical_and(trace_meta['State'] == s, np.logical_and(trace_meta['Cross type'] == cr, trace_meta['Mouse'] == m)))[0] \
                for cr in np.sort(np.unique(trace_meta['Cross type']))]
            if len(ent_indx) == 0: #No matched cells in certain stage - create an empty plot 
                ax[i].pie([1,1,1],colors=['white','white','white'])
                ax[i].set_title(f"{s} - Responsiveness")
                next
            else:
                cross_res, res_match = [], []
                for ra, rb  in itertools.product(['EX','INH','NS'], repeat  = 2):
                    cross_res.append(f"R_{ra}_L_{rb}") 
                    res_match.append(np.where(np.logical_and(trace_meta['Response type'][ent_indx] == ra,\
                        trace_meta['Response type'][ext_indx] == rb))[0])
                
                keep_indx = np.where(np.logical_xor(np.zeros(len(cross_res)) == 0,  np.array([r.size for r in res_match]) == 0))[0] #Remove response combination with no matched cells
                res_match_del = np.array(res_match,dtype = 'object')[keep_indx] #indexes of matched cells
                cross_res_del = np.array(cross_res)[keep_indx]

                #Quantifying cell counts responding to either crossing event 
                # response type conversion
                res_rend = np.array([[re.split('_')[1] != 'NS', re.split('_')[3] != 'NS']\
                    for re in cross_res_del])
                keep = [np.logical_and(res_rend[:,0] == k[0], res_rend[:,1]== k[1]).sum()>0 for k in itertools.product([True,False],repeat = 2)]
                res_count = [np.concatenate(res_match_del[np.logical_and(res_rend[:,0] == k[0], res_rend[:,1]== k[1])]).size \
                            if keep[j] else 0 for j, k in enumerate(itertools.product([True,False],repeat = 2))]
                #plotting - pie (donut) charts
                ax[i].pie(res_count, 
                    labels = res_count, 
                    colors = pie_color_rough, 
                    wedgeprops={'width':0.6,'alpha' : 0.5,'linewidth':3,'edgecolor':'w'}, 
                    shadow=False, 
                    startangle=90)
                ax[i].text(0, 0, 
                    f'N = {len(ent_indx)}', 
                    ha='center', va='center', fontsize=10)
                ax[i].legend([f"{r}: {round(cnt/len(ent_indx)*100,1)}%" for r,cnt in zip(['Both','Enter only','Exit only','None'],res_count)],
                                    bbox_to_anchor=(0.9,0), # Legend position 
                                    loc='upper right', 
                                    ncol = 2,
                                    frameon = False, 
                                    fancybox=True) 
                ax[i].set_title(f"{s} - Responsiveness")
            fig.suptitle(f'{m} - {plot_name}',weight= 'bold')
            fig.tight_layout()
            os.chdir(save_dir)
            [fig.savefig(f'{m} - {plot_name}.png', dpi = 600, bbox_inches = 'tight') if save_plot else None]

def heatmap_trace_plot(trace_meta, state_list,thr,heatmap_save_name, mean_trace_save_name, save_plot, save_dir): 
    '''
    Inputs
    -------
    trace_meta : dictionary
        Metadata for trajectory analysis
    state_list : 1-D array
        List of CP states included in analysis
    thr : float
        Threshold of Z-score activity for neuron to be classfied as crossing-responsiv 
    heatmap_save_name : string
        Name of the exported heatmap
    mean_trace_save_name : string 
        Name of the exported "figure" of averaged traces
    save_plot : Boolean
        Whether or not the plotted graphs should be saved
    save_dir : r-string
        The directory in which the output will be saved

    Returns
    -------
    plotting 
    '''
    for s in state_list: #Run thorugh CP states
        #find cell responses with the same cross event, animal, and trial
        [ent_indx , ext_indx] = [np.where(np.logical_and(trace_meta['State'] == s, trace_meta['Cross type'] == cr))[0] for cr in np.sort(np.unique(trace_meta['Cross type']))] 
        cross_res, res_match = [], []
        #Cell traces for certain response combination 
        match_trace_ext, match_trace_ent = [], [] 

        #prepare cell traces 
        for ra , rb in itertools.product(['EX','INH','NS'], repeat  = 2): 
            #Find cell index with matched response type 
            match = np.where(np.logical_and(trace_meta['Response type'][ent_indx] == ra,trace_meta['Response type'][ext_indx] == rb))[0]  
            if len(match) > 0: #Exclude response combination with no matched cell
                    #Document response combination 
                    cross_res.append(f"in_{ra}_out_{rb}") 
                    res_match.append(match) 
                    #Include matched cell traces
                    match_trace_ent.append(trace_meta['Extracted trace mean'][ent_indx][match])
                    match_trace_ext.append(trace_meta['Extracted trace mean'][ext_indx][match])
        #plotting 
        time = trace_meta['Trace time']
        #heatmap 
        fig  = plt.figure(figsize = (10,2*len(cross_res)))
        gs = gridspec.GridSpec(len(cross_res),2,height_ratios=[len(l) for l in res_match],wspace=0.1, hspace=0.2)
        for j, comb in enumerate(cross_res):
            for n in range(np.unique(trace_meta['Cross type']).size):
                heatmap_trace = np.array([match_trace_ent[j], match_trace_ext[j]][n]) #row: cell, column: data point
                ax = plt.subplot(gs[j,n])
                #Plot out heatmap 
                im = ax.imshow(heatmap_trace, cmap='jet', interpolation='none',aspect='auto',vmin = -2, vmax = 10)
                #Marking out the timine of crossing event (t = 0 s)
                ax.vlines([heatmap_trace.shape[1]/2], -.5, heatmap_trace.shape[0]-.5, color = 'white', linestyles='dotted')
                [ax.set_title(f"{s}\n{['Enter','Exit'][n]} conditioned side \n") if j == 0 else None]
                [ax.set_ylabel(f'i-{comb.split("_")[1].lower()}\no-{comb.split("_")[-1].lower()}') if n == 0 else None]
                [ax.set_yticks([0,heatmap_trace.shape[0]-1],[1,heatmap_trace.shape[0]]) if n == 0 else ax.set_yticks([])]
                [ax.set_xticks(np.linspace(0,len(time)-1,5,dtype = 'int'),np.linspace(round(time.min()),round(time.max()),5,dtype = 'int'))\
                if j == len(cross_res)-1 else ax.set_xticks([])]
                ax.set_xlabel(['Time from event (s)' if j == len(cross_res)-1 else None][0])
        # Add colobar
        cax = fig.add_axes([.93, 0.27, 0.02, 0.1])
        cbar = fig.colorbar(im,cax = cax)
        cbar.ax.set_yticks([10,8,6,4,2,0,-2],[f'>10',8,6,4,2,0,f'<-2']);
        cbar.set_label('Z-score ($\sigma$)', labelpad=-15,y=1.21,rotation=360)
        os.chdir(save_dir)
        [fig.savefig(f'{s} - {heatmap_save_name}.png', dpi = 600, bbox_inches  = 'tight') if save_plot else None]

        #Average Traces
        fig = plt.figure(figsize = (10,2*len(cross_res)))
        gs = gridspec.GridSpec(len(cross_res),2,wspace=0.1, hspace=0.2)
        for j, comb in enumerate(cross_res):
            for n in range(np.unique(trace_meta['Cross type']).size):
                include_trace = np.array([match_trace_ent[j], match_trace_ext[j]][n]) #row: cell, column: data point
                # heatmap_trace = heatmap_trace[np.argsort(np.mean(heatmap_trace, axis = 1)),:] 
                ax = plt.subplot(gs[j,n])
                ax.plot(time, ave_trace:=include_trace.mean(axis = 0),label = f'Mean\n(n = {include_trace.shape[0]})')#Average trace 
                ax.fill_between(time, ave_trace + include_trace.std(axis = 0)/np.sqrt(include_trace.shape[0])\
                                ,ave_trace - include_trace.std(axis = 0)/np.sqrt(include_trace.shape[0]), alpha = 0.2)
                ax.vlines([0], -6,18, color = 'red', linestyles='dotted')#min(ave_trace),max(ave_trace),
                ax.hlines([-thr,thr], time.min(), time.max(),color = 'gray', linestyles='dashed')
                ax.hlines([0], time.min(), time.max(),color = 'black', linestyles='dashed')
                ax.set(ylabel = [f'i-{comb.split("_")[1].lower()}\no-{comb.split("_")[-1].lower()}\n\nZ-score ($\sigma$)' if n == 0 else None][0],
                        xlabel= ['Time from event (s)' if j == len(cross_res)-1 else None][0],
                        title = [f"{s}\n{['Enter','Exit'][n]} conditioned side \n" if j == 0 else None][0],
                        ylim= (-6,18)
                )
                [ax.set_yticks(np.linspace(-6,18,5)) if n == 0 else ax.set_yticklabels([])]
                [ax.set_xticks(np.linspace(round(time.min()),round(time.max()),11,dtype = 'int')) if j == len(cross_res)-1 else ax.set_xticklabels([])]
                ax.legend(loc = 2, frameon = False)
        os.chdir(save_dir)
        [fig.savefig(f'{s} - {mean_trace_save_name}.png', dpi = 600, bbox_inches  = 'tight') if save_plot else None]

def pref_class_heatmap(state, trace_meta, cell_pref, cond_indx, uncond_indx,class_colors, save_name ,save_plot, save_dir):
    '''
    Inputs
    -------

    state : 1-D array
        Corresponding CP state of every calcium trace
    trace_meta : dictionary
        Metadata for trajectory analysis
    cell_pref : 1-D array of cell preference 
        'INT' = unconditioned preferring, 'EXT' = conditioned preferring, 'NR' = no preference
    cond_indx : 1-D array
        index of conditioned side enter associated trace
    uncond_indx : 1-D array
        index of unconditioned side enter associated trace
    class_colors : list with lenght of 3 
        Color (of averaged trace) denoting cond-pref, uncond-pref, or no pref cells
    save_ename : string
        Name of the exported figure
    save_plot : Boolean
        Whether or not the plotted graphs should be saved
    save_dir : r-string
        The directory in which the output will be saved
    Returns
    -------
    plotting 
    '''
    time = trace_meta['Trace time']
    type_indx  = [np.where(cell_pref == re)[0] for re in ['INH','EXT','NS']]
    f, ax  = plt.subplots(ncols = 2, nrows = 4, figsize = (10,13))
    gs = gridspec.GridSpec(4,2,height_ratios=np.concatenate([np.array([len(l) for l in type_indx]),[len(cell_pref)/4]]),wspace=0.1, hspace=0.2)
    for j,indx in enumerate(type_indx):
        for n ,cr in enumerate(np.sort(np.unique(trace_meta['Cham']))):
            #plot out heatmap of various cell types
            heatmap_trace = np.array([trace_meta['Extracted trace mean'][cond_indx[indx]], trace_meta['Extracted trace mean'][uncond_indx[indx]]][n]) #row: cell, column: data point
            ax = plt.subplot(gs[j,n])
            im = ax.imshow(heatmap_trace, cmap='jet', interpolation='none',aspect='auto',vmin = -2, vmax = 10)#Plot out heatmap 
            ax.vlines([heatmap_trace.shape[1]/2], -.5, heatmap_trace.shape[0]-.5, color = 'white', linestyles='dotted')
            [ax.set_title(f"{['Conditioned','Unconditioned'][n]} Side\n") if j == 0 else None]
            [ax.set_ylabel(f'{["Cond.","Uncond.","No"][j]}\npref.') if n == 0 else None]
            [ax.set_yticks([0,heatmap_trace.shape[0]-1],[1,heatmap_trace.shape[0]]) if n == 0 else ax.set_yticks([])]
            [ax.set_xticks(np.linspace(0,len(time)-1,5,dtype = 'int'),np.linspace(round(time.min()),round(time.max()),5,dtype = 'float'))\
            if j == 2 else ax.set_xticks([])]
            ax.set_xlabel(['Time from event (s)' if j == 2 else None][0])
            #Plot out averaged traces
            ax = plt.subplot(gs[3,n])
            ax.plot(time, ave:= heatmap_trace.mean(axis = 0), color = class_colors[j],\
                            label = ['Cond. Pref', 'Uncond. Pref', 'No Pref'][j])
            ax.fill_between(time, ave+heatmap_trace.std(axis = 0)/np.sqrt(len(indx)),\
                            ave-heatmap_trace.std(axis = 0)/np.sqrt(len(indx)),\
                            facecolor = class_colors[j], alpha = .1)
            if j == 2:
                ax.set(xticks = np.linspace(round(time.min()),round(time.max()),5,dtype = 'float'), 
                    xlim = (round(time.min()), round(time.max())),
                    xlabel='Time from event (s)',
                    ylabel=['Z-score ($\sigma$)' if n == 0 else ''][0])
                
                ax.hlines(0, 0, time.max(),color = 'black', linestyle = 'dashed')
                ax.vlines(0,trace_meta['Extracted trace mean'].min(),trace_meta['Extracted trace mean'].max()/2,color = 'black', linestyle = 'dotted')
                [ax.legend(loc = 2, frameon = False) if n == 0 else None]
    #Add color bar
    cax = f.add_axes([.93, 0.27, 0.02, 0.1])
    cbar = f.colorbar(im,cax = cax)
    cbar.ax.set_yticks([10,8,6,4,2,0,-2],[f'>10',8,6,4,2,0,f'<-2']);
    cbar.set_label('Z-score ($\sigma$)', labelpad=-15,y=1.21,rotation=360)
    f.suptitle(f'{state}', weight ='bold')
    
    if save_plot:
        os.chdir(save_dir)
        f.savefig(save_name, dpi = 600, bbox_inches  = 'tight')
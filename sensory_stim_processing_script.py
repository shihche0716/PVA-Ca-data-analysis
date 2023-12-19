#Final edit 20231215 21:03
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
import scipy

from matplotlib import gridspec
import seaborn as sns
import itertools

from matplotlib.ticker import FormatStrFormatter
from iteration_utilities import flatten
from matplotlib.colors import ListedColormap

from pylab import *
# Set the ticks and label of figure

# plt.rcParams.update(matplotlib.rcParamsDefault)
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

#######################################################################################
#Calibration
def binning_seg(data,sprt,bsz,min_snr,base_ave = None, base_sd = None, norm_meth = [False,False]):
    '''
    input parameters: 
        1. sprt: Sampling rate (Hz)
        2. bsz: Bin size (Frames)
        3. min_snr: Cutoff value for detecting responsive event (Z-score or normalized value)
        5. norm_meth: Vector with two Boolean values  
                First: restandardize the bin base on formula: F - base_ave / base_sd
                Second: Restandardize the bin base on formula: F - base_ave / base_ave

    output parameters:
        1. tt: Total length of signal (s)
        2. min_snr: Cutoff value for detecting responsive event (Z-score or normalized value)
        3. cnt: Array of binning results from original signal
        4. ext_seg: Array of time points with active event 
        5. int_seg: Array of time points with silent event 
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

def normalize(trace, dev, denominator): 
    return np.array((trace - dev)/denominator, dtype = 'float')

def neuron_res(ext_seg,inh_seg,cutoff):
    if len(ext_seg) == len(inh_seg):
        res = ['EXT' if ext_seg[i]>= cutoff else 'INH' if inh_seg[i] >= cutoff else 'NR' for i in range(len(ext_seg))]
    else: 
        print('Unmatched amount between length of EXT/INH segments!')
    return np.array(res,dtype = 'str') 

def neuron_res_Z(diff,cutoff):
    res = ['EXT' if diff[i]>= cutoff else 'INH' if diff[i] < -cutoff else 'NR' for i in range(len(diff))]
    return np.array(res,dtype = 'str')

def entropy(stim_res):
    p_logp = []
    K = 1/(len(stim_res)*(1/len(stim_res))*(-np.log(1/len(stim_res))))#Maximum entropy - calibrate entropy scaling factor 
    for re in stim_res:
        p = re/sum(stim_res)
        p_logp.append(p*np.log(p)) 
    return -K*(sum(p_logp))

def noise_to_signal(stim_res):
    sort_res = stim_res[np.argsort(stim_res)]
    return sort_res[-2]/sort_res[-1]
def star_report(p_value):
    star = ['***' if p_value < 0.001 else '**' if p_value<0.01 else '*' if p_value<0.05 else 'N.S']
    return star[0]
#######################################################################################
#Data preparation

def generate_sensory_metadata(file_dir, file_name, bin_num = 6):
    data = pd.read_csv(os.path.join(file_dir,file_name),header = None, low_memory=False).values
    row_names = data[:,0]
    bin_start_index = np.where(np.array(['Bin SNR' in tt for tt in row_names]))[0][0]+1
    trace_start_index = np.where(row_names == 'Time (s)')[0][0]+1

    metadata = {
        'Cell ID': data[1,1:],
        'Stimulus':data[2,1:],
        'File name': data[0,1:],
        'Trace time': np.array(data[trace_start_index:, 0] ,dtype = 'float'), 
        'Calcium trace': np.array(data[trace_start_index:, 1:],dtype = 'float'),
        'Standardized bin': np.array(data[bin_start_index:bin_start_index+bin_num-1,1:],dtype = 'float'),
        'AUC': np.array(data[6,1:],dtype = 'float'),
        'Baseline AUC': np.array(data[5,1:],dtype = 'float'),
        'Baseline mean': np.array(data[3,1:],dtype = 'float'),
        'Baseline std': np.array(data[4,1:],dtype = 'float')
    }

    metadata['CP stage']= np.array([f.split(' ')[-1].\
                    split('.csv')[0].\
                        removeprefix('formalin-') \
                            for f in metadata['File name']], dtype = 'str')
    metadata['Mouse ID']= np.array([f.split(' ')[1] for f in metadata['File name']],dtype = 'str')
    metadata['Unique stim list']=np.unique(metadata['Stimulus'])
    metadata['Unique state list']= np.unique(metadata['CP stage'])

    return metadata

def neuron_classification_bin(metadata, bin_thres, bin_num_cutoff, 
                              save_prefix, export, save_dir, class_annot = ['INH', 'EXT', 'NR']):
    active_bin, inhibited_bin = (metadata['Standardized bin']>=bin_thres).sum(axis =0) , \
    (metadata['Standardized bin']<-bin_thres).sum(axis = 0) # Count responsive bin  baseed on 'thres'

    net_response = metadata['AUC'] - metadata['Baseline AUC'] #Calibrate direction of response after stim onset (rise > 0 or fall < 0)

    res = neuron_res(active_bin,inhibited_bin,bin_num_cutoff) #Classifty cells based on responsive bin amount 
    response_type = np.array([r if r == class_annot[0] and net_response[i]<0 else r \
                      if r == class_annot[1] and net_response[i]>0 else class_annot[2] \
                      for i,r in enumerate(res)]) #Correction base on net response direction, redefine mismatched cases to Non-responsive
    #Export classification results
    if export:
        df = np.c_[metadata['Mouse ID'], metadata['Cell ID'], metadata['CP stage'], metadata['Stimulus'], response_type, active_bin, inhibited_bin, net_response]
        col_names = np.array(['Mouse ID', 'Cell ID', 'Trial', 'Stimulus', \
                        f'Response type (bin response threshold = {bin_thres}, min responsive bin number = {bin_num_cutoff})',
                        'Excited bin number', 'Inhibited bin number', 'Net AUC change'])
        df = np.r_[col_names.reshape(1,col_names.size), df]
        df = pd.DataFrame(np.array(df, dtype = 'str'))
        os.chdir(save_dir)
        df.to_csv(f"{save_prefix} neural classification all.csv",index = False, header =False)

    return active_bin, inhibited_bin, net_response, response_type

#######################################################################################
#Plotting functions

def sort_stim_pref_response_heatmap(data,uni_ID, stimulus,state, state_type, min_criteria, \
                                    col_range, chosen_stim, cbar_label , save_text, pair_cell,save_fig, save_dir):
    '''
    Input:
    data : 1-D array of stimulus responses
    uni_ID : 1-D string array of unique neural ID (animal ID + cell ID)
    stimulus : 1-D array of stimulus encountered
    state : 1-D array of the treatment type of corresponding trial
    state_type : a list of treatments indlucded in analysis (order specified)
    chosen_stim : a list of stimuli included in analysis (order specified)
    pair_cell : Boolean, whether cell index of other column shall match with the first column

    min_criteria : a float specifying the minimum value for stimulus-responses to be defined as responsive
    col_range : [min, max], a 2-elements vector specifying the range of heatmap colorbar 
    cbar_label : a string placed above colorbar denoting the unit of cellular responses (Z-score, Normalized activity etc.)
    save_text : Prefix for the filename of saved figure
    save_fig : Boolean to save the figure

    '''
    #Adjusting optimal size of figure  
    f, ax = plt.subplots(nrows = 1, ncols = len(state_type),figsize=(4*len(state_type)+.5,5))
    fin_ratio =  np.array(np.repeat(4,len(state_type)),dtype = 'float')
    fin_ratio[-1] += 0.5
    spec = gridspec.GridSpec(ncols=len(state_type), nrows=1,width_ratios=fin_ratio)
    
    for i,st in enumerate(state_type):
        #Extract stimulus-associated response in given state
        norm_matrix = []
        st_indx = np.where(state == st)[0] #Index of cell with certain state 
        stim = stimulus[st_indx]
        norm_sig = data[st_indx]
        total_s = []
        #Generate matrix for heatmap
        keep_cell = np.array([sum([np.where(np.logical_and(uni_ID[st_indx] == ci,stim == s))[0].size >0\
                         for s in chosen_stim])== len(chosen_stim) for ci in np.unique(uni_ID[st_indx])]) #Remove cells with missing value
        cell_in_keep = np.array([c in np.unique(uni_ID[st_indx])[keep_cell] for c in uni_ID[st_indx]])

        for s in chosen_stim:
            total_s.append(s)
            norm_matrix.append(norm_sig[np.where(np.logical_and(cell_in_keep, stim == s))[0]]) 
        norm_matrix = np.array(norm_matrix)

        #Classify cells based on maximum responses 
        max_indx = np.array([np.argmax(norm_matrix[:,i]) for i in range(norm_matrix.shape[1])])
        all_indx, nr_indx, class_amount = [],[],[] # Boundary of every class 
        #Sort cell within class
        for j in range(len(chosen_stim)):
            stim_indx = np.where(max_indx == j)[0] #Select cell indexes withing given class  
            pref_norm = norm_matrix[:,stim_indx]
            stim_sort_indx = np.argsort(pref_norm[j,:])[::-1] #Sort the index within class 
            sort_pref_norm = pref_norm[:,stim_sort_indx]
            
            #Select significant units
            stim_sort_res_indx = np.where(sort_pref_norm[j,:] >= min_criteria)[0] #Index of cells with above-criteria value 
            stim_sort_NR_indx = np.where(sort_pref_norm[j,:] < min_criteria)[0] #Index of cells with beneath-criteria value 
            sort_pref_norm_res = sort_pref_norm[:,stim_sort_res_indx]
            all_indx.append(stim_indx[stim_sort_indx[stim_sort_res_indx]])
            nr_indx.append(stim_indx[stim_sort_indx[stim_sort_NR_indx]])

            class_amount.append(len(stim_sort_res_indx))
            if j == 0:
                norm_sort = sort_pref_norm_res
                nr_units = sort_pref_norm[:,stim_sort_NR_indx]
            else: 
                norm_sort = np.c_[norm_sort,sort_pref_norm_res]
                nr_units = np.c_[nr_units,sort_pref_norm[:,stim_sort_NR_indx]]
        #Combine Non-responsive units with original heatmap 
        class_amount.append(len(nr_indx))
        norm_sort = np.c_[norm_sort,nr_units]
        if pair_cell: #Remain the same indexing order for other states
            if i == 0: 
                all_indx = np.concatenate([list(flatten(all_indx)),list(flatten(nr_indx))]) #Flattened & concatenated
                class_amount1 = class_amount
            else:
                norm_sort = norm_matrix[:,all_indx]
        else: 
            class_amount1= class_amount

        loc = [sum(class_amount1[:k]) for k in range(len(class_amount1)) if k >=1] #Location of horizontal dash lines 

        #Plotting
        ax = plt.subplot(spec[i])#Determine plot location 
        im = ax.imshow(np.array(np.matrix.transpose(norm_sort),       
        dtype = 'float'), cmap='Blues',interpolation = 'none', \
            vmin = col_range[0], vmax = col_range[1],aspect='auto')#plot out heatmap 
        ax.hlines(loc,-0.5,len(chosen_stim)-0.5, color = 'k', linestyles='dashed') #Indicate location of non-responsive neurons 
        
        ax.set(title = f'State: {st}',
                ylabel=f'Cell #',
                yticks=[0,*loc,norm_sort.shape[1]-1],
                yticklabels=[1,*np.array(loc)+1,norm_sort.shape[1]],
                xticks=np.linspace(0,len(chosen_stim)-1,len(chosen_stim)))
        ax.set_xticklabels(total_s,rotation=45)

    f.colorbar(im,label = cbar_label) 
    f.tight_layout()
    os.chdir(save_dir)
    if save_fig:
        f.savefig(f'{save_text} Normalized res combined {"- Paired " if pair_cell else ""}.png', bbox_inches='tight',dpi = 400)
        # #Save results as csv file for replotting 
        # df = np.c_[G_range,np.array(exp_dat)]
        # fst_r =np.insert(np.array(np.arange(0,301), dtype = 'str'),0,'G_mM\K(TALK)_pS')
        # df = np.r_[np.reshape(fst_r, (1,fst_r.size)),\
        #     np.array(df, dtype = 'str')]
        # df = pd.DataFrame(np.array(df, dtype = 'str'))
        # os.chdir(save_dir)
        # df.to_csv(f"gKCa = {[10,200][j]}," + "Mean cytosolic Ca level (uM) vs tuning.csv" , index = False, header =False)
    f.show()

def response_stack_bar(neuron_class,metadata, spec_state_types, spec_stim_types, fig_ratio, bar_width, \
                       bar_color, save_prefix,save_plot, save_dir, class_annot = ['EXT', 'INH', 'NR']):

    fig= plt.figure(figsize = fig_ratio)
    gs = gridspec.GridSpec(int(np.ceil(len(spec_stim_types)/4)),4)

    ratio_summary = {
        'Stimulus':[],
        'Chronic pain stage':[],
        'Total cell count':[],
        'Excited percentage (%)':[],
        'Inhibited percentage (%)':[],
        'No response percentage (%)':[],
        'Chi-square statistics (with first group)':[],
        'Chi-square p-value (with first group)':[],
        'Annotation':[]
    }

    for i , stim in enumerate(spec_stim_types):
        #Specify plot coordinate
        ax = plt.subplot(gs[i])
        t_n, bar_label = [], []
        ext_ratio, inh_ratio, nr_ratio = np.array([]), np.array([]), np.array([])

    
        for j , state in enumerate(spec_state_types):
            match_neural_response = neuron_class[np.where(np.logical_and(metadata['Stimulus'] == stim, metadata['CP stage'] == state))[0]]
            t_n.append(match_neural_response.size) #Count the amount of cells in this group
            bar_label.append(state+'\nn = '+str(t_n[-1])) #With chronic pain stage and corresponding cell number annotated

            ext_ratio = np.append(ext_ratio, 100*list(match_neural_response).count(class_annot[0])/t_n[-1]) #Excited neurons
            inh_ratio = np.append(inh_ratio, 100*list(match_neural_response).count(class_annot[1])/t_n[-1]) #Inhibited neurons
            nr_ratio = np.append(nr_ratio, 100*list(match_neural_response).count(class_annot[2])/t_n[-1]) #No response neurons

            ratio_summary['Stimulus'].append(stim)
            ratio_summary['Chronic pain stage'].append(state)
            ratio_summary['Total cell count'].append( t_n[-1])         
            ratio_summary['Excited percentage (%)'].append(ext_ratio[-1])   
            ratio_summary['Inhibited percentage (%)'].append(inh_ratio[-1])  
            ratio_summary['No response percentage (%)'].append(nr_ratio[-1])  

        #Draw stack bar plot 
        ax.bar(bar_label, ext_ratio,\
                bar_width, label='Excited',color =bar_color[0])
        ax.bar(bar_label,inh_ratio,\
               bar_width,bottom=ext_ratio, label='Inhibited',color =bar_color[1])
        ax.bar(bar_label, nr_ratio,\
                bar_width, bottom=ext_ratio+inh_ratio, label='No response',color =bar_color[2])
        ax.set(ylim=(0,100),
               ylabel='Percentage of cell (%)',
               title =f'{stim}\n\n')
        [ax.legend(bbox_to_anchor=(1,1), loc=2, frameon = False) if i == len(spec_stim_types)-1 else None]#Show label only in the last graph 
        
        #Statistics
        ref = np.array([ext_ratio[0],inh_ratio[0],nr_ratio[0]])

        ratio_summary['Chi-square statistics (with first group)'].append('ref')
        ratio_summary['Chi-square p-value (with first group)'].append('ref')
        ratio_summary['Annotation'].append('ref')

        for j in range(1,len(spec_state_types)):
            stat, pv = scipy.stats.chisquare(np.array([ext_ratio[j],inh_ratio[j],nr_ratio[j]]), f_exp=ref)
            annot = star_report(pv)
            ratio_summary['Chi-square statistics (with first group)'].append(str(stat))
            ratio_summary['Chi-square p-value (with first group)'].append(str(pv))
            ratio_summary['Annotation'].append(annot)
            #Add annotation for statistical test result
            ax.text(j,105,annot,horizontalalignment='center') 

    fig.tight_layout()
    os.chdir(save_dir)
    #Convert dictionary to csv
    df = pd.DataFrame.from_dict(ratio_summary)
    df.to_csv('Responsive ratio ref data table.csv', index = False, header = True)

    [fig.savefig(f'{save_prefix}responsive ratio stack bar.png',dpi = 600) if save_plot else None]
#20230523 new func 

def all_stim_heatmap(signal, metadata, save_plot, save_dir):
    state_types = metadata['Unique state list']
    stim_types = metadata['Unique stim list']
    #Data preparation 
    for f_n in np.unique(metadata['File name']):#run through included files
        f_indx = np.where(metadata['File name'] == f_n)[0] #Columns for certain file
        indx_ID = np.unique(metadata['Cell ID'][f_indx])
        for i,s in enumerate(np.unique(metadata['Stimulus'])):
            if s not in  np.unique(metadata['Stimulus'][f_indx]):#add null signal for non-existing stimulus-evoked responses 
                stimulus_hm = np.concatenate([metadata['Stimulus'],np.repeat(s,len(indx_ID))])
                state_hm = np.concatenate([metadata['CP stage'],np.repeat(metadata['CP stage'][np.min(f_indx)],len(indx_ID))])
                signal_hm = np.hstack([signal, np.zeros([np.shape(signal)[0], len(indx_ID)])*10])
            else:
                stimulus_hm = np.copy(metadata['Stimulus'])
                state_hm = np.copy(metadata['CP stage'])
                signal_hm = np.copy(signal)
    
    time = metadata['Trace time']
    st_indx = np.where(time <0)[0].max()
    signal_hm = signal_hm - np.mean(signal_hm[:st_indx,:], axis = 0) #Vertical signal shifting 

    #Plotting
    f ,axes = plt.subplots(nrows = len(state_types),ncols = len(stim_types),figsize = (30,20),sharex=False, sharey=False)
    for j, st in enumerate(state_types): #Loop across CP stages
        for i,s in enumerate(stim_types): # Loop across stimulus
            match_indx = np.where(np.logical_and(stimulus_hm == s, state_hm == st))[0]
            sig = signal_hm[:,match_indx]
            dif = sig[:st_indx,:].mean(axis = 0)-sig[st_indx+1:,:].mean(axis = 0)
            #Plot heatmap     
            im=axes[j,i].imshow(np.array(np.matrix.transpose(sig[:,np.argsort(dif)]),
        dtype = 'float'), cmap='jet', interpolation='none' ,vmin = -5, vmax= 20,aspect='auto')
            if j == 0:
                axes[j,i].set_title(s)
            if i == 0: 
                axes[j,i].set(ylabel=f'{st} Cell # ',
                                yticks=[0,sig.shape[1]-1],
                                yticklabels=[sig.shape[1],1])
            else: 
                axes[j,i].set_yticks([])
            axes[j,i].vlines([st_indx], 0, sig.shape[1]-1, color = 'white', linestyles = 'dotted')
            axes[j,i].set(xticks = [0,st_indx+1,sig.shape[0]],
                        xticklabels=[int(time.min()),0,int(time.max())],
                        xlabel = 'Time (s)')
    cb_ax = f.add_axes([0.92, 0.1, 0.01, 0.1])
    cbar = f.colorbar(im, cax=cb_ax,label = 'Z-score')
    os.chdir(save_dir)
    [f.savefig('All stim heatmap.png', dpi = 700) if save_plot else None]

def all_stim_heatmap_ind(signal, metadata, save_plot, save_dir):
    state_types = metadata['Unique state list']
    stim_types = metadata['Unique stim list']
    #Data preparation 
    for f_n in np.unique(metadata['File name']):
        f_indx = np.where(metadata['File name'] == f_n)[0]
        indx_ID = np.unique(metadata['Cell ID'][f_indx])
        for i,s in enumerate(np.unique(metadata['Stimulus'])):
            if s not in  np.unique(metadata['Stimulus'][f_indx]):
                stimulus_hm = np.concatenate([metadata['Stimulus'],np.repeat(s,len(indx_ID))])
                state_hm = np.concatenate([metadata['CP stage'],np.repeat(metadata['CP stage'][np.min(f_indx)],len(indx_ID))])
                signal_hm = np.hstack([signal, np.zeros([np.shape(signal)[0], len(indx_ID)])*10])
                m_ID_hm = np.concatenate([metadata['Mouse ID'],np.repeat(s,len(indx_ID))])
            else:
                stimulus_hm = np.copy(metadata['Stimulus'])
                state_hm = np.copy(metadata['CP stage'])
                signal_hm = np.copy(signal)
                m_ID_hm = np.copy(metadata['Mouse ID'])
    
    time = metadata['Trace time']
    st_indx = np.where(time <0)[0].max()
    signal_hm = signal_hm - np.mean(signal_hm[:st_indx,:], axis = 0) #Vertical signal shifting 
    #Plotting
    
    for m in np.unique(metadata['Mouse ID']):
        print(m)
        f ,axes = plt.subplots(nrows = len(state_types),ncols = len(stim_types),figsize = (30,20),sharex=False, sharey=False)
        for j, st in enumerate(state_types): #Loop across CP stages
            for i,s in enumerate(stim_types): # Loop across stimulus
                match_indx = np.where(np.logical_and(stimulus_hm == s, np.logical_and(state_hm == st, m_ID_hm == m)))[0]
                sig = signal_hm[:,match_indx]
                dif = sig[:st_indx,:].mean(axis = 0)-sig[st_indx+1:,:].mean(axis = 0)
                #Plot heatmap     
                im=axes[j,i].imshow(np.array(np.matrix.transpose(sig[:,np.argsort(dif)]),
            dtype = 'float'), cmap='jet', interpolation='none' ,vmin = -5, vmax= 20,aspect='auto')
                if j == 0:
                    axes[j,i].set_title(s)
                if i == 0: 
                    axes[j,i].set(ylabel=f'{st} Cell # ',
                                    yticks=[0,sig.shape[1]-1],
                                    yticklabels=[sig.shape[1],1])
                else: 
                    axes[j,i].set_yticks([])
                axes[j,i].vlines([st_indx], 0, sig.shape[1]-1, color = 'white', linestyles = 'dotted')
                axes[j,i].set(xticks = [0,st_indx+1,sig.shape[0]],
                            xticklabels=[int(time.min()),0,int(time.max())],
                            xlabel = 'Time (s)')
        cb_ax = f.add_axes([0.92, 0.1, 0.01, 0.1])
        cbar = f.colorbar(im, cax=cb_ax,label = 'Z-score')
        f.suptitle(m, weight= 'bold')
        os.chdir(save_dir)
        [f.savefig(f'{m} All stim heatmap.png', dpi = 700) if save_plot else None]


def cell_subset_heatmap(signal, metadata, neuron_class, net_response , spec_stim_types, spec_state_types, group_color, save_plot, save_dir):
    #signal prep
    time = metadata['Trace time']
    st_indx = np.where(time < 0)[0].max()
    for sti in spec_stim_types:
        fig = plt.figure(figsize = (14,6))
        gs = gridspec.GridSpec(2,len(spec_state_types),height_ratios=[4,1],wspace=.3, hspace=.3)
        for j, st in enumerate(spec_state_types):
            match_indx = np.where(np.logical_and(metadata['CP stage'] == st, metadata['Stimulus'] == sti))[0]
            #prepare averaged cell traces & heatmap traces
            ave_trace, sem_trace, hm_indx, sig_exist= [],[],[],[]
            if len(match_indx)>0:
                [ext,inh,ns] = [match_indx[np.where(neuron_class[match_indx] == r)] for r in ['EXT', 'INH', 'NR']]
                for k, indx in enumerate([ext,inh,ns]):#Exclude response combination with no matched cell
                    sig_exist.append(len(indx)>0) #exclude cell type without cell
                    ave_trace.append(signal[:,indx].mean(axis =1))
                    sem_trace.append(signal[:,indx].std(axis =1)/np.sqrt(len(match_indx)))
                    hm_indx.append(indx[np.argsort(net_response[indx])[::-1]])#(signal_z[:,indx].mean(axis =0))[::-1]])
            else:
                print(f'{st} {sti} no match cells!')

            hm_traces = signal[:,np.concatenate(hm_indx)]
            # hm_traces = hm_traces- np.mean(hm_traces[:st_indx,:], axis = 0) #Vertical signal shifting
            time = metadata['Trace time']
            # plot out the heatmap 
            axes =  plt.subplot(gs[0,j])
            im = axes.imshow(hm_traces.T, cmap = 'jet',interpolation = 'none',aspect='auto', vmin = -5, vmax = 20);
            
            axes.vlines([st_indx], 0, hm_traces.shape[1]-.5, color = 'white', linestyles='dotted')
            sep_p = [len(ext), len(ext)+len(inh)]
            axes.hlines(sep_p, 0, hm_traces.shape[0]-1,  color = 'white', linestyles='dotted')
            axes.set(xlabel = 'Time from stim onset (s)',
                    title = f'{st}',
                    ylabel = ['Cell #' if j==0 else None][0],
                    yticks = [0,len(ext)-1,len(ext)+len(inh)-1,len(match_indx)-1],
                    yticklabels= [1,len(ext),len(ext)+len(inh),len(match_indx)],
                    xticks = np.linspace(0,hm_traces.shape[0]-1, 8, dtype = 'int'),
                    xticklabels = np.linspace(round(time.min()),round(time.max()),8,dtype = 'int'))
            #Traces
            axes = plt.subplot(gs[1,j])
            for k in range(3):
                if sig_exist[k]:
                    base = ave_trace[k][:st_indx].mean()
                    axes.plot(time, ave_trace[k]-base, color = group_color[k])
                    axes.fill_between(time, ave_trace[k] - base + sem_trace[k],\
                                    ave_trace[k]-base - sem_trace[k],\
                                        alpha = .3, color = group_color[k])
                    axes.set(ylim = (-10,15),
                        xlim = (int(time.min()),int(time.max())),
                        xlabel= 'Time from stim onset (s)',
                        ylabel = ['Z-score ($\sigma$)' if j == 0 else None][0],
                        yticks = [[] if j !=0 else [-5,0,5,10,15]][0],
                        xticks = np.linspace(round(time.min()),round(time.max()),8,dtype = 'int'))
            axes.vlines([0], -10,15, color = 'black', linestyles='dotted')
            [axes.plot(0,0, label = ['Ext','Inh','NR'][k], color = group_color[k]) for k in range(3)]
            [axes.legend(loc = 2,frameon = False,bbox_to_anchor = (1.05,1)) if j == len(spec_state_types)-1 else None]

        cax = fig.add_axes([.92, 0.4, 0.01, .3])
        cbar = fig.colorbar(im,cax = cax)
        cbar.set_label('Z-score ($\sigma$)', labelpad=-15,y=1.21,rotation=360)

        fig.suptitle(sti, weight = 'bold')
        os.chdir(save_dir)
        [fig.savefig(f'{sti} cell subgroup trace & heatmap.png', dpi = 600, bbox_inches  = 'tight') if save_plot else None]



# Not used plotting functions 
###############################################################################################################
def response_type_paired_map(fig_ratio,res,stimulus,state,stim,state_type1,save = True):#Demonstrating the cell type conversion across trials
    #Convert response class into numerical form 
    state_res = np.array([1 if R == 'EXT' else -1 if R == 'INH' else 0 for R in res])

    #Plot 
    f,ax = plt.subplots(nrows = int(len(stim)/4), ncols = 4,figsize = fig_ratio)
    gs = gridspec.GridSpec(int(np.ceil(len(stim)/4)),4)
    indx = 0
    res_matrix = []
    for i,comb in enumerate(itertools.product(stim, state_type1)):
        ax = plt.subplot(gs[indx])
        #Extract stimulus-associated response in given state
        res_st = state_res[np.where(np.logical_and(stimulus == comb[0], state == comb[1]))[0][()]]
        if i%len(state_type1) == 0:
            #Sort by response type - keep constant in following states
            res_indx = np.argsort(res_st)
            class_amount = [sum(res_st == R) for R in [1,0,-1]]
        
        res_matrix.append(res_st[res_indx[::-1]])
        if (i+1)%len(state_type1) == 0:
            res_matrix = np.array(res_matrix) 
            ax = sns.heatmap(-(res_matrix).T, vmin = -1, vmax = 1,\
                cmap= ListedColormap(['blue', 'gray', 'red']),cbar_kws={'ticks':[],'label': 'Response - (Red: INH / Blue: EXT / Gray: NR)'}, ax = ax)
            ax.set_yticks(a := np.array([*[sum(class_amount[:k]) for k in range(3)],len(res_st)]),a+1)
            ax.hlines(a,0,5,linestyle = 'dashed',color = 'white')
            ax.set_ylabel('Cell #')
            ax.set_xticklabels(state_type1)
            ax.set_xlabel('Stage')
            ax.set_title(f'{comb[0]}')
            indx+=1
            res_matrix = []
    f.tight_layout()
    if save:
        f.savefig(f'Response class across stages.png', bbox_inches='tight',dpi = 400)


def cell_subset_heatmap_ind(sensory_meta, res_n, diff, stim_types, state_type, signal_z, group_color, save_plot):
    time = sensory_meta['Trace time']
    st_indx = np.where(time <0)[0].max()

    for m in np.unique(sensory_meta['Mouse ID']):
        print(m)
        for sti in stim_types:
            fig, _  = plt.subplots(ncols = len(state_type), nrows = 2, figsize = (14,6))
            gs = gridspec.GridSpec(2,5,height_ratios=[4,1],wspace=.3, hspace=.3)
            for j, st in enumerate(state_type[[1,0,2,3,4]]):
                match_indx = np.where(np.logical_and(sensory_meta['CP stage'] == st, np.logical_and(sensory_meta['Mouse ID'] == m,sensory_meta['Stimulus'] == sti)))[0]
                [ext,inh,ns] = [match_indx[np.where(res_n[match_indx] == r)] for r in ['EXT', 'INH', 'NR']]
                #prepare averaged cell traces & heatmap traces
                ave_trace, sem_trace, hm_indx, sig_exist=[],[],[],[]
                if len(match_indx)>0:
                    for k, indx in enumerate([ext,inh,ns]):#Exclude response combination with no matched cell
                        sig_exist.append(len(indx)>0) #exclude cell type without cell
                        ave_trace.append(signal_z[:,indx].mean(axis =1))
                        sem_trace.append(signal_z[:,indx].std(axis =1)/np.sqrt(len(match_indx)))
                        hm_indx.append(indx[np.argsort(diff[indx])[::-1]])#(signal_z[:,indx].mean(axis =0))[::-1]])
                else:
                    print(f'{st} {sti} no match cells!')

                hm_traces=signal_z[:,np.concatenate(hm_indx)]
                hm_traces = hm_traces- np.mean(hm_traces[:st_indx,:], axis = 0)
                time = sensory_meta['Trace time']
                # heatmap 
                axes =  plt.subplot(gs[0,j])
                im = axes.imshow(hm_traces.T, cmap = 'jet',interpolation = 'none',aspect='auto', vmin = -5, vmax = 20);
                
                axes.vlines([st_indx], 0, hm_traces.shape[1]-.5, color = 'white', linestyles='dotted')
                sep_p = [len(ext), len(ext)+len(inh)]
                axes.hlines(sep_p, 0, hm_traces.shape[0]-1,  color = 'white', linestyles='dotted')
                axes.set(xlabel = 'Time from stim onset (s)',
                        title = f'{st} {m}',
                        ylabel = ['Cell #' if j==0 else None][0],
                        yticks = [0,len(ext)-1,len(ext)+len(inh)-1,len(match_indx)-1],
                        yticklabels= [1,len(ext),len(ext)+len(inh),len(match_indx)],
                        xticks = np.linspace(0,hm_traces.shape[0]-1, 8, dtype = 'int'),
                        xticklabels = np.linspace(round(time.min()),round(time.max()),8,dtype = 'int'))
                #Traces
                axes = plt.subplot(gs[1,j])
                for k in range(3):
                    if sig_exist[k]:
                        base = ave_trace[k][:st_indx].mean()
                        axes.plot(time, ave_trace[k]-base, color = group_color[k])
                        axes.fill_between(time, ave_trace[k] - base + sem_trace[k],\
                                        ave_trace[k]-base - sem_trace[k],\
                                            alpha = .3, color = group_color[k])
                        axes.set(ylim = (-10,15),
                            xlim = (int(time.min()),int(time.max())),
                            xlabel= 'Time from stim onset (s)',
                            ylabel = ['Z-score ($\sigma$)' if j == 0 else None][0],
                            yticks = [[] if j !=0 else [-5,0,5,10,15]][0],
                            xticks = np.linspace(round(time.min()),round(time.max()),8,dtype = 'int'))
                axes.vlines([0], -10,15, color = 'black', linestyles='dotted')
                [axes.plot(0,0, label = ['Ext','Inh','NR'][k], color = group_color[k]) for k in range(3)]
                [axes.legend(loc = 2,frameon = False,bbox_to_anchor = (1.05,1)) if j == len(state_type)-1 else None]

            cax = fig.add_axes([.92, 0.4, 0.01, .3])
            cbar = fig.colorbar(im,cax = cax)
            cbar.set_label('Z-score ($\sigma$)', labelpad=-15,y=1.21,rotation=360)

            fig.suptitle(sti, weight = 'bold')

            [fig.savefig(f'{sti} {m} cell subgroup trace & heatmap.png', dpi = 600, bbox_inches  = 'tight') if save_plot else None]

def stim_cluster_heatmap(sensory_meta,clustering_method, left_stim_indx, right_stim_indx,  uni_ID, state_type, stim_types, save_plot, save_dir):
    for st in state_type:
        cell_num = np.unique(uni_ID[np.where(sensory_meta['CP stage'] == st)[0]]).size
        for k, stim_sub in enumerate([left_stim_indx,right_stim_indx]): #Plot left and right response seperatly
            res_matrix = np.zeros([cell_num, len(stim_sub)])
            for i,s  in enumerate(stim_types[stim_sub]):
                indx = np.where(np.logical_and(sensory_meta['Stimulus']== s, sensory_meta['CP stage'] == st))[0] #Order the cells 
                ref_ID = np.array([uni_ID[indx] if i == 0 else ref_ID][0])
                #Identify missing cell elements 
                missing_index = np.array([np.where(ref_ID == ind) for ind in np.setdiff1d(ref_ID, uni_ID[indx])])
                # correspond_indx = np.array(np.where(ref_ID == ind)[0] for ind in uni_ID[indx])
                # cell_res = signal[99:,indx].mean(axis = 0) -  signal[:98,indx].mean(axis = 0)
                cell_res = [sensory_meta['Standardized bin'][np.argmax(sensory_meta['Standardized bin'][:,k]),k] for k in indx]

                if len(missing_index)> 0:
                    cell_res_cor = np.zeros(len(ref_ID))
                    used_element = 0
                    for j in range(len(ref_ID)):
                        if j not in missing_index:
                            cell_res_cor[j] = cell_res[used_element]
                            used_element+=1
                        else: 
                            cell_res_cor[j] = 0 #np.nan  #Add in 0 as missing values 
                else:
                    cell_res_cor = np.copy(cell_res)
                res_matrix[:,i] = cell_res_cor

            ax = sns.clustermap(res_matrix.T, method = clustering_method , \
                            row_cluster= False  , z_score = 1,#vmin = -.25, vmax = .25, \
                            figsize = (6,3.5), cmap = 'coolwarm', \
                            yticklabels= stim_types[stim_sub], xticklabels= [len(ref_ID)],\
                            cbar_pos = (1.08,0.1,0.03,0.3), cbar_kws={'location':'left','label':f'{st} Z-score'});
            os.chdir(save_dir)
            [ax.savefig(f"{st}_{['left','right'][k]} cluster heatmap.png", dpi =300) if save_plot else None]

def Entropy_NSR_scatter_histogram(data,uni_ID, stimulus,state, chosen_stim, state_type, save_fig_text, color, save_plot):
    '''
    Input:
    data: 1-D array of stimulus responses
    uni_ID: 1-D string array of unique neural ID (animal ID + cell ID)
    stimulus: 1-D array of stimulus encountered
    state: 1-D array of the treatment type of corresponding trial
    state_type: a list of treatments indlucded in analysis (order specified)
    chosen_stim: a list of stimuli included in analysis (order specified)

    color: [color that fills the histogram, color of data points in the scatter plot]
    save_fig_text: Prefix for the filename of saved figure
    save_fig: Boolean to save the figure

    '''
    for st in state_type:
        norm_matrix = []
        st_indx = np.where(state == st)[0]
        stim = stimulus[st_indx]
        norm_sig = data[st_indx]
        total_s = []
        keep_cell = np.array([sum([np.where(np.logical_and(uni_ID[st_indx] == ci,stim == s))[0].size >0\
                         for s in chosen_stim])== len(chosen_stim) for ci in np.unique(uni_ID[st_indx])]) #Remove cells with missing value
        cell_in_keep = np.array([c in np.unique(uni_ID[st_indx])[keep_cell] for c in uni_ID[st_indx]])
        for s in chosen_stim:
            total_s.append(s)
            norm_matrix.append(norm_sig[np.where(np.logical_and(cell_in_keep,stim == s))[0]])
        # print(norm_matrix)
        norm_matrix = np.array(norm_matrix)
        entropy_dis = [entropy(norm_matrix[:,i]) for i in range(norm_matrix.shape[1])]
        nsr_dis = [noise_to_signal(norm_matrix[:,i]) for i in range(norm_matrix.shape[1])]
        
        #Plotting
        f, ax  = plt.subplots(ncols = 2, nrows = 2, figsize = (6,6))
        gs = gridspec.GridSpec(2,2,height_ratios=[2,5],width_ratios=[5,2],wspace=0.07, hspace=0.07)

        # # fit model to linear model
        # X = np.c_[np.ones_like(entropy_dis),entropy_dis]
        # b = inv(X.T@X)@X.T@nsr_dis

        #Scatter plot with regression line
        ax = plt.subplot(gs[1,0])
        scat_dat = {'Entropy': entropy_dis, 'Noise-to-signal Ratio':nsr_dis}
 
        ax.set(xlim= (0,1.01),ylim=(0,1.01))
        g = sns.regplot(x= 'Entropy',y='Noise-to-signal Ratio',data =pd.DataFrame(scat_dat),truncate = False,\
            scatter_kws = {'color': color[1]}, \
                line_kws = {'color': 'black','linestyle':'--'},\
                           # 'label': f'Linear fit: y = {round(b[1],3)}x {round(b[0],3)}'},\
                ax = ax,marker="+")
        # g.legend(frameon=False)
        g.spines['right'].set_visible(True)
        g.spines['top'].set_visible(True)

        #Distribution of entropy
        ax = plt.subplot(gs[0,0])
        ax.hist(entropy_dis, bins=np.arange(0,1.03, 0.02),alpha = 0.6,color = color[0],\
            histtype='stepfilled',edgecolor = 'black')
        ax.set(ylabel='Count',xlim =(0,1.01),xticks=[])
        ax.spines['left'].set_visible(False)

        #Distribution of NSR 
        ax = plt.subplot(gs[1,1])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
        ax.hist(nsr_dis, bins=np.arange(0,1.03, 0.02),alpha = 0.6, orientation="horizontal",\
            color = color[0],histtype='stepfilled',edgecolor = 'black')
        ax.set(xlabel='Count',ylim=(0,1.01),yticks=[])
        ax.spines['bottom'].set_visible(False)
        f.suptitle(f'{st} (total cell number: {len(entropy_dis)})\n {[s for s in chosen_stim]}\n',weight='bold')
        [f.savefig(f'{st} - {save_fig_text} Entropy-NSR distribution.png',dpi = 400) if save_plot else None]

#Old version of plotting functions that have been massively modified
###############################################################################################################

# def response_stack_bar(fig_ratio,res,stimulus,state,stim_types,state_type,bar_width, \
#                        bar_color, save_name,save_plot, save_dir, class_annot = ['EXT', 'INH', 'NR']):
    
#     f,ax = plt.subplots(nrows = int(len(stim_types)/4), ncols = 4,figsize = fig_ratio)
#     gs = gridspec.GridSpec(int(np.ceil(len(stim_types)/4)),4)
#     res_filt= []
#     indx = 0
#     for i,comb in enumerate(itertools.product(stim_types, state_type)):
#         ax = plt.subplot(gs[indx])
#         res_filt.append(list(res[np.where(np.logical_and(stimulus == comb[0], state == comb[1]))]))

#         if (i+1)%len(state_type) == 0 and i > 0: #All states for a given stimulus has ran through 

#             t_n = np.array([len(re) for re in res_filt]) #Total cell number under different states
#             bar_label = [stat+'\nn = '+str(num) for stat, num  in zip(state_type, t_n)] #State identity with corresponded cell number 
            
#             #Draw stack bar plot 
#             ax.bar(bar_label, ext:=100*np.array([res_filt[j].count(class_annot[0]) for j in range(len(state_type))])/t_n,\
#                  bar_width, label='Excited',color =bar_color[0],alpha = 1 )
#             ax.bar(bar_label,inh:=100*np.array([res_filt[j].count(class_annot[1]) for j in range(len(state_type))])/t_n,bar_width,\
#                  bottom=ext, label='Inhibited',color =bar_color[1])
#             ax.bar(bar_label, ns:=100*np.array([res_filt[j].count(class_annot[2]) for j in range(len(state_type))])/t_n,\
#                  bar_width, bottom=ext+inh, label='No response',color =bar_color[2])
#             ax.set_ylim(0,100)
#             ax.set_ylabel('Percentage of cell (%)')

#             if i == len(state_type)*len(stim_types)-1:
#                 ax.legend(bbox_to_anchor=(1,1), loc=2, frameon = False)#Show label only in the last graph 
#             #Statistics 
#             ref =np.array([ext[0],inh[0],ns[0]])
#             p_val = [scipy.stats.chisquare(np.array([ext[j+1],inh[j+1],ns[j+1]])*100/np.array([ext[j+1],inh[j+1],ns[j+1]]).sum(), f_exp=ref*100/ref.sum())[1] for j in range(len(state_type)-1)]
#             [ax.text(j,105,star_report(p_val[j-1]),horizontalalignment='center') for j in range(1,len(state_type)) ]

#             ax.set_title(f'{comb[0]}\n\n')
#             res_filt= []#Reset state 
#             indx+=1 

#     f.tight_layout()
#     os.chdir(save_dir)
#     [f.savefig(f'{save_name}responsive ratio stack bar.png',dpi = 600) if save_plot else None]

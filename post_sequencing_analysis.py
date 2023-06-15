import numpy as np, pandas as pd
import numpy.ma as ma
import os, re, glob, shutil
import functools, operator
import math
from matplotlib import pyplot as plt
import matplotlib



class FC: # (abbv. flowcell) This is a collection of standard values for experiments 
    options = {
        "channels":["G1","G2","R3","R4"],
        "lanes":[1,2,3,4],
        "colors":{
            "G1":"dodgerblue", "G2":"tab:orange", "R3":"crimson","R4":"darkorchid",
            "A":"dodgerblue","T":"tab:orange","G":"crimson","C":"darkorchid"
        },
        'expID':'',
        'normalize':False
    }
    
    # pattern to split tile_id into groups
    tile_pattern = re.compile(r"(?P<surf>[tb])L(?P<lane>\d)(?P<pos>\d{2})(?P<half>[A-Z])")


class BCQC:

    dtype = np.dtype( 
        [
            ("tile_no","<U6"),
            ("cycle_no",np.int32),
            ("Q30",np.float32),
            ("Q20",np.float32),
            ("medianQ",np.float32),
            ("n_clusters",np.int32),
            ("pf_rate",np.float32),
            ("err_r",np.float32),
            ("A_perc",np.float32),
            ("T_perc",np.float32),
            ("G_perc",np.float32),
            ("C_perc",np.float32),
            ("A_int",np.float32),
            ("T_int",np.float32),
            ("G_int",np.float32),
            ("C_int",np.float32)
        ]
    )

    int_col = {
        "G1":"A_int", "G2":"T_int", "R3":"G_int", "R4":"C_int"
    }
################################################################################
## UTILITY #####################################################################
################################################################################

def ATGC(char):
    '''Convert 1,2,3,4 or A,T,G,C into G1,G2,R3,R4 or G1,G2,R3,R4 into A,T,G,C''' 

    atgc = dict()
    bases = ['A','T','G','C']
    channels = ['G1','G2','R3','R4']
    nums = [0,1,2,3]

    tf = [char in i for i in [bases,channels,nums]]
    if not np.any(tf): 
        raise Exception("Not a valid base ID")
    
    keys = bases+nums+channels
    values = channels+channels+bases
    atgc = dict(zip(keys,values)) 
    
    return atgc[char]


def proc_to_cych(proc_num):
    # Convert proc_######-qc.csv digits to cycle & channel

    num = int(proc_num)
    channels = ["G1","G2","R3","R4"]

    return channels[num%4], num//4


def order_magnitude(num):
    return math.floor(math.log(num,10))


def change_qc_label(label):
    qc_dict = {
        "relative_intensity_mean":"RFL",
        "noise_mean":"Noise",
        "background_mean":"Background",
        "snr_mean":"SNR"
    }
    return qc_dict[label]


def split_channel(df):
    '''Input DataFrame, output dict with channels as keys'''
    
    unique_channel = df['channel'].unique()
    unique_channel = np.array(unique_channel.astype('str'))
    channel_dict = {}
    
    for i in unique_channel:
        query = "channel == '{}'".format(i)
        channel_dict[i] = df.query(query)
    
    return channel_dict

################################################################################
### METHODS TO COMPILE DATA ####################################################
################################################################################

def bcqc_read(bcqc_dir):
    '''
        Takes the original bcqc directory and compiles the separate 1_#_proc-int-bcqc.csv 
        files and compiles them into a single .csv file from a DataFrame. Returns 
        the created DataFrame.
    '''
    
    tf = re.search(r"/bcqc/([A-Za-z0-9_-]*)",bcqc_dir)
    
    read = (lambda x: "_"+x.group(1) if x else '')(tf)
     
    pathnames = glob.glob(bcqc_dir+f"/**/*_proc-int-bcqc.csv",recursive=True) # List all paths ending in .csv

    pattern = re.compile(r"([tb]L\d{3}[A-Z]).*1_(\d{1,2})_proc-int-bcqc.csv$") # Get tile and cycle information from paths 
    tile_cycle = [pattern.search(i) for i in pathnames]

    # Create two arrays corresponding to all pathnames

    tiles = np.array([i.group(1) for i in tile_cycle]) # array with tile names
    cycles = np.array([i.group(2) for i in tile_cycle],dtype=np.int32) # array with cycles

    # "tile_col" is the tile column to be put into the final array
    # "cyc" is the list of indices which point to the desired pathnames
    
    tile_col = [] 
    cyc = []

    for tile in np.unique(tiles): # build cyc and tile_col

        tf = np.where(tiles == tile,cycles,0) # Find the largest number for every tile in pathnames
        cyc.append(np.argmax(tf)) 
        tile_col += [tile for i in range(cycles[cyc[-1]]+1)]

    tile_col = np.array(tile_col)
    

    # Create the data structure for the final array

    dtype = BCQC.dtype

    filenames = np.asarray(pathnames)[cyc] # grab desired pathnames
    all_data = np.zeros([len(tile_col),],dtype=dtype) # initialize final array
    
    all_data['tile_no'] = tile_col # populate tile_no column

    linetext = [] # will contain all data read from original csv files

    for fn in filenames:
        
        linetext.append(np.loadtxt(fn,delimiter=',',skiprows=1))

    linetext = np.concatenate(linetext)

    for n, name in enumerate(dtype.names[1:]): # populate corresponding columns from original csv files
        all_data[name] = linetext[:,n]

    df = pd.DataFrame(all_data) # Convert to DataFrame and save as csv to working directory 
    df.to_csv(f"bcqc{read}.csv")

    return df


def fbqc_read(fbqc_dir):
    '''
        Takes the OLA directory of an experiment and compiles the separate proc_######-qc.csv
        files into a single .csv. Returns the resulting DataFrame.
    '''

    read = re.search(r"/([A-Za-z0-9_-]*)/OLA$",fbqc_dir)
    read = (lambda x: '-'+x.group(0) if x else '')(read)

    proc_dirs = glob.glob(fbqc_dir+"/**/*-qc.csv",recursive=True) 
    
    ola_pattern = re.compile(r"proc-([tb]L\d+[A-Z]).*/proc_(\d{6})-qc\.csv$")
    files = [ola_pattern.search(dirs) for dirs in proc_dirs if ola_pattern.search(dirs)]
    nums = [x.group(2) for x in files]
    tiles = [x.group(1) for x in files]
    print("number of filenames:",len(files))
    
    chcy = list(map(proc_to_cych,nums)) # Should return two lists
    chcy = np.array(chcy,dtype="<U6")
    cycle = list(chcy[:,1].astype(np.int32))
    channel = list(chcy[:,0].astype("<U2"))
    id_tuple = list(zip(tiles,cycle,channel))

    dtype2 = np.dtype(
        [
            ("tile_no","<U6"),
            ("cycle_no",np.int32),
            ("channel","<U2")
        ]
    )

    id_proc = np.array(id_tuple,dtype=dtype2)
    dtype = np.dtype(
        [
            ("fov_num",np.int32),
            ("n_features",np.int32),
            ("frac_good_features",np.float32),
            ("template_frac",np.float32),
            ("spot_frac",np.float32),
            ("size_mean",np.float32),
            ("snr_mean",np.float32),
            ("raw_intensity_mean",np.float32),
            ("relative_intensity_mean",np.float32),
            ("background_mean",np.float32),
            ("noise_mean",np.float32),
            ("image_focus_mean",np.float32),
            ("pixel_ct_mean",np.float32),
            ("packing_frac",np.float32),
            ("frac_couplets",np.float32),
            ("frac_max_shift_x",np.float32),
            ("frac_max_shift_y",np.float32)
        ]
    )


    nfiles = len(chcy)
    read_proc = np.empty([nfiles,],dtype=dtype)

    for i in range(nfiles):
        read_proc[i] = np.loadtxt(
            proc_dirs[i],
            dtype=dtype,
            skiprows=1,
            delimiter=',',
            usecols= list(range(1,7)) + list(range(8,22,2)) + list(range(22,26))
        )


    df1 = pd.DataFrame(read_proc)
    df2 = pd.DataFrame(id_proc)
    df = pd.concat([df2,df1],axis=1)
    
    df.to_csv(f"fbqc{read}.csv")

    return df


def bcqc_copy(bcqc_dir):

    '''
        Copy files from original bcqc directory to a new subdirectory within the
        current working directory
    '''

    tf = re.search(r"/bcqc/([A-Za-z0-9_-]*)",bcqc_dir)
    read = (lambda x: "_"+x.group(1) if x else '')(tf) 

    new_path = f"./bcqc_csv{read}" 
    isExist = os.path.exists(new_path)

    if not isExist:
        os.mkdir(new_path)
     
    pathnames = glob.glob(bcqc_dir+f"/*/*.csv")

    
    pattern = re.compile(r"([tb]L\d{3}[A-Z]).*1_(\d{1,2})_proc-int-bcqc.csv$") # Get tile and cycle information from paths 
    tile_cycle = [pattern.search(i) for i in pathnames]

    # Create two arrays corresponding to all pathnames

    tiles = np.array([i.group(1) for i in tile_cycle]) # array with tile names
    cycles = np.array([i.group(2) for i in tile_cycle],dtype=np.int32) # array with cycle

    # "cyc" is the list of indices which point to the desired pathnames
    
    cyc = []

    for tile in np.unique(tiles): # build cyc and tile_col

        tf = np.where(tiles == tile,cycles,0) # Find the largest number for every tile in pathnames
        cyc.append(np.argmax(tf)) 

    filenames = np.asarray(pathnames)[cyc] # grab desired pathnames
    basenames = [re.search(r"([tb]L\d{3}[A-Z])",i).group(1) + "_" + os.path.basename(i) for i in filenames]

    
    for path,name in zip(filenames,basenames):
        
        shutil.copy(path,os.path.join(new_path,name))


def fbqc_copy(fbqc_dir): 
    '''
        Copy proc_######-qc.csv files from OLA directory into new tile subdirectories
        created in the current working directory
    '''
    tf = re.search(r"/([A-Za-z0-9_-]*)/OLA$",fbqc_dir)
    
    read = (lambda x: '_'+x.group(1) if x else '')(tf)

    new_path = f"./fbqc_csv{read}"

    isExist = os.path.exists(new_path)
    if not isExist: 
        os.mkdir(new_path)

    proc_dirs = glob.glob(fbqc_dir+"/*/qc/*.csv")

    pattern = re.compile(r"./proc-([tb]L\d+[A-Z])/qc/proc_(\d{6})-qc\.csv$")
    
    files = [pattern.search(dirs) for dirs in proc_dirs if pattern.search(dirs)]
    tiles = [file.group(1) for file in files]

    for fn,file in enumerate(files):

        path = file.string
        newfolder = f"proc-{tiles[fn]}"
        new_qc_dir = new_path+f"/{newfolder}" # one down
        isExist = os.path.exists(new_qc_dir)
        
        if not isExist:
            os.mkdir(new_qc_dir)

        shutil.copyfile(path,new_qc_dir+f"/{os.path.basename(path)}")


################################################################################
## GRAPHING FUNCTIONS ##########################################################
################################################################################

def bcqc_mean_rfl_bylane(df,**kwargs):
    '''
        Accepts a Pandas DataFrame of bcqc data (contains A_int,T_ int, etc.) and outputs 
        graphs of relative_intensity_mean averaged across the lane vs. cycle, producing 
        one graph for each lane.

        Possible keyword arguments:
            expID - name of experiment
            normalize - one of 'G1','G2','R3','R4' or False
            colors - colors to associate with each of the four channels
            channels - list of channels to include, default ['G1','G2','R3','R4']
            lanes - list of lanes to include, default [1,2,3,4]
        
    '''

    options = FC.options.copy() # insert keyword options from FC class 
    options.update(kwargs)
    
    channels = options["channels"] # assign names for brevity
    lanes = options["lanes"]
    c = options["colors"]


    fig,ax = plt.subplots(1,len(lanes),figsize=(8,3))
    plt.subplots_adjust(wspace=0.5,top=0.7,bottom=0.22,right=0.8)
    if len(lanes) == 1:
        ax = np.expand_dims(ax,axis=0)

    ax[0].set_ylabel("RFL [-]")



    for ch in channels:

        for ln in lanes:

            # match first digit to get mask corresponding to tile in a specific lane
            pattern = r"[tb]L{}\d+[A-Z]".format(ln)
            mask1 = df.tile_no.str.contains(pattern) 

            one_lane = df[mask1]
            
            ydat = one_lane.groupby("cycle_no").mean()[BCQC.int_col[ch]]
            xdat = ydat.index

            if options['normalize']: # normalize by specified channel
                ynorm = one_lane.groupby("cycle_no").mean()[BCQC.int_col[options['normalize']]]
            else:
                ynorm = 1

            ax[ln-1].plot(xdat,ydat/ynorm,label=ch,color=c[ch])    
            ax[ln-1].set_title(f"Lane {ln}")
            ax[ln-1].set_xlabel(f"Cycle #")

    ax[-1].legend(bbox_to_anchor=[1.2,0.9])
    title_tag = (lambda x: x+": " if x else '')(options['expID'])
    norm_tag = (lambda x: f"\n Normalized by {x} channel" if x else '')(options['normalize'])
    fig.suptitle(title_tag+"(bcqc) Average RFL across each lane"+norm_tag)
    plt.show()

    return fig,ax


def fbqc_heatmap(df,Y,chosen_cycles='auto',**kwargs): 
    
    '''
        Given a Pandas DataFrame of fbqc data (df) and the name of a qc value (Y), 
        create a grid of heatmaps organized by channel and chosen cycles. Plots 
        all lanes & their halves included in the dataset and cannot be changed.

        Possible keyword arguments:
            expID - name of experiment
            colors - colors to associate with each of the four channels
            channels - list of channels to include, default ['G1','G2','R3','R4']
            surface - the top or bottom surface of the FC 't' or 'b', default 't'            
    '''

    def lane_box(lanes):
        # Print a text box that IDs each row of the heatmap to a location on the FC
        text = '\n'.join(lanes)
        plt.figtext(0.88,0.5,text,bbox=dict(facecolor="none",edgecolor="gray",pad=6))

    options = FC.options.copy()    # load in defaults from FC class
    options.update({'surface':'t'}) # add new "surface" keyword
    options.update(kwargs)  # update based on user input
     
    tb = options['surface']     # assign names for brevity
    channels = options['channels']

    if type(chosen_cycles) == str and chosen_cycles == 'auto': # by default graphs 8 cycles between the first and last cycle
        max_cycle = df.cycle_no.max()
        min_cycle = df.cycle_no.min()
        chosen_cycles = np.linspace(min_cycle,max_cycle,8).astype(int)

    elif type(chosen_cycles) == str and chosen_cycles != 'auto':
        raise Exception("'chosen_cycles' keyword must be an iterable of integers or 'auto'")
    
    nrows = len(chosen_cycles)
    ncols = len(channels)
    
    tb_label = {"t":"top", "b":"bottom"} 
    mask_tb = df["tile_no"].str.contains(tb)
    df = df.loc[mask_tb]
    
    lane_pattern = tb+r"L(\d)(\d{2})([A-Z])"
    lane_match = df.tile_no.str.extract(lane_pattern)

    
    lanes = lane_match[0].astype(np.int32).unique()
    tpos = lane_match[1].astype(np.int32).max()
    tset = set(lane_match[1])

    half = lane_match[2].astype(str).unique()
    half.sort()
    half = np.flip(half)

    tile_order = []
    lanes.sort()
    lanes = np.flip(lanes)

    for i in lanes:
        tile_order += [f"{i}{j}" for j in half]
    
    

    fig, ax = plt.subplots(nrows,ncols)

    if nrows == 1:
        ax = np.expand_dims(ax,axis=0)
    if ncols == 1:
        ax = np.expand_dims(ax,axis=1)
       
    fig.subplots_adjust(left=0.15,right=0.85,top=1)
    cmap = matplotlib.cm.get_cmap('jet').copy()
    cmap.set_bad('white') # color tiles with missing data white

    # fig.subplots_adjust(top=0.5,hspace=0.1)

    qc_dict = dict()

    for c, ch in enumerate(channels):

        imin, imax = [],[]        
        
        ax[0,c].set_title(ch)
        qc_dict[ch] = []

        for y, cy in enumerate(chosen_cycles):

            qry = "channel == '{}' & cycle_no == {}".format(ch,cy)    
            df_qry = df.query(qry)
            tile_df = df_qry.tile_no.str.extract(FC.tile_pattern)

            # This breaks if a tile in the middle is completely missing from the dataset
            im_array = np.empty([len(half)*len(lanes),tpos],dtype=float) 
            ax[y,0].set_ylabel(f"Inc {cy}",rotation='horizontal',labelpad=18,fontsize=8)
            
            for t,ln in enumerate(tile_order):
                
                tf1 = tile_df.surf == tb
                tf2 = tile_df.lane == ln[0]
                tf3 = tile_df.half == ln[1]

                tf = tf1*tf2*tf3 

                if not np.any(tf): # if empty, print tile, channel, and cycle 
                    print(tb+"L"+ln[0]+"##"+ln[1],qry)
                
                zdat = df_qry.loc[tf].sort_values('tile_no')[Y].to_numpy() # Sort values just in case tiles are out of order

                if len(zdat) != tpos:
                    # include placeholder values for missing tile data 
                    zset = set(tile_df.pos[tf])
                    disjoint = tset.difference(zset)
                    disjoint = list(disjoint)
                    disjoint.sort()

                    for dis in disjoint:
                        index = int(dis)-1
                        zdat = np.insert(zdat,index,-1)


                im_array[t,:] = zdat
                
                
                ax[y,c].set_yticks([])
                if cy != chosen_cycles[-1]:
                    ax[y,c].set_xticks([])
                
            im_array = np.ma.array(im_array,mask=im_array==-1)
            qc_dict[ch].append(im_array)
            imax.append(np.max(im_array))
            imin.append(np.min(im_array))

        xticks = np.linspace(0,tpos,6,dtype=np.int32)
        ax[y,c].set_xticks(xticks)
        ax[y,c].set_xticklabels(xticks+1,fontsize=8)
        
        ch_min, ch_max = np.min(imin), np.max(imax)
        qc_dict[ch] = np.array(qc_dict[ch])
        
        for j in np.arange(len(chosen_cycles)):
            im_scale = ax[j,c].imshow(
                np.ma.array(qc_dict[ch][j],mask=qc_dict[ch][j]==-1),
                vmin = ch_min,
                vmax = ch_max,
                cmap = cmap,
                # extent = [0,tpos-1,0,2*len(lanes)-1],
                aspect = 'equal',
                interpolation = 'nearest'
            )
            

            '''Find order of magnitude, then scale accordingly'''
            oom_min = order_magnitude(ch_min)
            oom_max = order_magnitude(ch_max)

            mintick = math.ceil(ch_min/10**(oom_min-2))*10**(oom_min-2)
            maxtick = math.floor(ch_max/10**(oom_max-2))*10**(oom_max-2)
            
        cb2 = fig.colorbar(
            im_scale,
            ax=ax[:,c],
            location='top',
            pad = 0.1,
            shrink=0.8,
            ticks=[mintick,maxtick]
        )

        cb2.ax.tick_params(labelsize=6)
    title_tag = (lambda x: x+': ' if x else '')(options['expID'])
    fig.suptitle(f"{title_tag}{change_qc_label(Y)} ({tb_label[tb]})")
    lane_box(tile_order)
        
    return fig, ax, qc_dict


def fbqc_plot_lines(df,Y,**kwargs): 
    '''
        Given a Pandas DataFrame of fbqc data (df) and the name of a qc value (Y), 
        create graphs which plot Y vs. cycle for individual tiles. Each graph is 
        organized in a grid by lane number and channel.

        Possible keyword arguments:
            expID - name of experiment
            normalize - one of 'G1','G2','R3','R4' or False
            colors - colors to associate with each of the four channels
            channels - list of channels to include, default ['G1','G2','R3','R4']
            lanes - list of lanes to include, default [1,2,3,4]
            surface - the top or bottom surface of the FC 't' or 'b', default 't'
            tiles - takes up to six positions along the lane 
            half - include A, B, C, etc. half, or all
    '''
    
    options = FC.options.copy() 
    match_half = df.tile_no.str.extract(r".*([A-Z]{1})$")

    options.update(
        {
            'surface':'t',
            'tiles':["02","08","15","21"],
            'half':list(match_half[0].unique())
        }
    )

    options.update(kwargs)
    
    surface = options['surface'] # assign new names for brevity
    tiles = options['tiles']
    half = options['half']
    lanes = options['lanes']

    channel_dict = split_channel(df)
    channel_keys = channel_dict.keys()

    nrows, ncols = len(lanes), df["channel"].nunique()
    fig, ax = plt.subplots(nrows,ncols,figsize=(12,3+1.5*nrows))
    
    
    fig.subplots_adjust(wspace=0.3,top=0.60+0.05*nrows,hspace=0.35)
    colors = ['tab:blue','tab:orange','tab:green','crimson','mediumorchid','mediumturquoise']        
    handle,label = [],[]
    
    if len(ax.shape) == 1:
        ax = np.expand_dims(ax,0)
        
    
    for c,ch in enumerate(channel_keys):

        lane_k = 0
        df_ch = channel_dict[ch]
        ylim = (np.min(df_ch[Y]),np.max(df_ch[Y]))
        
        if re.match(r"[GR][1234]",ch): ch_tag = ch
        else: ch_tag = ATGC(ch)

        ax[0,c].set_title(ch_tag,fontsize=14)
        for ln in lanes:
            if c == 0:
                ax[lane_k,c].set_ylabel(f"Lane {ln}",fontsize=14)
            if ln == lanes[-1]:
                ax[lane_k,c].set_xlabel(f"Cycle #",fontsize=14)


            tile_labels = [
                [
                    surface+"L"+str(ln)+"{}{}".format(i,k) for i in tiles
                ] 
                for k in half
            ]
            ax[lane_k,c].ticklabel_format(axis='y',scilimits=[-4,5])
            ax[lane_k,c].set_ylim(ylim)
            tile_labels = functools.reduce(operator.iconcat,tile_labels,[])

            for ti,tlabel in enumerate(tile_labels):

                single_tile = df_ch["tile_no"] == tlabel
                xdat = df_ch.loc[single_tile]["cycle_no"]
                ydat = df_ch.loc[single_tile][Y]
                
                h = ax[lane_k,c].plot(
                    xdat,
                    ydat,
                    linestyle=['-','--',':','-.'][ti//len(tiles)],
                    color=colors[ti%len(tiles)],
                    # linewidth=2
                )[0]

                if c == 0 and ln == lanes[0]:
                    handle.append(h)
                    label.append(
                        f"{surface}L#{tiles[ti%len(tiles)]}{half[ti//len(tiles)]}"
                    )
            lane_k += 1
        
        title_tag = (lambda x: x+': ' if x else '')(options['expID'])
        fig.legend(handle,label,ncol=len(half),bbox_to_anchor=[0.9,1],title="Tile ID")
        fig.suptitle(f"{title_tag}{change_qc_label(Y)}",fontsize=16,x=0.35,y=0.93,fontweight="bold")
        
    return fig, ax

def fbqc_mean_lane(df,Y,**kwargs):

    '''
        Takes a Pandas DataFrame of fbqc data and outputs graphs of 
        relative_intensity_mean averaged across the lane, producing one graph
        for each lane.

        Possible keyword arguments:
            expID - name of experiment
            normalize - one of 'G1','G2','R3','R4' or False, default False
            colors - colors to associate with each of the four channels
            channels - list of channels to include, default ['G1','G2','R3','R4']
            lanes - list of lanes to include, default [1,2,3,4]

    '''

    options = FC.options.copy()
    options.update(kwargs)

    c = options['colors'] # assign name for brevity
            
    fig,ax = plt.subplots(1,len(options['lanes']),figsize=(8,2))
    plt.subplots_adjust(wspace=0.5,top=0.7,bottom=0.22,right=0.8)
    
    if len(options['lanes']) == 1:
        ax = np.expand_dims(ax,axis=0)

    for ch in options['channels']:

        for ln in options['lanes']:

            pattern = r"[tb]L{}\d+[AB]".format(ln)
            mask1 = df.tile_no.str.contains(pattern)
            mask2 = df.channel == ch
            mask3 = df.channel == options['normalize']

            one_lane = df[mask1*mask2]
            norm = df[mask1*mask3]
            
            ydat = one_lane.groupby("cycle_no").mean()[Y]
            ynorm = (lambda x: norm.groupby("cycle_no").mean()[Y] if x else 1)(options['normalize'])
            
            xdat = ydat.index

            ax[ln-1].plot(xdat,ydat/ynorm,label=ch,color=c[ch])
            ax[ln-1].set_title(f"Lane {ln}")
            ax[ln-1].set_xlabel(f"Cycle #")

    ax[-1].legend(bbox_to_anchor=[1.2,0.9])
    ax[0].set_ylabel(f"{change_qc_label(Y)} [-]")

    norm_tag = (lambda x: f", normalized by {ATGC(x)}" if x else '')(options['normalize'])
    fig.suptitle(f"(fbqc) Average {change_qc_label(Y)} across each lane"+norm_tag)

    return fig, ax

def fbqc_by_tile(df,Y,chosen_tiles=['03A','09B','17A','20B'],**kwargs):
    # Plot by chosen tiles, plot values from these positions for each lane
    # Input the last 3 characters from the tile ID

    options = FC.options.copy()
    options.update({'surface':'t'}) 
    options.update(kwargs)

    ncols = len(chosen_tiles)
    
    lanes = df.tile_no.str.extract(r"[tb]L(\d)\d{2}[A-Z]$")
    lanes = lanes[0].astype(np.int32).unique()
    nrows = len(lanes)

    fig,ax = plt.subplots(nrows,ncols,figsize=(15,2+nrows*2))
    ax = np.asarray(ax)
    if len(ax.shape) == 1:
        ax = np.expand_dims(ax,axis=0)
    
    tb = options['surface']
    tb_tag = {'t':"top",'b':"bottom"}

    c = options['colors']

    chosen_tiles = [tb+'L{}'+t for t in chosen_tiles]

    fig.subplots_adjust(wspace=0.3,hspace=0.5,top=0.80)
    title_tag = (lambda x: x['expID']+': ' if x['expID'] else '')(options)
    norm_tag = (lambda x: f", normalized by {x['normalize']} channel" if x['normalize'] else '')(options)

    fig.suptitle(
        f"{title_tag}{change_qc_label(Y)} from select tiles ({tb_tag[tb]}){norm_tag}",
        fontsize=18, fontweight="bold", x=0.2, y=0.96, horizontalalignment="left"
    )
 

    for lane in lanes:

        for x,axis in enumerate(ax[lane-1]):

            picktiles = chosen_tiles[x].format(lane)

            for ch in options['channels']:

                mask1 = (f"tile_no == '{picktiles}' & channel == '{ch}'")
                df1 = df.query(mask1)

                if options['normalize']:

                    maskn = (f"tile_no == '{picktiles}' & channel == '{options['normalize']}'")
                    ynorm = (lambda x: df.query(maskn)[Y] if x['normalize'] else 1)(options)
                
                else: 
                    ynorm = 1


                axis.plot(df1.cycle_no, df1[Y]/ynorm.to_numpy(), linestyle='-', color=c[ch], label=ch)
                axis.set_xticks(np.linspace(df1.cycle_no.min(),df1.cycle_no.max(),num=5).astype(np.int32))
                
                axis.set_title(picktiles,pad=8)

                if lane == lanes[-1]:
                    axis.set_xlabel("Cycle #")

                if lane == lanes[0] and axis == ax[0,-1]:
                    axis.legend(ncol=len(options['channels']),bbox_to_anchor=(1.1,1.6))
                
                if axis == ax[lane-1,0]:
                    axis.set_ylabel(f"Lane {lane} {change_qc_label(Y)}",fontsize=14,labelpad=10)
    
    ax[0][-1].legend(bbox_to_anchor=[1.2,0.9])
    
    return fig, ax

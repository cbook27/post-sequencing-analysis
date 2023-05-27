import numpy as np, pandas as pd
import os, re, glob
import shutil
from matplotlib import pyplot as plt

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

def bcqc_direct_read(bcqc_dir,read="Read1"):
    
    # Read files directly from original bcqc directory and output a csv file 
    # with all bcqc data combined. Returns the created DataFrame

    pathnames = glob.glob(bcqc_dir+f"\\{read}\\*\\*.csv") # List all paths ending in .csv

    pattern = re.compile(r"([tb]L\d{3}[AB]).*1_(\d{1,2})_proc-int-bcqc.csv$") # Get tile and cycle information from paths 
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
    df.to_csv(f"bcqc-{read}.csv")

    return df



def bcqc_copy(bcqc_dir,read="Read1"):

    # Copy files from original bcqc directory 

    new_path = f".\\bcqc_csv_{read}" 
    isExist = os.path.exists(new_path)
    
    if not isExist:
        os.mkdir(new_path)

    pathnames = glob.glob(bcqc_dir+f"\\{read}\\*\\*.csv") # List all paths ending in .csv

    pattern = re.compile(r"([tb]L\d{3}[AB]).*1_(\d{1,2})_proc-int-bcqc.csv$") # Get tile and cycle information from paths 
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
    basenames = [re.search(r"([tb]L\d{3}[AB])",i).group(1) + "_" + os.path.basename(i) for i in filenames]

    
    for path,name in zip(filenames,basenames):
        
        shutil.copy(path,os.path.join(new_path,name))



def bcqc_read(bcqc_csv,read=''):

    filenames = glob.glob(f".\\{bcqc_csv}\\*.csv")

    read_match = re.search(r"(Read1)|(Index1)",bcqc_csv)
    
    if not read and read_match:
        read = read_match.group(1)

    dtype = BCQC.dtype

    tile_col = [] # will contain tile names for new array
    linetext = [] # will contain all data read from original csv files
    pattern = re.compile(r"([tb]L\d{3}[AB])_1_(\d+)_proc-int-bcqc.csv")

    for fn in filenames:

        
        tile,cycles = pattern.match(os.path.basename(fn)).group(1,2)
        tile_col += [tile for i in range(int(cycles)+1)]

        linetext.append(np.loadtxt(fn,delimiter=',',skiprows=1))

    linetext = np.concatenate(linetext)

    all_data = np.zeros([len(tile_col),],dtype=dtype) # initialize final array
    
    all_data['tile_no'] = tile_col # populate tile_no column

    for n, name in enumerate(dtype.names[1:]): # populate corresponding columns from original csv files
        all_data[name] = linetext[:,n]

    df = pd.DataFrame(all_data) # Convert to DataFrame and save as csv to working directory 
    df.to_csv(f"bcqc-{read}.csv")

    return df

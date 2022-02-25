# -*- coding: utf-8 -*-
"""
PDMAC (Repurposed)
Created on 02/09/2022

Vendor 1

The objective of this tool is to run through all of the .csv files in a folder, 
identify the scf_database.xlsx file, and prepare the data input files to use in
the ML model.
"""
import pandas as pd
import numpy as np
import math
import time
import os
# Data Smoothing Modules
from scipy.interpolate import splrep, splev
from scipy.signal import savgol_filter
# Plotting Modules
import matplotlib.pyplot as plt

# Variable nomenclature
# dr - dent registry
# rd - raw data
# sd - smooth data
# pd - image data
# rd_axial  - [inches]  Axial relative position from start value
# rd_circ   - [degrees] Circumferential position
# rd_radius - [inches]  Radius difference from nominal internal radius

# Vendor 1
# Data is all in Column A with delimiter ';' with 12 header rows that need to
# be removed before accessing the actual data
rd_axial_row = 12  # Row 13 = Index 12
rd_drop_tail = 2
dr_header_row = 1  # Row 2  = Index 1

# Output Data size
axial_len = 200
circ_len = 200
# Data Smoothing
data_smoothing_bool = True

time_start = time.time()
raw_data_path = 'raw_data'
processed_data_path = 'processed_data'
training_data_path = 'training_data'

print('========== START ==========')
print('%03d | Program began execution.' % (time.time() - time_start))

# =============================================================================
# FUNCTIONS
# =============================================================================

def collect_raw_data_v1(rd_path):
    rd = pd.read_csv(rd_path, header=None)
    # Drop the first set of rows that are not being used
    rd.drop(rd.head(rd_axial_row).index, inplace=True)
    # Drop the last two rows that are not being used
    rd.drop(rd.tail(rd_drop_tail).index, inplace=True)
    rd = rd[0].str.split(';', expand=True)
    rd = rd.apply(lambda x: x.str.strip())
    # Drop the last column since it is empty
    rd.drop(rd.columns[-1], axis=1, inplace=True)
    # Relative axial positioning values
    rd_axial = rd.loc[rd_axial_row].to_numpy()
    # Delete the first two values which are 'Offset' and '(ft)'
    rd_axial = np.delete(rd_axial, [0,1])
    rd_axial = rd_axial.astype(float)
    # Convert the axial values to inches
    rd_axial = rd_axial*12
    # Drop the two top rows: Offset and Radius
    rd.drop(rd.head(2).index, inplace=True)
    # Circumferential positioning in [degrees]
    rd_circ = rd[0].to_numpy()
    # Convert from clock to degrees
    rd_circ = [x.split(':') for x in rd_circ]
    rd_circ = [round((float(x[0]) + float(x[1])/60)*360/12,1) for x in rd_circ]
    rd_circ = np.array(rd_circ)
    # Drop the two first columns: Circumferential in o'Clock and in Length inches
    rd.drop(rd.columns[[0,1]], axis=1, inplace=True)
    rd_radius = rd.to_numpy().astype(float)
    return rd_axial, rd_circ, rd_radius

def collect_dent_registry_v1(dr, rd_DentRef):
    # Find the row with the matching rd_DentRef
    # dr_row = dr.loc[dr[dr.columns[1]] == rd_DentRef]
    dr_row = dr.loc[dr['Dent Ref #'] == rd_DentRef]
    # Outside Diameter (inches)
    dr_OD = dr_row['Outside Diameter (inches)'].to_numpy()[0]
    # Collect the Wall Thickness value (inches)
    dr_WT = dr_row['Wall Thickness (inches)'].to_numpy()[0]
    # Collect the SCF value
    dr_SCF = dr_row['SCF (OD)'].to_numpy()[0]
    return dr_OD, dr_WT, dr_SCF

def data_smoothing(rd_axial, rd_circ, rd_radius, dr_OD):
    # print('%03d | ===== DATA SMOOTHING =======' % (time.time() - time_start))
    
    # Check the axial spacing
    rd_axial_d = 0
    for i in range(1,len(rd_axial)):
        rd_axial_d += rd_axial[i] - rd_axial[i-1]
    rd_axial_d = rd_axial_d/i
    
    sd_OR = dr_OD/2   # nominal radius from ASME B31.8
    cwindow = 5       # smoothing window (#points) for circumferential smoothing filter - must be an odd number (Default = 5)
    csmooth = 0.0015  # circumferential smoothing parameter for splines (default 0.0015)
    axwindow = 83     # smoothing window (#points) for axial smoothing filter
    asmooth = 0.01    # axial smoothing parameter for splines
    
    #smoothed file output intervals (for ABAQUS, etc.)
    out_ax = 0.5    # axial length
    out_circ = 0.5  # circumferential length
    
    calwt = np.ones(rd_circ.shape[0])
    axwt = np.ones(rd_axial.shape[0])
    
    sd_axial    = rd_axial.copy()
    sd_radius   = rd_radius.copy()
    sd_circ_rad = np.deg2rad(rd_circ)
    
    pipe_cpts = np.zeros(rd_radius.shape)
    pipe_apts = np.zeros(rd_radius.shape)
    
    # Step 1 - Circumferential Profiles Smoothing and Spline Functions
    for axpos, rad in enumerate(sd_radius[0,:]):
        rad=sd_radius[:,axpos] #isolate current 
        circ_filt = savgol_filter(rad, cwindow, 3, mode='wrap') #filter circumferential section
        circ_spl = splrep(sd_circ_rad, circ_filt, w=calwt, k=3, s=csmooth, per=1) #find tuples for current circumferential section
        circ_pts = splev(sd_circ_rad, circ_spl) #evalutate splines for current cross section
        pipe_cpts[:,axpos] = circ_pts #Variable to store all pipe points based on circumferential sections
        

    # Step 2 - Axial Profiles Smoothing and Spline Functions
    for cal, traces in enumerate(sd_radius[:,0]):
        traces=sd_radius[cal,:] #isolate current axial section
        ax_filt = savgol_filter(traces, axwindow, 3) #smooth current axial section
        ax_spl = splrep(sd_axial, ax_filt, w=axwt, k=3, s=asmooth) #calcuate spline coefficients for current axial location
        ax_pts = splev(sd_axial, ax_spl) #calculate axial points
        pipe_apts[cal,:] = ax_pts #Variable to store all pipe points based on axial cross sections      


    # Step 3 - Create Weighted Average Profile from Axial and Circumfernetial Profiles
    err_circ = abs(pipe_cpts - sd_radius) #calcualtes the variation for the circumferential profiles
    err_axial = abs(pipe_apts - sd_radius) #calculates the variation for the axial profiles
    pipe_avgpts = (err_circ * pipe_cpts + err_axial * pipe_apts) / (err_circ + err_axial) #Average Weighted Positions based on magnitude of variations, greater the variation the higher the weight

    #---------------Create Soothed Output Data on Requested Interval from Pipe_Avgpts----------------
    #---------------Evaluate Strain on Desired Interval from Pipe Avgpts-----------------------------

    c_out_pts = math.ceil((math.pi*2.0*sd_OR / out_circ) / 4) * 4 #find circumferential interval closest to out_circ length on a multiple of four
    circ_out_int = np.linspace(0, 2*math.pi, c_out_pts, False) #create circumferential interval of radians
    pipe_out=np.empty((len(circ_out_int), len(sd_axial)),) #create empty array to store variables

    a_out_pts = int(max(sd_axial) / out_ax)
    ax_out_int = np.linspace(0, round(max(sd_axial),0), a_out_pts, False) #create equally spaced axial points baed on a_ax_pts for smoothing
    f_pipe_out=np.empty((len(circ_out_int), len(ax_out_int)),) #create an empty array to store radii for smoothing


    # Step 4 - Calculate Final Profiles - Axial First
    for axpos, rad in enumerate(pipe_avgpts[0,:]):
        rad=pipe_avgpts[:,axpos] #isolate current cirumferential section
        out_spl = splrep(sd_circ_rad, rad, k=3, s=csmooth, per=1) #claculate interpolating spline
        out_pts = splev(circ_out_int, out_spl) #output circumferential sections on requested interval for smoothed output
        pipe_out[:,axpos] = out_pts #Variable to store all smoothed pipe points based on circumferential cross sections

    for cal, traces in enumerate(pipe_out[:,0]):
        traces=pipe_out[cal,:] #isolate current axial trace
        ax_filt = savgol_filter(traces, axwindow, 3) #smooth current axial section
        f_out_spl = splrep(sd_axial, ax_filt, k=3, s=asmooth) #calcualte axial splines
        ax_pts = splev(ax_out_int, f_out_spl)
        f_pipe_out [cal,:] = ax_pts #final output points for smoothed file

    theta_f  = np.rad2deg(circ_out_int)
    z_f      = ax_out_int
    radius_f = f_pipe_out
    
    return z_f, theta_f, radius_f

def data_to_image(sd_radius, dr_IR, axial_len=100, circ_len=100):
    # Starting from the center of the dent, collect a perimeter
    # Output data must be in (axial_len, circ_len) dimensions
    # axial_len = 100
    # circ_len = 100
    
    # Location of the lowest point
    min_ind = np.unravel_index(np.argmin(sd_radius), sd_radius.shape)
    # Copy and paste from the center to the limits
    axial_LB = min_ind[0] - int(axial_len/2)
    axial_UB = min_ind[0] + int(axial_len/2)
    circ_LB  = min_ind[1] - int(circ_len/2)
    circ_UB  = min_ind[1] + int(circ_len/2)
    
    if axial_LB < 0: axial_LB = 0
    if circ_LB < 0: circ_LB = 0
    
    pd_radius = sd_radius[axial_LB:axial_UB, circ_LB:circ_UB]

    # Normalize the Radial values
    # Since the input data is already the difference form the nominal
    # radius, then no need to subtract from dr_IR
    pd_radius = pd_radius/dr_IR
    
    # If sd_radius was below the desired dimensions, then add empty columns
    if pd_radius.shape[1] < axial_len:
        while pd_radius.shape[1] < axial_len:
            # Add column on both sides at the same time
            pd_radius = np.hstack((pd_radius, np.zeros([pd_radius.shape[0],1])))
            pd_radius = np.hstack((np.zeros([pd_radius.shape[0],1]), pd_radius))
        # Check if the dimensions match now
        if pd_radius.shape[1] > axial_len:
            pd_radius = np.delete(pd_radius, 0, 1)
    if pd_radius.shape[0] < circ_len:
        while pd_radius.shape[0] < circ_len:
            # Add row on both sides at the same time
            pd_radius = np.vstack((pd_radius, np.zeros([1,pd_radius.shape[1]])))
            pd_radius = np.vstack((np.zeros([1,pd_radius.shape[1]]), pd_radius))
        # Check if the dimensions match now
        if pd_radius.shape[0] > circ_len:
            pd_radius = np.delete(pd_radius, 0, 0)
            
    # Print image to display resultant dent shape
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(pd_radius, cmap=plt.cm.binary)
    plt.xlabel(str(rd_DentRef))
    plt.colorbar()
    plt.grid(False)
    plt.show()
    
    return pd_radius
    
# =============================================================================
# ITERATE THROUGH ALL ILI DATA FILES
# =============================================================================

for subdir, dirs, files in os.walk(raw_data_path):
    # Skip the first iteration
    if not '\\' in subdir:
        continue
    
    # Ignore folder titled 'ignore'
    if 'ignore' in subdir.lower():
        continue
    # Keep track of time for each subdir folder
    time_subdir_start = time.time()
    print('========== SUBDIR START ==========')
    print('%03d | Began processing subdir: %s' % (time.time() - time_start, subdir))
    
    # Find the corresponding dent registry
    for file in files:
        if not file.find('dent_registry') == -1:
            print('%03d | Found dent registry file.' % (time.time() - time_start))
            dr_path = os.path.join(subdir, file)
            dr = pd.read_excel(dr_path, sheet_name=0, header=dr_header_row)
            
    # Continue looping through all of the other data files
    rd_first_file = True
    for file in files:
        if not file.find('.csv') == -1:
            print('%03d | Began processing file: %s' % (time.time() - time_start, file))
            # File name information
            rd_DentRef = file.split('.csv')[0].split('_')
            # This removes the first three items assuming that they are line information
            del rd_DentRef[0:3]
            # Determine the number of dents. If more than one, SKIP
            if len(rd_DentRef) > 1:
                continue
            rd_DentRef = int(rd_DentRef[0])
            
            rd_path = os.path.join(subdir, file)
            # Load the raw data information
            rd_axial, rd_circ, rd_radius = collect_raw_data_v1(rd_path)
            
            # Load the information from the dent registry
            dr_OD, dr_WT, dr_SCF = collect_dent_registry_v1(dr, rd_DentRef)
            
            dr_IR = dr_OD/2 - dr_WT
            
            # Perform data smoothing on the raw data
            if data_smoothing_bool == True:
                sd_axial, sd_circ, sd_radius = data_smoothing(rd_axial, rd_circ, rd_radius, dr_OD)
            else:
                sd_axial  = rd_axial.copy()
                sd_circ   = rd_circ.copy()
                sd_radius = rd_radius.copy()
            
            # Output the dent shape in pixel format
            pd_radius = data_to_image(sd_radius, dr_IR, axial_len, circ_len)
            
            # Save the pd_radius into a new .csv file for future reference
            pd_folder = subdir.split(os.path.sep)[-1].replace(" ", "_")
            # Add label to keep track instances with data smoothing
            if data_smoothing_bool == True:
                pd_folder = pd_folder + '_(SMOOTH)'
            pd_dir  = processed_data_path + '/' + pd_folder + '/'
            pd_path = pd_dir + str(rd_DentRef) + '.csv'
            # Create the new folder directory if it does not already exist
            if not os.path.exists(pd_dir): os.mkdir(pd_dir)
            np.savetxt(pd_path, pd_radius, delimiter=',')
            
            # Create python inventory for quick access
            pd_radius = np.expand_dims(pd_radius, axis=0)
            if rd_first_file == True:
                # Do nothing
                rd_first_file = False
                od_radius = pd_radius.copy()
                od_SCF = np.array([dr_SCF])
            else:
                od_radius = np.concatenate((od_radius, pd_radius), axis=0)
                od_SCF    = np.concatenate((od_SCF, np.array([dr_SCF])), axis=0)
                
# =============================================================================
# EXPORT DATA FOR CURRENT SUBDIR
# =============================================================================

    # Save the resultant od_radius and od_SCF as python data files.
    # These data files are easily loaded in python for future reference.
    
    od_path = training_data_path + '/' + pd_folder
    od_radius_path = od_path + '_radius'
    od_SCF_path = od_path + '_SCF'
    
    np.save(od_radius_path, od_radius)
    np.save(od_SCF_path, od_SCF)
    
    time_total = time.time() - time_subdir_start
    time_m, time_s = divmod(time_total, 60)
    time_h, time_m = divmod(time_m, 60)
    print('Finished execution of subdir %s after %.0f:%.0f:%.0f' % (pd_folder, time_h, time_m, time_s))
    print('=========== SUBDIR END ===========')
    
# =============================================================================
# TOTAL EXECUTION TIME
# =============================================================================

# Total time
time_total = time.time() - time_start
time_m, time_s = divmod(time_total, 60)
time_h, time_m = divmod(time_m, 60)
print('Total execution time of %.0f:%.0f:%.0f' % (time_h, time_m, time_s))
print('=========== END ===========')
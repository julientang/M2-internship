import os 
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import toast
import astropy.units as u 


# nthread = os.environ["OMP_NUM_THREADS"]


def init_comm():
    comm, procs, rank = toast.get_world()
    return(comm)

def init_data(comm = None):
    if comm ==None : 
        comm = init_comm()
    toast_comm = toast.Comm(world=comm, groupsize=1)
    data = toast.Data(comm=toast_comm)
    return(data)

def PWG(mode = "IQU",NSIDE = 512):
    pointing = toast.ops.PointingDetectorSimple() ##boresight pointing into detector frame (RA/DEC by default)
    weights = toast.ops.StokesWeights(detector_pointing = pointing,mode = "IQU")
    pixels = toast.ops.PixelsHealpix(detector_pointing = pointing, nside = NSIDE)
    return(weights,pixels)

def filter_0(obs, det_data = 'signal'):
    obs_arr = obs.detdata[det_data]
    obs_arr2 = np.zeros(obs_arr.shape)
    i = 0
    for detec in obs_arr:
        obs_arr2[i] = detec- np.mean(detec)
        i+=1
    obs.detdata[det_data][:,:]  = obs_arr2
    
def init_focalplane(fplane,comm,srate =10*u.Hz):
    focalplane = toast.instrument.Focalplane(sample_rate=srate#,thinfp=256
                                        )
    with toast.io.H5File(fplane, "r", comm=comm, force_serial=True) as f:
            focalplane.load_hdf5(f.handle, comm=comm)
    return(focalplane)

def init_schedule(comm,sched):
    schedule = toast.schedule.GroundSchedule()
    schedule.read(sched, comm=comm)
    return(schedule)

def init_site(schedule):
    site = toast.instrument.GroundSite(
        schedule.site_name,
        schedule.site_lat,
        schedule.site_lon,
        schedule.site_alt,
        weather=None,
    )
    return(site)

def init_telescope(focalplane, site):
    telescope = toast.instrument.Telescope(
        "My_telescope", focalplane=focalplane, site=site
    )
    return(telescope)
    
def sim_ground(data,telescope, schedule,name="sim_ground", weather="south_pole", detset_key="pixel"):
    sim_ground = toast.ops.SimGround(name="sim_ground", 
                                 weather="south_pole", 
                                 detset_key="pixel", 
                                 telescope = telescope, 
                                 schedule = schedule
                                ) ##simulate motion of the boresight
    sim_ground.apply(data)

def noise(data,noiseless = True):    
    ob = data.obs[0]
    ob.detdata.create(name = 'noise',units = u.K)
    noise_model = toast.ops.DefaultNoiseModel()
    sim_noise = toast.ops.SimNoise() ###Need to instantiate Noise Model
    if noiseless : 
        sim_noise.det_data= 'noise'
    noise_model.apply(data) ## Read detector noise from the focalplane


        
def init_template_matrix(step_0 = 4*u.second):
    templates = [toast.templates.Offset(name="baselines", step_time = step_0)]
    template_matrix = toast.ops.TemplateMatrix(templates=templates)
    return(template_matrix)

def atmosphere(data, weights):
    hpointing = toast.ops.PointingDetectorSimple(boresight = 'boresight_azel')
    sim_atmosphere = toast.ops.SimAtmosphere(detector_pointing=hpointing, detector_weights= weights)
    sim_atmosphere.apply(data)
    
def scan_map(file_in, pixels, weights,data,det_data = 'signal'):
    scan_map = toast.ops.ScanHealpixMap(name="scan_healpix_map", file=file_in,pixel_pointing = pixels, stokes_weights = weights)
    scan_map.detdata = det_data
    scan_map.apply(data)
    
def mapmaker(pixels, weights, template_matrix, data,output_dir=None, det_data = 'signal',iter_max=50):
    binner = toast.ops.BinMap(pixel_pointing = pixels, stokes_weights = weights)
    filterbin = toast.ops.FilterBin(#cache_dir = 'filterbin'                                ,
                                det_flag_mask = 3 # Bit mask value for optional detector flagging
                                ,ground_filter_order=10
                                ,poly_filter_order=3
                                ,poly_filter_view = "scanning" # Intervals for polynomial filtering
                                ,rcond_threshold = 0.001 # Minimum value for inverse pixel condition number cut.
                                ,report_memory = False # Report memory throughout the execution
                                ,reset_pix_dist = False # Clear any existing pixel distribution.  Useful when applyingrepeatedly to different data objects.
                                ,rightleft_mask = 16 # Bit mask value for right-to-left scans
                                ,shared_flag_mask = 3 # Bit mask value for optional telescope flagging
                                ,shared_flags = "flags" # Observation shared key for telescope flags to use
                                ,split_ground_template = False # Apply a different template for left and right scans
                                ,write_binmap = True # If True, write the unfiltered map
                                ,write_cov = False # If True, write the white noise covariance matrices.
                                #,write_hdf5 = False # If True, output maps are in HDF5 rather than FITS format.
                                #,write_hdf5_serial = False # If True, force serial HDF5 write of output maps.
                                ,write_hits = False # If True, write the hits map
                                ,write_invcov = False # If True, write the inverse white noise covariance matrices.
                                ,write_map = True # If True, write the filtered map
                                ,write_noiseweighted_binmap = False # If True, write the noise-weighted unfiltered map
                                ,write_noiseweighted_map = False # If True, write the noise-weighted filtered map
                                ,write_obs_matrix = False # Write the observation matrix
                                ,write_rcond = True # If True, write the reciprocal condition numbers.
                                           )
    filterbin.det_data = det_data
    filterbin.binning= binner
    filterbin.output_dir = output_dir
    filterbin.apply(data)
    
def main(file_in, fplane,sched, output_dir,noiseless = False,atm = False):
    comm = init_comm()
    data = init_data(comm)
    weights,pixels = PWG()
    focalplane = init_focalplane(fplane,comm)
    schedule = init_schedule(comm,sched)
    site = init_site(schedule)
    telescope = init_telescope(focalplane,site)
    sim_ground(data,telescope,schedule)
    ob = data.obs[0]
    noise(data,noiseless)
    if atmosphere:
        atmosphere(data,weights)
    filter_0(ob)    
    template_matrix = init_template_matrix()
    scan_map(file_in,pixels,weights,data)
    mapmaker(pixels, weights, template_matrix, data, output_dir)
    
if __name__  == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Apply filtering and destriping to a simulated TOD from an input map')
    parser.add_argument('--file_in', metavar='path', required=True,
                        help='path to input map')
    parser.add_argument('--fplane', metavar='path', required=True,
                        help='path to focalplane')
    parser.add_argument('--sched', metavar='text', required=True,
                        help='path to scan strategy')
    parser.add_argument('--output_dir', metavar='dir', required=True,
                        help='output directory')
    parser.add_argument('--noiseless', metavar='Bool', required=False,
                        help='Enables noise')
    parser.add_argument('--atm', metavar='Bool', required=False,
                        help='Enables atmosphere')
    args = parser.parse_args()
    main(file_in=args.file_in, fplane=args.fplane, sched=args.sched,output_dir = args.output_dir,noiseless = args.noiseless, atm= args.atm)
    

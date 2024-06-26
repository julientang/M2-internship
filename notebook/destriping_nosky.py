import os 
import sys

import healpy as hp
import matplotlib.pyplot as plt
import numpy as np

import toast
import astropy.units as u 
import mpi4py as mp

from toast.mpi import MPI, Comm

def init_comm():
    comm, procs, rank = toast.get_world()
    return(comm)

def init_data(comm = None):
    if comm ==None : 
        comm = init_comm()
    toast_comm = toast.Comm(world=comm, groupsize=1)
    data = toast.Data(comm=toast_comm)
    return(data)

def PWG(data, mode = "IQU",NSIDE = 512):
    pointing = toast.ops.PointingDetectorSimple() ##boresight pointing into detector frame (RA/DEC by default)
    weights = toast.ops.StokesWeights(detector_pointing = pointing,mode = "IQU")
    pixels = toast.ops.PixelsHealpix(detector_pointing = pointing, nside = NSIDE)
    pointing.apply(data)
    weights.apply(data)
    pixels.apply(data)
    return(pointing, weights,pixels)

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

def noise(data): 
    for ob in data.obs:
        ob.detdata.create(name = 'noise',units = u.K)
    noise_model = toast.ops.DefaultNoiseModel()
    sim_noise = toast.ops.SimNoise() ###Need to instantiate Noise Model
    sim_noise.det_data= 'noise'
    noise_model.apply(data) ## Read detector noise from the focalplane
    sim_noise.apply(data) ##Write detector noise in associated timestream
    
def filter_0(obs, det_data = 'noise'):
    ## For each detector's timestream, remove the offset, so the TOD is centered around 0
    obs_arr = obs.detdata[det_data]
    obs_arr2 = np.zeros(obs_arr.shape)
    i = 0
    for detec in obs_arr:
        obs_arr2[i] = detec- np.mean(detec)
        i+=1
    obs.detdata[det_data][:,:]  = obs_arr2
        
def init_template_matrix(step_0 = 10*u.second):
    ## Template baseline for destriping
    templates = [toast.templates.Offset(name="baselines", step_time = step_0)]
    template_matrix = toast.ops.TemplateMatrix(templates=templates)
    return(template_matrix)

def atmosphere(data, weights):
    for ob in data.obs:
        ob.detdata.create(name = 'atmosphere',units = u.K)
    hpointing = toast.ops.PointingDetectorSimple(boresight = 'boresight_azel')
    sim_atmosphere = toast.ops.SimAtmosphere(detector_pointing=hpointing, detector_weights= weights)
    sim_atmosphere.apply(data)
    
def mapmaker(pixels, weights, template_matrix, data,output_dir=None, det_data = 'noise',iter_max=50):
    binner = toast.ops.BinMap(pixel_pointing = pixels, stokes_weights = weights)
    binner.det_data =det_data
    mapmaker = toast.ops.MapMaker(binning = binner, template_matrix=template_matrix)
    mapmaker.iter_max = iter_max
    mapmaker.binning= binner
    mapmaker.det_data = det_data
    mapmaker.output_dir = output_dir
    mapmaker.apply(data)
    
def main(fplane,sched, output_dir,atm = False):
    comm = init_comm()
    data = init_data(comm)
    pointing,weights,pixels = PWG(data)
    focalplane = init_focalplane(fplane,comm)
    schedule = init_schedule(comm,sched)
    site = init_site(schedule)
    telescope = init_telescope(focalplane,site)
    sim_ground(data,telescope,schedule)
    noise(data)
    if atm:
        atmosphere(data,weights)
    for ob in data.obs:
        filter_0(ob)
    template_matrix = init_template_matrix()
    mapmaker(pixels, weights, template_matrix, data, output_dir)
    
if __name__  == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Apply filtering and destriping to a simulated TOD from an input map')
    parser.add_argument('--fplane', metavar='path', required=True,
                        help='path to focalplane')
    parser.add_argument('--sched', metavar='text', required=True,
                        help='path to scan strategy')
    parser.add_argument('--output_dir', metavar='dir', required=True,
                        help='output directory')
    parser.add_argument('--atm', metavar='Bool', required=False,
                        help='Enables atmosphere')
    args = parser.parse_args()
    main(fplane=args.fplane, sched=args.sched,output_dir = args.output_dir,atm= args.atm)
    

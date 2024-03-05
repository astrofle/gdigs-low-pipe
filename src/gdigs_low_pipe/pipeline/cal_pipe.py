import gc
import os
import sys
import copy
import glob
import numpy as np
import bottleneck as bn

from pathlib import Path
from astropy.io import fits
from dysh.fits.gbtfitsload import GBTFITSLoad
from astropy.convolution import Gaussian1DKernel, interpolate_replace_nans

sys.path.append("/home/scratch/psalas/projects/GDIGS-Low/gdigs-low-pipe/src")
from gdigs_low_pipe import utils
from gdigs_low_pipe.rfi import arpls
from gdigs_low_pipe.calibration import baseline, fluxcal

sys.path.append("/home/scratch/psalas/groundhog/")
from groundhog import spectral_axis
from groundhog.fluxscales import calibrators


def output_fun(path: Path, ifnum: int, plnum: int, vbank: str, ext: str):
    """
    """
    return path / f"ifnum_{ifnum}_plnum_{plnum}_vbank_{vbank}{ext}"


def cal_pipe(sdfitsfile, outpath, ifnums=None, plnums=None,
             nchan=None, ch_edge=0.04,
             nd_win=512, ref_win=128, sig_win=32, 
             overwrite=False):
    """
    """

    print(f"Loading: {sdfitsfile}")
    sdf = GBTFITSLoad(sdfitsfile)
    print("Loaded.")

    if ifnums is None:
        ifnums = list(set(sdf._ptable[0]["IFNUM"]))
    if plnums is None:
        plnums = list(set(sdf._ptable[0]["PLNUM"]))
    if nchan is None:
        nchan = int(fits.getval(sdfitsfile, "TFORM7", ext=1)[:-1])

    # Define the range of channels to process.
    ch0 = int(ch_edge*nchan)
    chf = nchan - ch0
    chrng = slice(ch0, chf, 1)

    # Find the VEGAS bank.
    vbank = str(sdfitsfile).split(".")[-2]

    print(f"Will process IF={ifnums}")
    print(f"Will process PL={plnums}")
    print(f"Data has {nchan} channels")
    print(f"Will only consider channels {ch0} to {chf}")
    print(f"Data comes from VEGAS bank {vbank}")

    # Find scan numbers.
    scans = sdf._ptable[0]["SCAN"]

    mask = (np.asarray(sdf._ptable[0]["PROC"]) == "OnOff") | (np.asarray(sdf._ptable[0]["PROC"]) == "OffOn")
    cal_scans = np.sort(list(set(scans[mask])))
    print(f"Flux calibration scans: {cal_scans}")
    if len(cal_scans) <= 1:
        print("No flux calibration data.")
        return

    mask = (np.asarray(sdf._ptable[0]["PROC"]) == "Track")
    ref_scans = np.sort(np.array(list(set(scans[mask]))))
    print(f"Reference scans: {ref_scans}")

    mask = (np.asarray(sdf._ptable[0]["PROC"]) == "RALongMap") | (np.asarray(sdf._ptable[0]["PROC"]) == "DecLatMap")
    map_scans = np.sort(list(set(scans[mask])))
    print(f"Mapping scans: {map_scans}")

    if len(map_scans) == 0:
        print("No mapping data.")
        return

    uscans = np.sort(list(set(scans)))
    print(f"All scans: {uscans}")

    sig_ref_pairs = {}
    for s in map_scans:
        sig_ref_pairs[s] = ref_scans[np.argmin(abs(ref_scans - s))]
    print(f"Signal and reference pairs: {sig_ref_pairs}")

    if len(ifnums) == 0:
        ifnums = list(set(sdf._ptable[0]["IFNUM"]))

    for ifnum in ifnums:
        for plnum in plnums:

            print(f"Working on ifnum: {ifnum} and plnum: {plnum}")
            

            # Work with one spectral window and polarization at a time.
            df = sdf.select("IFNUM", ifnum, sdf._ptable[0])
            df = sdf.select("PLNUM", plnum, df)

            # Select data for flux calibration.
            fluxcal_df = sdf.select_scans(cal_scans, df)
            if set(fluxcal_df["PROC"]) == {"OffOn"}:
                ref_scan = cal_scans[0]
                sig_scan = cal_scans[1]
            else:
                ref_scan = cal_scans[1]
                sig_scan = cal_scans[0]
                
            sig_df     = sdf.select("SCAN", sig_scan, fluxcal_df)
            sig_on_df  = sdf.select("CAL", "T", sig_df)
            sig_off_df = sdf.select("CAL", "F", sig_df)
            ref_df     = sdf.select("SCAN", ref_scan, fluxcal_df)
            ref_on_df  = sdf.select("CAL", "T", ref_df)
            ref_off_df = sdf.select("CAL", "F", ref_df)


            # Extract data values.
            sig_on  = sdf._bintable[0].data[sig_on_df.index]["DATA"]
            sig_off = sdf._bintable[0].data[sig_off_df.index]["DATA"]
            ref_on  = sdf._bintable[0].data[ref_on_df.index]["DATA"]
            ref_off = sdf._bintable[0].data[ref_off_df.index]["DATA"]

            freq = spectral_axis.compute_spectral_axis(sdf._bintable[0].data[sig_on_df.index], chstart=1, chstop=-1, apply_doppler=True)
            source = list(set(sdf._bintable[0].data[sig_on_df.index]["OBJECT"]))[0].rstrip()
            ta_src = calibrators.compute_sed(freq, "Perley-Butler 2017", source, units='K')

            # Compute the gain, system temperature and noise diode temperature.
            linr = fluxcal.linear_cal(freq, sig_on, sig_off, ref_on, ref_off, ta_src)
            # Compute weights.
            exp = sdf._bintable[0].data[sig_on_df.index]["EXPOSURE"]
            wei = exp/np.nanmedian(linr["Tsys"], axis=1)**2.
            # Time average and smooth along frequency axis.
            tcal = np.ma.average(np.ma.masked_invalid(linr["Tcal"]), axis=0, weights=wei)
            tcal = utils.rolling(tcal, func=bn.move_median, win=nd_win)
            tsys = np.ma.average(np.ma.masked_invalid(linr["Tsys"]), axis=0, weights=wei)
            tsys = utils.rolling(tsys, func=bn.move_median, win=nd_win)

            print(f"System temperature during flux calibration: {np.nanmedian(tsys[chrng]):.2f} K -- linear")



            x = np.arange(-nchan//2,nchan//2,1)
            npol = 1
            kernel = Gaussian1DKernel(stddev=64)

            ref_on_arr = np.ma.empty((len(ref_scans), npol, nchan), dtype='d')
            ref_off_arr = np.ma.empty((len(ref_scans), npol, nchan), dtype='d')
            tsys_ref_arr = np.empty((len(ref_scans), npol, nchan), dtype='d')
            exp_on_arr = np.empty((len(ref_scans), npol, nchan), dtype='d')
            exp_off_arr = np.empty((len(ref_scans), npol, nchan), dtype='d')

            for i,s in enumerate(ref_scans[:]):
                for j,p_ in enumerate([plnum]):
                    #print(i,j)

                    scan_df     = sdf.select("SCAN", s, df)
                    scan_on_df  = sdf.select("CAL", "T", scan_df)
                    scan_off_df = sdf.select("CAL", "F", scan_df)

                    ref_on  = sdf._bintable[0].data[scan_on_df.index]
                    ref_off = sdf._bintable[0].data[scan_off_df.index]
                   
                    on_blanks  = utils.find_blanks(ref_on["DATA"])
                    off_blanks = utils.find_blanks(ref_off["DATA"])

                    ref_on_data  = np.ma.masked_invalid(ref_on["DATA"])
                    ref_off_data = np.ma.masked_invalid(ref_off["DATA"])
                    
                    diff     = ref_on_data - ref_off_data
                    tsys     = ref_off_data/diff * tcal + tcal/2.0
                    exp      = ref_on["EXPOSURE"] + ref_off["EXPOSURE"]
                    wei      = exp[:,np.newaxis]/tsys**2.
                    tsys_ref = np.ma.average(tsys, axis=0, weights=wei)

                    y  = tsys_ref[chrng]
                    xm = x[chrng][~np.isnan(y)]
                    ym = y[~np.isnan(y)]
                    
                    tsys_pfit         = np.polyfit(xm, ym, 1)
                    tsys_poly         = np.poly1d(tsys_pfit)(x)
                    tsys_ref_arr[i,j] = tsys_poly

                    # Flag RFI using ArPLS.
                    data = ref_on["DATA"][~on_blanks].T
                    rfi_mask = arpls.arpls_mask(data)
                    ref_on["DATA"][~on_blanks] = np.ma.masked_where(rfi_mask, data).filled(np.nan).T
                    data = ref_off["DATA"][~off_blanks].T
                    rfi_mask = arpls.arpls_mask(data)
                    ref_off["DATA"][~off_blanks] = np.ma.masked_where(rfi_mask, data).filled(np.nan).T

                    ref_on_       = np.ma.average(np.ma.masked_invalid(ref_on["DATA"]), axis=0, weights=wei)
                    result        = interpolate_replace_nans(ref_on_.filled(np.nan), kernel)
                    ref_on_f      = ref_on_.copy()
                    ref_on_f[ref_on_f.mask] = result[ref_on_f.mask]
                    ref_on_f.mask = False

                    ref_off_       = np.ma.average(np.ma.masked_invalid(ref_off["DATA"]), axis=0, weights=wei)
                    result         = interpolate_replace_nans(ref_off_.filled(np.nan), kernel)
                    ref_off_f      = ref_off_.copy()
                    ref_off_f[ref_off_f.mask] = result[ref_off_f.mask]
                    ref_off_f.mask = False

                    ref_on_smo  = utils.rolling(ref_on_f.filled(np.nan), func=bn.move_mean, win=ref_win)
                    ref_on_smo  = np.ma.masked_where(ref_on_f.mask, ref_on_smo)
                    ref_off_smo = utils.rolling(ref_off_f.filled(np.nan), func=bn.move_mean, win=ref_win)
                    ref_off_smo = np.ma.masked_where(ref_off_f.mask, ref_off_smo)
                    gtcal       = utils.rolling(ref_on_smo - ref_off_smo, func=bn.move_mean, win=ref_win)
                    
                    ref_on_arr[i,j] = np.ma.masked_invalid(ref_on_smo - gtcal)
                    ref_off_arr[i,j] = np.ma.masked_invalid(ref_off_smo)
                
                    exp_on_arr[i,j] = ref_on["EXPOSURE"].sum()
                    exp_off_arr[i,j] = ref_off["EXPOSURE"].sum()
                    

            ref_on_avg = np.ma.average(ref_on_arr, axis=0, weights=exp_on_arr*np.power(tsys_ref_arr, -2.))
            ref_on_avg = arpls.ArPLS(ref_on_avg[0], lam=1e7)
            ref_off_avg = np.ma.average(ref_off_arr, axis=0, weights=exp_off_arr*np.power(tsys_ref_arr, -2.))
            ref_off_avg = arpls.ArPLS(ref_off_avg[0], lam=1e7)
            ref_on_exp = exp_on_arr.sum(axis=0)
            ref_off_exp = exp_off_arr.sum(axis=0)

            # Process mapping scans.
            mask = df["SCAN"].isin(map_scans)
            stable = sdf._bintable[0].data[df.index][mask].copy()
            #stable = table.data[df.index][mask]

            for i,s in enumerate(map_scans):
                
                ref_idx = list(ref_scans).index(sig_ref_pairs[s])
                
                scan_df     = sdf.select("SCAN", s, df)
                scan_on_df  = sdf.select("CAL", "T", scan_df)
                scan_off_df = sdf.select("CAL", "F", scan_df)

                sig_on  = sdf._bintable[0].data[scan_on_df.index]
                sig_off = sdf._bintable[0].data[scan_off_df.index]
                
                tsys_ = tsys_ref_arr[ref_idx]
                ta_on = (sig_on["DATA"] - ref_on_avg)/ref_on_avg * tsys_ - tcal
                ta_off = (sig_off["DATA"] - ref_off_avg)/ref_off_avg * tsys_
                #ta_on  = (sig_on["DATA"] - ref_on_arr[ref_idx,0,:])/ref_on_arr[ref_idx,0,:] * tsys_ - tcal
                #ta_off = (sig_off["DATA"] - ref_off_arr[ref_idx,0,:])/ref_off_arr[ref_idx,0,:] * tsys_

                sig_on["DATA"]      = ta_on
                sig_off["DATA"]     = ta_off
                sig_on["TSYS"]      = np.nanmean(tsys_)
                sig_off["TSYS"]     = np.nanmean(tsys_)
                ref_exp             = np.nanmedian(ref_on_exp)*ref_win
                sig_on["EXPOSURE"]  = sig_on["EXPOSURE"]*ref_exp/(sig_on["EXPOSURE"] + ref_exp)
                ref_exp             = np.nanmedian(ref_off_exp)*ref_win
                sig_off["EXPOSURE"] = sig_off["EXPOSURE"]*ref_exp/(sig_off["EXPOSURE"] + ref_exp)

                mask = (stable["SCAN"] == s) & (stable["CAL"] == "T")
                stable["TSYS"][mask] = np.nanmean(tsys_)
                stable["EXPOSURE"][mask] = sig_on["EXPOSURE"]
                stable["DATA"][mask] = ta_on#.filled(np.nan)
                mask = (stable["SCAN"] == s) & (stable["CAL"] == "F")
                stable["TSYS"][mask] = np.nanmean(tsys_)
                stable["EXPOSURE"][mask] = sig_off["EXPOSURE"]
                stable["DATA"][mask] = ta_off#.filled(np.nan)


            blank_mask = np.all(np.isnan(stable["DATA"]), axis=1)
            print(f"Found {blank_mask.sum()} blanked integrations.")

            # Remove baseline and flag RFI.
            base = baseline.baseline(stable["DATA"][~blank_mask], lam=1e7, rms_window=32)
            z_norm_data = (base["baselined_data"]/base["rms"])
            rfi_mask = arpls.arpls_mask(z_norm_data[:,chrng].T).T
            #masked_data = np.ma.masked_where(rfi_mask, base["baselined_data"][:,chrng])
            masked_data = np.ma.masked_where(rfi_mask, stable["DATA"][~blank_mask,chrng])
            
            stable["DATA"][~blank_mask,chrng] = masked_data.filled(np.nan)

            # Save calibrated data.
            os.makedirs(outpath, exist_ok=True)

            hdu0 = sdf._hdu[0].copy()
            thdu = fits.BinTableHDU(stable, header=sdf._hdu[1].header)
            outhdu = fits.HDUList([hdu0, thdu])
            out_fits = output_fun(outpath, ifnum, plnum, vbank, ".fits")
            outhdu.writeto(out_fits, overwrite=overwrite)
            print(f"Saved calibrated data to: {out_fits}")
   
            # Save baseline.
            masked_data = np.ma.masked_where(rfi_mask, base["baseline"][:,chrng])
            stable["DATA"][~blank_mask,chrng] = masked_data.filled(np.nan)
            thdu = fits.BinTableHDU(stable, header=sdf._hdu[1].header)
            outhdu = fits.HDUList([hdu0, thdu])
            out_fits = output_fun(outpath, ifnum, plnum, vbank, "_bl.fits")
            outhdu.writeto(out_fits, overwrite=overwrite)
            print(f"Saved baselines to: {out_fits}")

            # Clear.
            del df
            del stable
            del hdu0
            del thdu
            del outhdu
            gc.collect()


#if __name__ == "__main__":
#
#
#
#    path        = "/home/scratch/psalas/projects/GDIGS-Low/raw-data/"
#    proj        = "AGBT22A_437"
#    #sessions    = ["23"] + [f"{s:02d}" for s in np.arange(55, 68)]
#    #sessions    = [f"{s:02d}" for s in np.arange(33, 37)]
#    sessions    = ["54"]
#    plnums      = [0,1]
#    ifnums      = [11,12,13,14]
#    nchan       = 2**15
#    fluxcal_smo = 512
#    ref_smo_win = 128
#    ref_rfi_win = 128
#    ch0         = int(0.04*nchan)
#    chf         = nchan - ch0
#    chrng       = slice(ch0,chf,1)
#
#    for session in sessions:
#        projid       = f"{proj}_{session}"
#        sdfits_path  = f"{path}/{projid}/{projid}.raw.vegas/"
#        sdfits_files = glob.glob(f"{sdfits_path}/{projid}.raw.vegas.B.fits")
#        for sdfitsfile in sdfits_files:
#            out_path   = f"/home/scratch/psalas/projects/GDIGS-Low/outputs/test_v5_nobaseline/{projid}/"
#            vbank      = sdfitsfile.split(".")[-2]
#            output_fun = lambda ifnum, plnum, ext: f"{out_path}/ifnum_{ifnum}_plnum_{plnum}_vbank_{vbank}{ext}"
#            run_pipe(sdfitsfile, ifnums, plnums, nchan, fluxcal_smo, ref_smo_win, ref_rfi_win,
#                     ch0, chf, chrng, out_path, output_fun, overwrite=True)

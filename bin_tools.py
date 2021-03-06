"""Optimal window average code

This module allows for determing the best window average of one genomic dataset
that best predicts a second genomic dataset.  The ``main`` function is
optimum_window.

Example:
    Compair my.bed1 to my.bed2 ::
        bin_tools.split("my.bed1", "my_dir1/")
        bin_tools.bed_to_bin_dir("my_dir1/")
        bin_tools.split("my.bed2", "my_dir2/")
        bin_tools.bed_to_bin_dir("my_dir2/")
        bin_tools.optimum_window(dir1="my_dir1/", dir2="my_dir2/",
                            precut_frac1=0.5, outprefix="my_output",
                            cutoff2=0.5)

    Alturnatively ::
        bin_tools.optimum_window("../LADs/meth_Nhfl/", "../LADs/LAD_chroms/", "NHLF_LAD_Data")
        bin_tools.plot_optimum("./NHLF_LAD_Data")

"""

import numpy as np
from multiprocessing import Pool
import pickle
import math
from multiprocessing import Pool, Array, Process
from scipy import stats
hg19_chrom_lengths={
    "chr1":249250621
   , "chr2":243199373
   , "chr3":198022430
   , "chr4":191154276
   , "chr5":180915260
   , "chr6":171115067
   , "chr7":159138663
   , "chrX":155270560
   , "chr8":146364022
   , "chr9":141213431
   , "chr10":135534747
   , "chr11":135006516
   , "chr12":133851895
   , "chr13":115169878
   , "chr14":107349540
   , "chr15":102531392
   , "chr16":90354753
   , "chr17":81195210
   , "chr18":78077248
   , "chr20":63025520
   , "chrY":59373566
   , "chr19":59128983
   , "chr22":51304566
   , "chr21":48129895}
"""dict: Lengths of chromosomes in hg19"""
total_length_chr19=0
"""int: total length of hg19 in bp"""
total_bins_chr19=0
"""int: total length of hg19 in bins"""
file_bin_width=200
"""int: length of hg19 in bins
I typically use 200bp bins corrisponding to nucleosomes
"""
def veclen(length):
    return int(np.ceil(length/file_bin_width))
for chromosome, chr_end in hg19_chrom_lengths.items():
    total_length_chr19 += chr_end
    total_bins_chr19 += veclen(chr_end)
windows_default = np.unique((10**np.linspace(0.2,5.2,100)).astype(int))-1

# curl -s "http://hgdownload.cse.ucsc.edu/goldenPath/hg189/database/cytoBand.txt.gz" | gunzip -c | grep acen
def get_centromeres(filename = "../../centromereshg19"):
    """load {centromere: [start centromere bin, end centromere bin]}
    """
    out = {}
    with open(filename) as fl:
        for line in fl.readlines():
            temp = line.split()
            if temp[0] in out.keys():
                out[temp[0]][1] = veclen(int(temp[2]))
            else:
                out[temp[0]] = [veclen(int(temp[1])), 0]
    return out
centromeres = get_centromeres()

chromosome_numbers = {
     "chr1":1
   , "chr2":2
   , "chr3":3
   , "chr4":4
   , "chr5":5
   , "chr6":6
   , "chr7":7
   , "chr8":8
   , "chr9":9
   , "chr10":10
   , "chr11":11
   , "chr12":12
   , "chr13":13
   , "chr14":14
   , "chr15":15
   , "chr16":16
   , "chr17":17
   , "chr18":18
   , "chr19":19
   , "chr20":20
   , "chr21":21
   , "chr22":22
   , "chrX":23
   , "chrY":24}
def invert_dict(x):
    return {value: key for key, value in x.items()}
chromosome_names = invert_dict(chromosome_numbers)

class genomic_locations:
    # in constructor create Array for chrom, loc, and val
    def __init__(self, fileName):
        loc = []
        if isinstance(fileName, list):
            for fn in fileName:
                with open(fn,"r") as fl:
                    for line in fl.readlines():
                        vals = line.split()
                        loc.append((chromosome_numbers[vals[0]],
                                    int( (int(vals[1]) + int(vals[2]))/2 ),
                                    float(vals[3]) ))
        else:
            with open(fileName,"r") as fl:
                for line in fl.readlines():
                    vals = line.split()
                    loc.append((chromosome_numbers[vals[0]],
                                int( (int(vals[1]) + int(vals[2]))/2 ),
                                float(vals[3]) ))
        loc.sort() # Makes data retreaval more efficent
        self.npts = len(loc)
        self.chroms = Array('i', self.npts, lock=False)
        self.loc = Array('i', self.npts, lock=False)
        self.values = Array('d', self.npts, lock=False)
        for ii in range(self.npts):
            self.chroms[ii] = loc[ii][0]
            self.loc[ii] = loc[ii][1]
            self.values[ii] = loc[ii][2]

    def __getitem__(self, key):
        return [self.chroms[key], self.loc[key], self.values[key]]

    def __setitem__(self, key, value):
        self.values[key] = value

    def __len__(self):
        return self.npts

    def average(self):
        """Average by chromosome in format {'chr1':1.2, ...}
        """
        sum_by_chr = {}
        for chromosome_number in chromosome_names.keys():
            sum_by_chr[chromosome_number] = [0.0, 0]
            # example {'1': [0.0, 0] ...} where list is [total, count]
        for ii in range(self.npts):
            chromosome_number = self.chroms[ii]
            sum_by_chr[chromosome_number][0] += self.values[ii]
            sum_by_chr[chromosome_number][1] += 1
        out = {}
        for chromosome_number, value in sum_by_chr.items():
            if value[1] == 0:
                continue
            out[chromosome_names[chromosome_number]] = value[0]/value[1]
        return out


class genomic_vector:
    """Genomic information stored in memory.
    """
    def __init__(self, directory):
        self.shared_chroms = [Array('d', veclen(length), lock=False) for
                              length in  hg19_chrom_lengths.values()]
        self.chrom_names = []
        self.chrom_index ={}
        ii=0
        for chromosome, chr_end in hg19_chrom_lengths.items():
            self.chrom_names.append(chromosome)
            self.chrom_index[chromosome]=ii
            self.shared_chroms[ii]=np.loadtxt(directory+chromosome+"_bins")
            ii=ii+1
        self.get_chrom_stats()

    def get_chrom_stats(self):
        self.minimum=None
        self.maximum=None
        self.max_median=None
        self.total_length = 0
        first_chr=True
        self.mean = 0
        for chromosome, chr_end in hg19_chrom_lengths.items():
            values = self.get_chromosome(chromosome)
            self.total_length+=len(values)
            median=np.median(values)
            minimum=np.amin(values)
            maximum=np.amax(values)
            self.mean += np.sum(values)
            if first_chr:
                first_chr=False
                self.minimum=minimum
                self.maximum=maximum
                self.max_median=median
            else:
                self.minimum=min(self.minimum,minimum)
                self.maximum=max(self.maximum,maximum)
                self.max_median=max(self.max_median, median)
        self.mean = self.mean/self.total_length
        self.stats={"min":self.minimum, "max":self.maximum,
                    "max_median":self.max_median, "total_length":self.total_length,
                    "mean":self.mean}
        return self.stats

    def chrom_pointer(self, chromosome):
        ii = self.chrom_index[chromosome]
        return self.shared_chroms[ii]

    def get_chromosome(self, chromosome):
        out = np.zeros(veclen(hg19_chrom_lengths[chromosome]))
        ii = self.chrom_index[chromosome]
        out[:] = self.shared_chroms[ii][:]
        return out

    def get_chromosome_average(self, chromosome, exclude_centromere=False):
        ii = self.chrom_index[chromosome]
        if exclude_centromere:
            centro = centromeres[chromosome]
            total = sum(self.shared_chroms[ii][0:centro[0]])
            total += sum(self.shared_chroms[ii][centro[1]:-1])
            length = veclen(hg19_chrom_lengths[chromosome]) - (centro[1] - centro[0])
            return total/length
        total = sum(self.shared_chroms[ii])
        return total/veclen(hg19_chrom_lengths[chromosome])

    def apply_filter(self, args):
        for chromosome, chr_end in hg19_chrom_lengths.items():
            self.chrom_names.append(chromosome)
            ii = self.chrom_index[chromosome]
            self.shared_chroms[ii]=self.get_processed_vector(chromosome,
                                                             **args)
        self.get_chrom_stats()

    def get_processed_vector(self, chromosome, half_width=0, cutoff=None,
                             cutoff_as_prob_of_1=False, cutoff_after=None):
        """Load data file and apply cutoff(s) and filter.
        Input file expected to have a single number per line.

        Args:
            chromosome (str): Which chromosome to precess
            half_width (int): The half-width of the window average to apply.
            cutoff (float): Make input vec binary using cutoff as a cutoff
            cutoff_after (float): Make output vec binary using cutoff as a cutoff
        """
        out = self.get_chromosome(chromosome)
        if type(cutoff) != type(None):
            if cutoff_as_prob_of_1:
                out = np.minimum(out,cutoff)/cutoff
            else:
                out = (out>cutoff).astype(int)
        if half_width>0:
            out = fastSquareFilter(out,
                                  halfWidth=half_width)
        if type(cutoff_after) != type(None):
            out = (out>cutoff_after).astype(int)
        return out

def make_fake_data(width=500):
    import os
    for directory in ["fake_dir1/","fake_dir2/"]:
        if os.path.exists(directory):
            os.system("rm -r "+directory)
        os.makedirs(directory)

    for chromosome, chr_end in hg19_chrom_lengths.items():
        length = np.ceil(chr_end/file_bin_width).astype(int)
        set2 = np.zeros(length)
        for ii in range(length):
            set2[ii] = np.mod((ii//width),2)
        set1=set2.copy()
        for ii in range(length):
            if np.random.rand()<0.45:
                set1[ii]=1-set1[ii]
        np.savetxt("fake_dir1/"+chromosome+"_bins",set1, fmt='%.3f')
        np.savetxt("fake_dir2/"+chromosome+"_bins",set2, fmt='%.3f')



# ---------------------------------------------------------------
#         File processing
# ---------------------------------------------------------------

def split(infile, outdir=''):
    """Splits a single bed file into chromosomes

    Args:
        infile (str): input file name of type bed
        outdir (str): string to prepend to output name
    """
    out_file_handles=dict()
    for chromosome, chr_end in hg19_chrom_lengths.items():
        out_file_handles[chromosome] = open(outdir+chromosome,"w")

    with open(infile) as file_handle:
        for line in file_handle:
            vals = line.split()
            if vals[0] in out_file_handles:
                print(line,file=out_file_handles[vals[0]],end='')

    for chromosome, chr_end in hg19_chrom_lengths.items():
        out_file_handles[chromosome].close()

def bed_to_bin_dir(outdir, binwidth=file_bin_width):
    """Apply bed_to_bin to each chromosome

    Args:
        outdir (str): string to prepend to output name
        binwidth (int): width of bins in nucleoosomes
    """
    for chromosome, upperBound in hg19_chrom_lengths.items():
        fname = outdir+chromosome
        out_name = outdir+chromosome+"_bins"
        bed_to_bin(fname, out_name, binwidth=file_bin_width, upperBound=upperBound)

def bed_to_bin(fname, out_name, binwidth=file_bin_width, upperBound=1000, offset=0):
    """Integrate a bed file into evenly spaced bins.

    Args:
        fname (str): Name of input bed file (one chromosome please)
        out_name (str): Name of output file
        binwidth (int): width of bins in bp
        upperBound (int): length of chromosome
        offset (int): Set to 1 if missing first column
    """
    nbins=math.ceil(upperBound/binwidth)
    histvals=[0.0] * nbins
    binNumber=0

    with open(fname) as f:
        for line in f:
            temp=line.split()
            if len(temp) != 4:
                raise ValueError("Line of "+fname+": "+line)
            start = int(temp[1+offset])
            end = int(temp[2+offset])
            value = float(temp[3+offset])

            # Bed files start at 0, start in included in feature, end isn't

            binNumber_start = start//binwidth
            binNumber_end = (end-1)//binwidth

            if binNumber_start==binNumber_end:
                histvals[binNumber_start] += value*(end-start)
                continue

            histvals[binNumber_start] += \
                    ((binNumber_start+1)*binwidth - start)*value
            if binNumber_end>binNumber_start+1:
                for binNumber in range(binNumber_start+1,binNumber_end):
                    histvals[binNumber] += binwidth*value
            histvals[binNumber_end] += (end - binNumber_end*binwidth)*value


            #while (start >= (binNumber+1)*binwidth):
            #    binNumber=binNumber+1

            #if (end > (binNumber+1)*binwidth ):
            #
            #    histvals[binNumber]=histvals[binNumber]+ \
            #                         ((binNumber+1)*binwidth - start)*value
            #    binNumber=binNumber+1
            #    while end > (binNumber+1)*binwidth:
            #        histvals[binNumber]=binwidth*value
            #        binNumber=binNumber+1

            #    histvals[binNumber] = (end - binNumber*binwidth - 1)*value
            #else:
            #    histvals[binNumber]=histvals[binNumber]+(end-start)*value
    with open(out_name,"w") as f:
        for val in histvals:
            print(val, file=f)

def bins_to_prob(dir1='', dirout='./', mark_frac=0.5):
    """Converts from bined Chip Signal to mark probability.

    Probability is linear the Chip signal so that average is mark_frac

    """
    set1 = genomic_vector(dir1)
    nbins=1000
    bins=np.linspace(set1.minimum,
                         min(set1.maximum,set1.max_median*5.0),
                         nbins+1)

    # --- Bin post window dir1 to determine cutoff at each window ---
    print("Making histograms for each windowsize")
    args={"set1":set1, "bins":bins}
    hists = post_bin_hist_all_chr(args)
    cutoff = mark_prob_cutoff(hists, bins, mark_frac=0.5)
    for chromosome in hg19_chrom_lengths:
        with open(dir1+chromosome+"_prob","w") as file_handle:
            for val in set1.chrom_pointer(chromosome):
                print(min(val/cutoff,1.0),file=file_handle)

def coarseGrain(dir1,in_suffix='_prob',out_suffix='_meth'):
    pass

# --------------------------------------------------
#     New numpy commands
# --------------------------------------------------
from numba import jit

@jit(nopython =True)
def fastSquareFilter(invec, halfWidth=0):
    """Applies square filter.

    Args:
        invec (ndarray): Input vector.
        width (int): With of square_filter
    """
    NT=invec.shape[0]
    out = np.zeros(NT)

    total = invec[0]
    lower = 0
    upper = 0
    for center in range(0,NT):
        while upper < min(center+halfWidth, NT-1):
            upper = upper + 1
            total = total + invec[upper]
        while lower < max(0,center-halfWidth):
            total = total - invec[lower]
            lower = lower + 1
        out[center] = total/float(upper-lower+1)
    return out

def cumulent(array):
    """Caltulate running sum of input.
    """
    out = np.zeros(len(array))
    for idx, val in enumerate(array):
        if idx == 0:
            out[idx] = val
        else:
            out[idx] = val+out[idx-1]
    return out

def cut(data, upper_fraction=0.5, number_of_bins=1000, cut_val=False):
    """Which values are in the upper_fraction.

    Args:
        data (ndarray): input data vector
        upper_fraction (float): What fraction to return True for.
        number_of_bins (int): Accuracy of approximation.
        cut_val (bool): Only return cutoff value
    """
    bins = np.linspace(np.amin(data), np.amax(data), number_of_bins+1)
    hists = np.histogram(data, bins)[0]
    cum_f = cumulent(hists)/sum(hists)
    cutoff_idx = np.argmin(abs((1.0-cum_f) - upper_fraction))
    if cut_val:
        return bins[cutoff_idx+1]
    return data > bins[cutoff_idx + 1]


# ------------------------------------------------------------------
#      Loading/using binned chromosome files
# ------------------------------------------------------------------


def get_window_average_at(chip_set, half_width, chroms, locs, npts):
    """Return chip_set averaged over half_width at specified locations.

    Args:
        chip_set (genomic_vector): genomic data to be windows averaged.
        half_width (int): half widith of window average to be perfomed
        chroms (Array): List in ints corrisponding to chromosomes
        locs (Array): List of int genomic positions
        npts (int): length of arrays chroms and locs
    """
    smoothed = {}
    for chromosome, chr_num in chromosome_numbers.items():
        smoothed[chr_num] = chip_set.get_processed_vector(chromosome,
                                                          half_width)
    out = np.zeros(npts)
    for ii in range(npts):
        try:
            out[ii] = smoothed[chroms[ii]][ locs[ii]//file_bin_width ]
        except:
            print("Tried to access %d of chr%d, only goes to %d"%(
                   locs[ii]//file_bin_width, chroms[ii],
                   len(smoothed[chroms[ii]])))
            import pdb
            pdb.set_trace()
            pass
    return out

def dot_product(set1='', set2='', chromosome='chr1',
                half_width1=0,
                half_width2=0, cutoff1=None, cutoff2=None, cutoff_after1=None,
                cutoff_after2=None, binary_predict=False,
                precut1_as_prob=False, precut2_as_prob=False):
    """Calculate the product of 2 different genomic data sets.
    This can be used in correlating one data set to another.
    Cutoffs and window averaging can be applied to either data set.

    Args:
        set1 (obj): Genome vector object
        set2 (obj): Genome vector object
        chromosome (str): Chromosome name
        half_width1 (int): window average set1 half_width to either side
        half_width2 (int): window average set2 half_width to either side
        cutoff1 (int): Make set1 binary before window averaging
        cutoff2 (int): Make set2 binary before window averaging
        cutoff_after1 (int): Make set1 binary after window averaging
        cutoff_after2 (int): Make set2 binary after window averaging
    """
    #print("Working on "+chromosome+" half_width1="+str(half_width1))
    f1 = set1.get_processed_vector(chromosome, half_width=half_width1,
                                   cutoff=cutoff1, cutoff_after=cutoff_after1,
                                   cutoff_as_prob_of_1 = precut1_as_prob)
    f2 = set2.get_processed_vector(chromosome, half_width=half_width2,
                                   cutoff=cutoff2, cutoff_after=cutoff_after2,
                                   cutoff_as_prob_of_1 = precut2_as_prob)
    if binary_predict:
        return np.dot(f1,f2)+np.dot(1-f1,1-f2)
    return np.dot(f1, f2)
def dot_wrapper(args):
    """wrapper for dot_product"""
    return dot_product(**args)
def all_chromosome_dot(args):
    total = 0.0
    for chromosome, chr_end in hg19_chrom_lengths.items():
        total += dot_product(**args, chromosome=chromosome)
    return total

def make_split_hist(set1='', set2='', half_width1=0, half_width2=0,
                    cutoff1=None, cutoff2=0.5, cutoff_after1=None,
                    limits=None,nbins=50):
    """ Make histogram of set1 data, colored by set2.
    cutoff2 is the binary cutoff on set 2 to color set 1.
    Averaging and cutoffs can be applied before histogram is generated.

    Args:
        set1 (obj): Genome vector object
        set2 (obj): Genome vector object
        half_width1 (int): window average set1 half_width to either side
        half_width2 (int): window average set2 half_width to either side
        cutoff1 (int): Make set1 binary before window averaging
    """
    if type(limits)==type(None):
        limits = get_limits_chrom_vecs(dir1)
    bins = np.linspace(limits[0],limits[2],nbins+1)
    hists_top=np.zeros(nbins)
    hists_bot=np.zeros(nbins)
    for chromosome, chr_end in hg19_chrom_lengths.items():
        f2=set2.get_chromosome(chromosome)
        f2=f2>cutoff2
        f1 = set1.get_processed_vector(chromosome, half_width1,
                              cutoff=cutoff1, cutoff_after=cutoff_after1)
        hists_top += np.histogram(f1[f2],bins)[0]
        hists_bot += np.histogram(f1[np.invert(f2)],bins)[0]
    return [bins, hists_top, hists_bot]

def post_bin_histogram(set1='', bins=[0.0,0.5,1.0], chromosome='chr1',
                       half_width1=0, pre_cutoff=None):
    """Histogram values after applied window.
    Assumes directory format created by bed_to_bin_dir.

    Args:
        directory (str): Name of directory containing input files
        bins (ndarray): array of bin edges
        chromosome (str): Chromosome name
        half_width1 (int): Half width of window averaging
        pre_cutoff (float):  Apply this cutoff before windowing
    """
    f1 = set1.get_processed_vector(chromosome, half_width1,
                              cutoff=pre_cutoff)
    hist = np.histogram(f1,bins)[0]
    return hist

def post_bin_hist_all_chr(args):
    total = np.zeros(len(args["bins"])-1)
    for chromosome, chr_end in hg19_chrom_lengths.items():
        args["chromosome"]=chromosome
        total += post_bin_histogram(**args)
    del args["chromosome"]
    return total

def get_all_chromosome_cut(set1, upper_fraction=0.5, nbins=5000):
    """Cutoff with upper_fraction of values above.
    I.e. inverse of get_fraction_greater_than

    Args:
        set1 (obj):
        upper_fraction (flaot): Desired fraction above returned cutoff
        nbins (int): number of bins (the more the more accurate)
    """
    hists = np.zeros(nbins).astype(int)
    bins = np.linspace(set1.minimum, set1.max_median, nbins+1)
    for chromosome, chr_end in hg19_chrom_lengths.items():
        values = set1.get_chromosome(chromosome)
        hists += np.histogram(values, bins)[0]
    #cum_f = cumulent(hists)/sum(hists)
    cum_f = cumulent(hists)/set1.total_length
    cutoff_idx = np.argmin(abs((1.0-cum_f) - upper_fraction))
    return bins[cutoff_idx+1]

def get_fraction_greater_than(set1, cutoff, precut=None, halfWidth=0):
    """Fration of genome greater that cutoff
    I.e. inverse of get_all_chromosome_cut
    Assumes directory format created by bed_to_bin_dir.

    Args:
        directory (str): Name of directory containing input files
        cutoff (float):
    """
    total_length=0
    total_number=0
    for chromosome, chr_end in hg19_chrom_lengths.items():
        values = set1.get_processed_vector(chromosome,
                                      half_width=halfWidth, cutoff=precut,
                                      cutoff_after=cutoff)
        total_length += len(values)
        total_number += sum(values>cutoff)
    return float(total_number)/total_length

def make_2D_hist_all_chromosomes(args):
    """
    Example:
        Compair my.bed to a similarly processed data set ::
            bin_tools.split("my.bed", "my_dir/")
            bin_tools.bed_to_bin_dir("my_dir/")
            args={"dir_x":"my_dir/", "dir_y":"other_dir/"}
            outputs = bin_tools.make_2D_hist_all_chromosomes(args)
    """
    # generate bins
    nbins=100
    (min_x, max_x) = (args["set_x"].minimum, args["set_x"].maximum)
    args["bins_x"] = np.linspace(min_x,max_x,nbins+1)
    (min_y, max_y) = (args["set_y"].minimum, args["set_y"].maximum)
    args["bins_y"] = np.linspace(min_y,max_y,nbins+1)

    # Use seperate processer for each chromosome
    conditions = []
    for chromosome, chr_end in hg19_chrom_lengths.items():
        my_args=args.copy()
        my_args["chromosome"]=chromosome
        conditions.append(my_args)
    with Pool(23) as my_pool:
        output=my_pool.map(make_2D_hist_wrapper, conditions)
    return [sum(output), args["bins_x"], args["bins_y"]]

def make_2D_hist(setx='', sety='', half_width_x=0, half_width_y=0,
                 bins_x=None, bins_y=None, chromosome='Chr1'):
    fx = setx.get_processed_vector(chromosome, half_width=half_width_x, cutoff=None)
    fy = sety.get_processed_vector(chromosome, half_width=half_width_y, cutoff=None)
    out =  np.histogram2d(fx, fy, bins=[bins_x, bins_y])[0]
    return out
def make_2D_hist_wrapper(args):
    return make_2D_hist(**args)

#-------------------------------------------------------
#   Histogram tools
#-------------------------------------------------------

def winlist2cut(winlist, bins, upper_fraction=0.5, use_hg19_length=True):
    cutoffs = []

    for hists in winlist:
        if use_hg19_length:
            cum_f = cumulent(hists)/total_bins_chr19
        else:
            cum_f = cumulent(hists)/sum(hists)
        cutoff_idx = np.argmin(abs((1.0-cum_f) - upper_fraction))
        cutoffs.append(bins[cutoff_idx+1])
    return cutoffs

def mark_prob_cutoff(hist, bins, mark_frac=0.5):
    """Cutoff such that when given to bins_to_prob the average
    probability is mark_frac.

    hist (ndarray): histogram values of data
    bins (ndarray): edges of histogram bins (len(bins) = len(hist) + 1
    mark_frac (float): overall fraction marked (e.g. avj. prob.)
    """
    centers = 0.5*(bins[1:]+bins[:-1])
    prev_frac_un_marked = 0.0
    for i_star in range(1,len(centers)):
        # values in bin i_star and up excluded
        cutoff = bins[i_star]
        unmark = sum((1.0-centers[:i_star]/cutoff)*hist[:i_star])/total_bins_chr19
        if unmark < (1.0 - mark_frac):
            prev_frac_un_marked = unmark
        else:
            if abs(unmark-(1.0- mark_frac)) > \
               abs(prev_frac_un_marked - (1.0 - mark_frac)):
                return bins[i_star-1]
            else:
                return bins[i_star]
    raise ValueError("Unable to find half of chr 19 to unmark")

def plot_window_histograms(pickle_name, skip=1, title=None, maxWindow=None,
                           upper_fraction=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    #(bins, winlist)=pickle.load(open(pickle_name,"rb"))
    with open(pickle_name,"rb") as fl:
        data=pickle.load(fl)
    bins=data["bins"]
    winlist=data["winlist"]
    centers = 0.5*(bins[:-1]+bins[1:])
    if type(upper_fraction) != type(None):
        cutoffs = winlist2cut(winlist, bins, upper_fraction)
    else:
        cutoffs=data["post_window_cutoffs"]

    for i_win, hist in enumerate(winlist):
        if np.mod(i_win,skip)>0:
            continue
        if maxWindow:
            ncolors = sum( (2*windows_default+1)*file_bin_width <=maxWindow)
            if (2*windows_default[i_win]+1)*file_bin_width > maxWindow:
                continue
        else:
            ncolors = len(winlist)
        color = sns.cubehelix_palette(ncolors)[i_win]
        window = bace_pair_str( (2*windows_default[i_win]+1)*file_bin_width )
        plt.plot(centers,hist, color=color, label=window)
        plt.plot(cutoffs[i_win],0,"+",color=color)
    if type(title) != type(None):
        plt.title(title)
    plt.legend()
    plt.xlabel("Chi-seq Singnal")
    plt.ylabel("Count")
    plt.show()

# --------------------------------------------------
#     Multi-processing tools
# -----------------------------------------------

def run_multiWindow(args, function, cutoff_after1_by_window=None):
    """Use mulitprossing to run function with different windows.

    Args:
        args (dict): Passed on to function
        function (callable): Of form function(half_widt1=..., **args)
        cutoff_after1_by_window (list): list of cutoff_after's to also pass
    """
    invals=[]
    for i_win, half_width1 in enumerate(windows_default):
        my_args=args.copy()
        my_args["half_width1"]=half_width1
        if type(cutoff_after1_by_window) != type(None):
            my_args["cutoff_after1"]=cutoff_after1_by_window[i_win]
        invals.append(my_args)

    with Pool(31) as my_pool:
        output=my_pool.map(function, invals)
    return output

def bace_pair_str(bace_pairs):
    if bace_pairs < 800:
        return "%dbp"%int(bace_pairs)
    elif bace_pairs < 800000:
        return "%.2fkb"%(bace_pairs/1000)
    else:
        return "%.2fMb"%(bace_pairs/1000000)


def plot_optimum(pickle_name, title=None, logy=False):
    import matplotlib.pyplot as plt
    #(half_windows, dot_products)=pickle.load(open(pickle_name,"rb"))
    data=pickle.load(open(pickle_name,"rb"))
    half_windows=data["windows_default"]
    dot_products=data["dot_products"]
    if logy:
        plt.loglog((np.array(half_windows)*2+1)*file_bin_width, dot_products,"-o")
    else:
        plt.semilogx((np.array(half_windows)*2+1)*file_bin_width, dot_products,"-o")
    if type(title) != type(None):
        plt.title(title)
    plt.xlabel("window in bp")
    plt.ylabel("predition accuracy")
    plt.show()

# ------------------------------------------
#    Procedures
# ------------------------------------------

chip_dirs = {
    '../LADs/meth_AGO4450/':'AG_H3K9me3',
    '../LADs/meth_Nhfl/':'Nhlf_H3K9me3',
    '../h3K9me3_chroms_IMR90_hg19/':'IMR90_H3K9me3'}
LAD_files = {}
for ii in range(1,9):
    LAD_files['../../LAD_data/LAD_DATA'+str(ii)]='DamID'+str(ii)

def run_data_sets():
    my_locs = genomic_locations(list(LAD_files.keys()))
    for chip_dir, chip_name in chip_dirs.items():
        name = chip_name+'vs_DamID'
        raw_optimum_window(chip_dir, 'all', pickle_name=name,
                           fraction_marked=0.5, my_locs=my_locs)

def plot_optimum_window_data():
    from matplotlib import pyplot as plt
    import seaborn as sns

    ii = 0
    for chip_dir, chip_name in chip_dirs.items():
        chip_label = chip_name
        color = sns.color_palette('deep',len(chip_dirs))[ii]
        ii = ii+1
        name = chip_name+'vs_DamID'
        data = pickle.load(open(name,"rb"))
        plt.semilogx(data['windows_default']*file_bin_width, data['correlation'],
                     label=chip_label, color=color)
        chip_label=None

    plt.legend()
    plt.title('DamID vs windowed ChIP-seq correlation')
    plt.xlabel('ChIP-seq Window Half Width (bp)')
    plt.ylabel('Correlation')
    plt.show()



shared_data = {}

def my_correlation(half_width):
    my_locs = shared_data['my_locs']
    set1 = shared_data['set1']
    chip_vals = get_window_average_at(set1, half_width, my_locs.chroms, my_locs.loc,
                      len(my_locs))
    return list(stats.pearsonr( chip_vals, my_locs.values ))

def raw_optimum_window(chip_dir, LAD_file, pickle_name='window_data',
                       fraction_marked=0.5, my_locs=None):
    set1 = genomic_vector(chip_dir) # Chip-seq
    Chip_cutoff = get_all_chromosome_cut(set1, upper_fraction=fraction_marked)
    set1.apply_filter({"cutoff":Chip_cutoff})
    fraction_actual = set1.mean
    print("Fraction marked with Chip: %f ..."%(fraction_actual))

    if my_locs is None:
        my_locs = genomic_locations(LAD_file)


    shared_data['my_locs'] = my_locs
    shared_data['set1'] = set1
    #all_args = []
    #for half_width in windows_default:
    #    all_args.append( {'half_width':half_width, 'my_locs':my_locs,
    #                      'set1':set1} )

    with Pool(31) as my_pool:
        output = my_pool.map(my_correlation, windows_default)
    output = np.array(output)
    pearson_r = output[:,0]
    p_values = output[:,1]

    data={"chip_dir":chip_dir, "LAD_file":LAD_file, "windows_default":windows_default,
          "fraction_marked":fraction_marked, "correlation":pearson_r,
          "p_values":p_values}
    pickle.dump(data, open(pickle_name,"wb"))

def chip_DamID_histogram(half_width):
    my_locs = shared_data['my_locs']
    set1 = shared_data['set1']
    chip_vals = get_window_average_at(set1, half_width, my_locs.chroms, my_locs.loc,
                                      len(my_locs))
    chip_bins = np.linspace(0.0,1.0,21)
    DamID_bins = np.linspace(-8.0, 8.0, 65)

    twoD_hist = np.histogram2d(chip_vals, my_locs.values,
                               bins=[chip_bins, DamID_bins])
    return twoD_hist

def run_chip_DamID_histogram(fraction_marked=0.5,
                             LAD_file='../../LAD_data/LAD_DATA1'):
    my_locs = genomic_locations(LAD_file)
    shared_data['my_locs'] = my_locs

    for chip_dir, chip_name in chip_dirs.items():
        set1 = genomic_vector(chip_dir) # Chip-seq
        Chip_cutoff = get_all_chromosome_cut(set1, upper_fraction=fraction_marked)
        set1.apply_filter({"cutoff":Chip_cutoff})
        fraction_actual = set1.mean
        shared_data['set1'] = set1

        half_widths = [0, 103, 4839]
        with Pool(31) as my_pool:
            output = my_pool.map(chip_DamID_histogram, half_widths)

        for ii, half_width in enumerate(half_widths):
            name = 'hist2D_'+chip_name+'_'+str(half_width)
            data={"count":output[ii][0], "chip_edges":output[ii][1],
                  "DamID_edges":output[ii][2], "LAD_file":LAD_file,
                  "chip_dir":chip_dir, "half_width":half_width,
                  "fraction_marked":fraction_marked}
            pickle.dump(data, open(name,"wb"))

def plot_2D_histogram():
    from matplotlib.image import NonUniformImage
    import matplotlib.pyplot as plt
    import seaborn as sns
    for chip_dir, chip_name in chip_dirs.items():
        half_widths = [0, 103, 4839]
        for ii, half_width in enumerate(half_widths):
            name = 'hist2D_'+chip_name+'_'+str(half_width)
            data = pickle.load(open(name,"rb"))
            extent = [data['chip_edges'][0], data['chip_edges'][-1],
                      data['DamID_edges'][0], data['DamID_edges'][-1]]
            print(extent)
            plt.imshow(data['count'].transpose(), interpolation='nearest', origin='low',
                       extent = extent, aspect='auto',
                       cmap=sns.cubehelix_palette(light=1, as_cmap=True))
            plt.title(chip_name+', Half Width=%5.2e bp'%(data['half_width']*200))
            plt.xlabel('ChIP-seq')
            plt.ylabel('log2 DamID enrichment')
            plt.ylim((-4,4))
            plt.show()
            #im = NonUniformImage(interpolation='bilinear')
            #xcenters = (data['chip_edges'][:-1] + data['chip_edges'][1:] )/2
            #ycenters = (data['DamID_edges'][:-1] + data['DamID_edges'][1:] )/2
            #im.set_data(xcenters, ycenters, data['count'])



def chip_mean_by_chromosome(my_chip_dirs=chip_dirs, fraction_marked=0.5):
    """
    Args:
        my_chip_dirs (dict): example {"../path/to/meth_Nhfl/':'Nhlf_H3K9me3'}

    Returns:
        Dictionary of dictionaries for format data['chip_name']['chr1'] = avj
        chip content.
    """
    data = {}
    for chip_dir, chip_name in my_chip_dirs.items():
        set1 = genomic_vector(chip_dir) # Chip-seq
        Chip_cutoff = get_all_chromosome_cut(set1, upper_fraction=fraction_marked)
        set1.apply_filter({"cutoff":Chip_cutoff})
        fraction_actual = set1.mean

        chip_average = {}
        for chromosome, chr_end in hg19_chrom_lengths.items():
            chip_average[chromosome] = set1.get_chromosome_average(chromosome,
                                                        exclude_centromere=True)
        data[chip_name] = chip_average
    return data
def DamID_by_chromosome():
    """
    """
    my_locs = genomic_locations(list(LAD_files.keys()))
    return my_locs.average()
def plot_damID_vs_Chip(damID_data, chip_datas):
    import matplotlib.pyplot as plt
    import seaborn as sns
    ii = 0
    colors = sns.color_palette('deep', len(chip_datas))
    for cell_type, chip_data in chip_datas.items():
        label = cell_type
        color = colors[ii]
        ii +=1
        for chromosome, chip_value in chip_data.items():
            if chromosome not in damID_data.keys():
                print(chromosome)
                continue
            y = damID_data[chromosome]
            x = chip_value
            print([x,y])
            plt.plot(x,y,'o',color=color, label=label)
            plt.text(x,y+0.006,chromosome[3:], color=color, fontsize='8')
            label = None
    plt.xlabel('Mean H3K9me3 ChIP')
    plt.ylabel('Mean log2 DamID enrichment')
    plt.legend()
    plt.show()


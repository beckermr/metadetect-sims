import sys
import logging
import numpy as np

logger = logging.getLogger(__name__)

class TryAgainError(Exception):
    """
    signal to skip this image(s) and try a new one
    """
    def __init__(self, message):

        # Call the base class constructor with the parameters it needs
        Exception.__init__(self, message)

def setup_logging(level):
    if level=='info':
        l=logging.INFO
    elif level=='debug':
        l=logging.DEBUG
    elif level=='warning':
        l=logging.WARNING
    elif level=='error':
        l=logging.ERROR
    else:
        l=logging.CRITICAL

    logging.basicConfig(stream=sys.stdout, level=l)

def log_pars(pars, fmt='%8.3g',front=None):
    """
    print the parameters with a uniform width
    """

    s = []
    if front is not None:
        s.append(front)
    if pars is not None:
        fmt = ' '.join( [fmt+' ']*len(pars) )
        s.append( fmt % tuple(pars) )
    s = ' '.join(s)

    logger.debug(s)

class Namer(object):
    """
    create strings with a specified front prefix
    """
    def __init__(self, front=None, back=None):
        if front=='':
            front=None
        if back=='' or back=='noshear':
            back=None

        self.front=front
        self.back=back

        if self.front is None and self.back is None:
            self.nomod=True
        else:
            self.nomod=False



    def __call__(self, name):
        n = name
        if not self.nomod:
            if self.front is not None:
                n = '%s_%s' % (self.front, n)
            if self.back is not None:
                n = '%s_%s' % (n, self.back)
        
        return n

def convert_run_to_seed(run):
    """
    convert the input config file name to an integer for use
    as a seed
    """
    import hashlib

    h = hashlib.sha256(run.encode('utf-8')).hexdigest()
    seed = int(h, base=16) % 2**30 

    logger.info("got seed %d from run %s" % (seed,run))

    return seed

def get_trials_nsplit(c):
    """
    split into chunks
    """
    from math import ceil

    ntrials = c['ntrials']

    tmsec = c['desired_hours']*3600.0

    sec_per = c['sec_per']

    ntrials_per = int(round( tmsec/sec_per ) )

    nsplit = int(ceil( ntrials/float(ntrials_per) ))

    time_hours = ntrials_per*sec_per/3600.0

    logger.info("ntrials requested: %s" % (ntrials))
    logger.info('seconds per image: %s sec per with rand: %s' % (c['sec_per'],sec_per))
    logger.info('nsplit: %d ntrials per: %d time (hours): %s' % (nsplit,ntrials_per,time_hours))


    return ntrials_per, nsplit, time_hours

def get_trials_per_job_mpi(njobs, ntrials):
    """
    split for mpi
    """
    return int(round(float(ntrials)/njobs))


#
# matching by row,col
#

def match_truth(data, truth, radius_arcsec=0.2, pixel_scale=0.263):
    """
    get indices in the data that match truth catalog by x,y position
    """

    radius_pixels = radius_arcsec/pixel_scale

    print("matching")

    allow=1
    mdata, mtruth = close_match(
        data['x'],
        data['y'],
        truth['x'],
        truth['y'],
        radius_pixels,
        allow,
    )

    nmatch=mdata.size
    ntot=data.size
    frac=float(nmatch)/ntot
    print('        matched %d/%d %.3f within '
          '%.3f arcsec' % (nmatch, ntot, frac,radius_arcsec))

    return mdata

def close_match(t1,s1,t2,s2,ep,allow,verbose=False):
    """
    Find the nearest neighbors between two arrays of x/y

    parameters
    ----------
    x1, y1: scalar or array
         coordinates of a set of points.  Must be same length.
    x2, y2: scalar or array
         coordinates of a second set of points.  Must be same length.
    ep: scalar
         maximum match distance between pairs (pixels)
    allow: scalar
         maximum number of matches in second array to each element in first array.
    verbose: boolean
         make loud

    Original by Dave Johnston, University of Michigan, 1997
         
    Translated from IDL by Eli Rykoff, SLAC

    modified slightly by erin sheldon
    """
    t1=np.atleast_1d(t1)
    s1=np.atleast_1d(s1)
    t2=np.atleast_1d(t2)
    s2=np.atleast_1d(s2)

    n1=t1.size
    n2=t2.size

    matcharr=np.zeros([n1,allow],dtype='i8')
    matcharr.fill(-1)
    ind=np.arange(n2,dtype='i8')
    sor=t2.argsort()
    t2s=t2[sor]
    s2s=s2[sor]
    ind=ind[sor]
    runi=0
    endt=t2s[n2-1]

 
    for i in range(n1):
        t=t1[i]
        tm=t-ep
        tp=t+ep
        in1=_binary_search(t2s,tm)  # I can improve this?
        
        if in1 == -1:
            if (tm < endt) : in1=0
        if in1 != -1:
            in1=in1+1
            in2=in1-1
            jj=in2+1
            while (jj < n2):
                if (t2s[in2+1] < tp):
                    in2+=1
                    jj+=1
                else :
                    jj=n2
            if (n2 == 1) :
                in2=0  # hmmm

            if (in1 <= in2):
                if (n2 != 1) :
                    check = s2s[in1:in2+1]
                    tcheck = t2s[in1:in2+1]
                else :
                    check = s2s[0]
                    tcheck=t2s[0]
                s=s1[i]
                t=t1[i]
                offby=abs(check-s)
                toffby=abs(tcheck-t)
                good=np.where(np.logical_and(offby < ep,toffby < ep))[0]+in1
                ngood=good.size
                if (ngood != 0) :
                    if (ngood > allow) :
                        offby=offby[good-in1]
                        toffby=toffby[good-in1]
                        dist=np.sqrt(offby**2+toffby**2)
                        good=good[dist.argsort()]
                        ngood=allow
                    good=good[0:ngood]
                    matcharr[i,0:ngood]=good
                    runi=runi+ngood
        

    if verbose:
        print("total put in bytarr:",runi)

    #matches=np.where(matcharr != -1)[0]
    matches=np.where(matcharr != -1)
    #if (matches.size == 0):
    if (matches[0].size == 0):
        if verbose:
            print("no matches found")
        m1=np.array([])
        m2=np.array([])
        return m1,m2

    m1 = matches[0] % n1
    m2 = matcharr[matches]
    m2 = ind[m2].flatten()

    if verbose:
        print(m1.size,' matches')

    return m1,m2



def _binary_search(arr,x,edgedefault=False,round=False):
    n=arr.size
    if (x < arr[0]) or (x > arr[n-1]):
        if (edgedefault):
            if (x < arr[0]): index = 0
            elif (x > arr[n-1]): index = n-1
        else: index = -1
        return index

    down=-1
    up=n
    while (up-down) > 1:
        mid=down+(up-down)//2
        if x >= arr[mid]:
            down=mid
        else:
            up=mid

    index=down

    if (round) and (index != n-1):
        if (abs(x-arr[index]) >= abs(x-arr[index+1])): index=index+1

    return index

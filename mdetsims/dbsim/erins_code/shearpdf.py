import numpy
from ngmix import Shape
from esutil.stat import print_stats

def get_shear_pdf(shconf, rng):

    if isinstance(shconf, dict):

        if shconf['type'] == 'const-dist':
            # a seed is specified and we generate them
            pdf = ConstShearGenerator(
                shconf['seed'],
                shconf['nshear'],
                min_shear=shconf['min_shear'],
                max_shear=shconf['max_shear'],
            )
        else:
            raise ValueError("only shear 'const-dist' for now")

    else:
        pdf = ConstShearSelector(shconf, rng)


    return pdf

class ShearGeneratorBase(object):
    def get_shear(self):
        raise NotImplementedError("implement get_shear()")

class ConstShearSelector(ShearGeneratorBase):
    def __init__(self, shears, rng):

        self.rng=rng

        if not isinstance(shears[0], list):
            shears = [shears]

        shears = [Shape(s[0],s[1]) for s in shears]

        self.nshear=len(shears)
        self.shears=shears

    def get_shear(self):
        """
        return a random shear from the input list, plus an index
        """
        ri = self.rng.randint(0, self.nshear)
        return self.shears[ri], ri

class ConstShearGenerator(ShearGeneratorBase):
    def __init__(self, seed, nshear, min_shear=0.1, max_shear=0.08):

        self.seed=seed
        self.nshear=nshear
        self.min_shear=min_shear
        self.max_shear=max_shear

        self.gen_shear()

    def gen_shear(self):
        self.rng=numpy.random.RandomState(seed=self.seed)
        g = self.rng.uniform(
            low=self.min_shear,
            high=self.max_shear,
            size=self.nshear,
        )
        theta = self.rng.uniform(
            low=0.0,
            high=numpy.pi*2,
            size=self.nshear,
        )

        g1=g*numpy.cos(2.0*theta)
        g2=g*numpy.sin(2.0*theta)

        print("generated shears:")
        print_stats(g1)
        print_stats(g2)

        shears=[]
        for i in xrange(self.nshear):
            shears.append( Shape(g1[i], g2[i]) )

        self.shears=shears

    def get_shear(self):
        """
        return a random shear from the input list, plus an index
        """
        ri = self.rng.randint(0, self.nshear)
        return self.shears[ri], ri



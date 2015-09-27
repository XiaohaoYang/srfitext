#!/usr/bin/env python
# imports for DE and GA
from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.algorithms import varAnd, varOr
from srfitext.gatools import DE, GA
from srfitext.gatools import cxTwoPointsCopy, wirtepop, uniform, mutPolynomialBounded

##########################################################
# Differential evolution
##########################################################


def preprocess(recipe):
    recipe.verbose = 0

    creator.create("FitnessM", base.Fitness, weights=(-1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessM)
    # init toolbox()
    toolbox = base.Toolbox()
    bl, bh = recipe.getBounds2Flat()
    toolbox.register("attr_float", uniform, bl, bh)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # extra
    toolbox.bl = bl
    toolbox.bh = bh
    return toolbox


def deapDE(recipe, cr=0.25, f=1, mu=300, ngen=20, *args, **kwargs):
    '''differential evolution using deap package
    '''
    ndim = len(recipe.getValues())
    toolbox = preprocess(recipe)
    toolbox.register("select", tools.selRandom, k=3)
    toolbox.register("evaluate", recipe.scalarResidualTuple)

    rv = DE(toolbox, ndim, cr, f, mu, ngen, *args, **kwargs)
    wirtepop('pop.dat', rv['hof'])
    return rv


def deapGA(recipe, cxpb=0.5, mutpb=0.2, ngen=40, mu=2000, *args, **kwargs):
    '''differential evolution using deap package
    '''
    recipe.verbose = 0

    toolbox = preprocess(recipe)
    toolbox.register("mate", cxTwoPointsCopy)
    toolbox.register("mutate", mutPolynomialBounded, eta=1.0,
                     low=toolbox.bl, up=toolbox.bh, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", recipe.scalarResidualTuple)

    rv = GA(toolbox, cxpb, mutpb, ngen, mu, *args, **kwargs)
    wirtepop('pop.dat', rv['hof'])
    return rv

##########################################################
# Differential evolution
##########################################################


def RMC(recipe, steps=50000, *args, **kwargs):

    liveplot = LivePlot(recipe, plotstep=100, savestru='')
    cont = recipe._contributions.values()[0]
    gen = cont._generators.values()[0]
    phase = gen._phase
    gen._calc.evaluatortype = 'OPTIMIZED'

    plt.ion()

    counter = 0
    rescurr = recipe.scalarResidualRMC()[0]
    while counter < steps:
        phase.xyz.notify()
        counter += 1
        # recipe.scale = counter % 10.0
        randi = np.random.randint(-1, phase.n)
        xyz = phase.xyz.value[randi]
        newxyz = xyz + np.random.normal(0, 1, 3) * 0.1
        phase.updateXYZi(randi, newxyz)
        resnew = recipe.scalarResidualRMC()[0]
        if resnew >= rescurr:
            deltares = resnew - rescurr
            rand = np.random.exponential() * 0.01
            if deltares > rand:
                phase.updateXYZi(randi, newxyz)
                # print "%f rejected" % resnew
            else:
                print "%f accepted" % resnew
                rescurr = resnew
        else:
            print "%f accepted" % resnew
            rescurr = resnew
        # liveplot(0)
    saveStruOutput(gen.stru, 'fitresult')
    return
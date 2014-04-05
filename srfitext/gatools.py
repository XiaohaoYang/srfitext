import random
import numpy as np
import scipy as sp
import itertools

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools
from deap.algorithms import varAnd, varOr

######################################################################
# Algorithm
######################################################################

def DE(toolbox, ndim, cr=0.4, f=0.5, mu=300, ngen=20, *args, **kwargs):
    '''
    differential evolution using deap package
    '''
    # init stats
    stats, logbook, record, pop, hof = initMisc(toolbox, npop=mu, nhof=100, similar=np.array_equal)

    for g in range(1, ngen):
        offspring = list(map(toolbox.clone, pop))
        for k, agent in enumerate(offspring):
            a, b, c = toolbox.select(pop)
            rcr = np.random.random(ndim)
            index = rcr < cr
            temp = a + f * (b - c)
            agent[index] = temp[index]

        fitnesses = toolbox.map(toolbox.evaluate, offspring)
        for ind, fit in itertools.izip(offspring, fitnesses):
            ind.fitness.values = fit

        pop = [a if a.fitness > b.fitness else b for a, b in itertools.izip(pop, offspring)]
        hof.update(pop)

        if kwargs.has_key('callback'):
            kwargs['callback'](1, p=hof[0])

        record = stats.compile(pop)
        logbook.record(gen=g, evals=len(pop), **record)
        print(logbook.stream)

    print "Best individual is ", hof[0].fitness.values[0]
    if len(hof[0]) < 20:
        print hof[0]

    return {'x':hof[0],
            'population':pop,
            'stats':stats,
            'hof':hof,
            }

def GA(toolbox, cxpb=0.5, mutpb=0.2, ngen=40, mu=2000, *args, **kwargs):
    '''
    '''
    stats, logbook, record, pop, hof = initMisc(toolbox, npop=mu, nhof=100, similar=np.array_equal)
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        # Update the hall of fame with the generated individuals
        hof.update(offspring)
        if kwargs.has_key('callback'):
            kwargs['callback'](1, p=hof[0])
        # Replace the current population by the offspring
        pop[:] = offspring
        # Append the current generation statistics to the logbook
        record = stats.compile(pop) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print logbook.stream

    print "Best individual is ", hof[0].fitness.values[0]
    if len(hof[0]) < 20:
        print hof[0]

    return {'x':hof[0],
            'population':pop,
            'stats':stats,
            'hof':hof,
            }
######################################################################
# tools
######################################################################


def evalRecipeGA(parray, recipe, pconverter):
    plist = pconverter.toList(parray)
    return recipe.scalarResidualTuple(plist)

def initMisc(toolbox, npop=300, nhof=100, similar=np.array_equal):
    # init stats
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    pop = toolbox.population(n=npop)
    hof = tools.HallOfFame(nhof, similar=similar)

    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    hof.update(pop)
    record = stats.compile(pop) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print logbook.stream

    return stats, logbook, record, pop, hof


def cxTwoPointsCopy(ind1, ind2):
    """Execute a two points crossover with copy on the input individuals. The
    copy is required because the slicing in numpy returns a view of the data,
    which leads to a self overwritting in the swap operation. For example,
    ::
    
        >>> import numpy
        >>> a = numpy.array((1,2,3,4))
        >>> b = numpy.array((5.6.7.8))
        >>> a[1:3], b[1:3] = b[1:3], a[1:3]
        >>> print(a)
        [1 6 7 4]
        >>> print(b)
        [5 6 7 8]
    """
    size = len(ind1)
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = \
        ind2[cxpoint1:cxpoint2].copy(), ind1[cxpoint1:cxpoint2].copy()

    return ind1, ind2

def wirtepop(filename, pop):
    '''write the data of population into files.
    in each line, the first value is the fitness value, the rest is the value of each individual in population
    
    :param pop: population
    :param filename: str, filename
    :return: non, but write data into file
    '''
    lines = []
    if hasattr(pop[0], 'fitness'):
        for ind in pop:
            line = "%9.5f " % ind.fitness.values[0]
            line = line + ' '.join([str(p) for p in ind])
            lines.append(line)
    else:
        for sw in pop:
            for ind in sw:
                line = "%9.5f " % ind.fitness.values[0]
                line = line + '  '.join([str(p) for p in ind])
                lines.append(line)
    rv = "\r\n".join(lines) + "\r\n"
    f = open(filename, 'w')
    f.write(rv)
    f.close()
    return

def uniform(low, up, size=None):
    '''generate a list of random value with uniform distribution in (low, up), usually used in generating a new individual
    
    :param low: float or list of float, the lower bound of random value
    :param up: float or list of float, the upper bound of random value
    :param size: None or int, if None, return the list depends on low and up, if int and low and up are float
        return a list of random float with length equal to size
    :return: list of random number
    '''
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]

def mutPolynomialBounded(individual, eta, low, up, indpb):
    """Polynomial mutation as implemented in original NSGA-II algorithm in
    C by Deb.
    
    :param individual: Individual to be mutated.
    :param eta: Crowding degree of the mutation. A high eta will produce
                a mutant resembling its parent, while a small eta will
                produce a solution much more different.
    :param low: A value or a sequence of values that is the lower bound of the
                search space.
    :param up: A value or a sequence of values that is the upper bound of the
               search space.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    
    if isinstance(individual, np.ndarray):
        ind1 = np.random.rand(size) <= indpb
        rand = np.random.rand(size)
        ind2 = rand < 0.5
        ind11 = np.logical_and(ind1, ind2)
        ind22 = np.logical_and(ind1, np.logical_not(ind2))
        
        x = individual
        xl = low
        xu = up
        delta_1 = (x - xl) / (xu - xl)
        delta_2 = (xu - x) / (xu - xl)
        mut_pow = 1.0 / (eta + 1.)
        
        xy = 1.0 - delta_1
        val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
        delta_q1 = val ** mut_pow - 1.0
        
        xy = 1.0 - delta_2
        val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
        delta_q2 = 1.0 - val ** mut_pow
        
        x[ind11] = x[ind11] + delta_q1[ind11] * (xu[ind11] - xl[ind11])
        x[ind22] = x[ind22] + delta_q1[ind22] * (xu[ind22] - xl[ind22])
        x = np.maximum(x, xl)
        x = np.minimum(x, xu)
        individual[:] = x
        
    else:
        for i in xrange(size):
            if random.random() <= indpb:
                x = individual[i]
                xl = low[i]
                xu = up[i]
                delta_1 = (x - xl) / (xu - xl)
                delta_2 = (xu - x) / (xu - xl)
                rand = random.random()
                mut_pow = 1.0 / (eta + 1.)
    
                if rand < 0.5:
                    xy = 1.0 - delta_1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * xy ** (eta + 1)
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta_2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * xy ** (eta + 1)
                    delta_q = 1.0 - val ** mut_pow
    
                x = x + delta_q * (xu - xl)
                x = min(max(x, xl), xu)
                individual[i] = x
    return individual,

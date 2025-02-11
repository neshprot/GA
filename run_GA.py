import configparser
from functools import partial

from constraints import Constraints, constraint_distances, constraint_max_charge, constraint_max_num_changes

from evolution import *
from logger import FileLogger
from utils import *

# PARSING CONFIG
config = configparser.ConfigParser()
config.read('config.ini')

# задаём некоторые константы из config
pdb_file = config['PDB']['File']
descriptors = [float(x) for x in config['PDB']['Descriptors'].split()]
cros_prob = float(config['PARAMS']['CrosProb'])
mut_prob = float(config['PARAMS']['MutProb'])
mut_num = int(config['PARAMS']['MutNum'])
eval_param = float(config['PARAMS']['EvalParam'])
pop_size = int(config['PARAMS']['PopSize'])
weights = [float(x) for x in config['PARAMS']['Weights'].split()]
compute_lmb_inf = config['COMPUTING']['ComputeLambdaInf']
compute_lmb_ouf = config['COMPUTING']['ComputeLambdaOuf']
computed_proteins_path = config['COMPUTING']['ComputedProteinsFileName']
result_file_name = config['COMPUTING']['ResultFileName']
positionsset = {int(x) for x in config['PARAMS']['PositionsSet'].split()}
stop_step = int(config['PARAMS']['StopStep'])
attempts = int(config['PARAMS']['Attempts'])
tred_number = int(config['PARAMS']['TredNumber'])
pop_count = int(config['PARAMS']['PopNum'])
use_computed = config.getboolean('PARAMS', 'UseComputed')
logger = FileLogger(result_file_name)
# 3.62 9.84 3.32

# GENERATING CONSTRAINTS
constraints = Constraints()

coordinates = read_coordinates(pdb_file)
sequence = read_sequence(pdb_file)

# функции ограничений
f1 = partial(constraint_distances, min_distance=5.0, coords=coordinates, positions_set=positionsset)
f2 = partial(constraint_max_charge, max_charge=7)
f3 = partial(constraint_max_num_changes, max_num_changes=mut_num)

constraints.add(f1)
constraints.add(f2)
constraints.add(f3)

evolve = ProteinEvolution(pop_set=[], save_file='computed_proteins', logger=logger, positionsset=positionsset, checker=constraints,
                          input_file=compute_lmb_inf, output_file=compute_lmb_ouf, tred_number=tred_number)
evolve.generate_populations(default_sequence=sequence, default_descriptors=descriptors,
                               pop_size=pop_size, pop_count=pop_count, mut_prob=mut_prob, mut_num=mut_num, cros_prob=cros_prob)
ini_step = 1
sets, pulls, probs, consts = read_replacements('sites')

iteration, step = 1, 0
best_fitness = []
best_descriptors = []
for population in evolve.pop_set:
    best_f, best_d = population.get_best_fitness()
    best_fitness.append(best_f)
    best_descriptors.append(best_d)

while step < stop_step:
    logger(f"Iteration: {iteration}\n")
    proteins_for_computing = []
    chunk_numbers = []
    for pop_num, population in enumerate(evolve.pop_set):
        evolve.crossover(population, attempts=attempts, pop_num=pop_num+1)
        evolve.mutation(population, attempts=attempts, iteration=iteration, sets=sets, pulls=pulls, probs=probs,
                        consts=consts, pop_num=pop_num+1)
        population.clear_population()
        chunk_proteins, chunk_number = evolve.compute_push(population, pop_num+1)
        proteins_for_computing += chunk_proteins
        chunk_numbers.append(chunk_number)
    evolve.compute_input(proteins_for_computing, chunk_numbers)
    evolve.shuffle_pops(pop_size, iteration)

    step_checker = []
    for pop_num, population in enumerate(evolve.pop_set):
        cur_best_fitness, cur_best_descriptors = population.get_best_fitness()
        if cur_best_fitness > best_fitness[pop_num]:
            best_fitness[pop_num] = cur_best_fitness
            best_descriptors[pop_num] = cur_best_descriptors
            step_checker.append(True)
        else:
            step_checker.append(False)

        logger(f"Current population {pop_num+1}:\n")
        evolve.print_current_population(population)
        logger(f"The best value {pop_num+1}: {' '.join(map(str, best_descriptors[pop_num]))}\n")
        logger("\n")
    #if any(step_checker):
    #    step = 0
    #else:
    step += 1
    logger(f"Step/Stop {step}/{stop_step}\n\n")
    iteration += 1

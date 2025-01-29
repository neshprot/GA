import os
import random
import time
from copy import copy
import numpy as np
from data import Protein, Gene, Population

Pull = "ARNDVHGQEILKMPSYTWFV"  # список 20 существующих аминокислот

# класс с основными функциями эволюции
class ProteinEvolution():
    def __init__(self, pop_set, save_file, logger, positionsset={}, checker=None, input_file=None, output_file=None, tred_number=1):
        super().__init__()
        self._save_file = save_file
        self._computed = dict()
        self._logger = logger
        self._positionsset = positionsset
        self._checker = checker
        self._pop_set = pop_set
        self._input_file = input_file
        self._output_file = output_file
        self._tred_number = tred_number

    @property
    def positionsset(self):
        return self._positionsset

    @property
    def pop_set(self):
        return self._pop_set

    @pop_set.setter
    def pop_set(self, value):
        self._pop_set = value

    # mutation function
    def mutation(self, population, attempts=1, iteration=1, sets={}, pulls={}, probs={}, consts={}, pop_num=1):
        """

        :param attempts: число попыток инциниализации на один protein
        :return:
        """

        num_of_changed = 0      # кол-во измененных белков
        old_population = copy(population.population)
        # перебор белков в популяции
        for protein in old_population:
            new_protein = copy(protein)
            # условие возникновения мутации(с вероятностью mut_prob)
            if random.random() < population.mut_prob:
                attempt = 0
                num_of_changed += 1
                num_mut = 0
                mutations = []
                while attempt < attempts and num_mut <= population.mut_num and num_mut < iteration:
                    position = random.choice(tuple(self.positionsset))
                    if position in mutations:
                        continue
                    old_gene = new_protein.genes[position - 1]

                    # no changes for CHARGED and TRP
                    set_name = 'Set1'
                    for name, sites in sets.items():
                        if old_gene.value in sites:
                            set_name = name
                            continue
                    pull = random.choices(probs[set_name][0], weights=probs[set_name][1][:-1])[0]
                    if random.random() > probs[set_name][1][-1]*0.01:
                        new_value = old_gene.value
                    else:
                        new_value = random.choice(pulls[pull])

                    if int(position) in consts:
                        if new_value in pulls[consts[int(position)][0]]:
                            continue

                    new_gene = Gene(value=new_value)
                    new_protein.update_gene(position - 1, new_gene)

                    # проверка стабильности белка
                    if self.is_stable_protein(new_protein) and new_protein.num_changes <= iteration:
                        mutations.append(position)
                        num_mut += 1
                    else:
                        # Restore old gene
                        new_protein.update_gene(position - 1, old_gene)
                    attempt += 1
            population.add(new_protein)
        self._logger(
            f"Mutation for population {pop_num}: I will try to change {num_of_changed} proteins... {num_of_changed} proteins changed\n")

    def crossover(self, population, attempts=1, pop_num=1):
        for_cross = []  # белки для кроссовера

        for protein in population.population:
            new_protein = copy(protein)
            # условие на кроссинговер
            if random.random() < population.cros_prob:
                for_cross.append(new_protein)
            else:
                population.add(new_protein)

        # проверка на четное число белков в списке на кроссовер
        if len(for_cross) % 2 == 1:
            population.add(for_cross.pop())

        random.shuffle(for_cross)

        need = 0
        real = 0

        pair_cros_prob = 0.5  # crossover pair probability
        # цикл кроссовера(перемешивания генов белков)
        for protein1, protein2 in zip(for_cross[0:-1:2], for_cross[1::2]):
            need += 2
            new_protein1, new_protein2 = protein1, protein2
            for attempt in range(attempts):
                attempt_protein1, attempt_protein2 = copy(new_protein1), copy(new_protein2)
                mut_num = 0
                for i, (gene1, gene2) in enumerate(zip(attempt_protein1.genes, attempt_protein2.genes)):
                    if mut_num > population.mut_num:
                        continue
                    if random.random() < pair_cros_prob:
                        new_gene1 = Gene(value=gene2.value)
                        new_gene2 = Gene(value=gene1.value)
                        attempt_protein1.update_gene(i, new_gene1)
                        attempt_protein2.update_gene(i, new_gene2)
                        mut_num += 1

                if self.is_stable_protein(attempt_protein1) and self.is_stable_protein(attempt_protein2):
                    new_protein1 = attempt_protein1
                    new_protein2 = attempt_protein2
                    real += 2
                    break

            population.add(new_protein1)
            population.add(new_protein2)
        self._logger(f"Crossover for population {pop_num}: I will try to change {need} proteins... {real} proteins changed\n")

        # selection function
    def selection(self, population, pop_size):
        # increasing value
        new_population = sorted(population.population, key=lambda x: x.fitness, reverse=True)[:pop_size:]
        if pop_size > len(new_population):
            for _ in range(pop_size - len(new_population)):
                protein = copy(random.choices(new_population, np.linspace(1, 0, len(new_population)))[0])
                new_population.append(protein)

        population.population = new_population

    def shuffle_pops(self, pop_size, iteration):
        for population in self._pop_set:
            self.selection(population, pop_size)
        if iteration % 5 == 0:
            idx_array = [i for i in range(len(self._pop_set))]
            random.shuffle(idx_array)
            for idx1, idx2 in zip(idx_array[:-1] + [idx_array[0]], idx_array[1:] + [idx_array[-1]]):
                swap_num = round(len(self._pop_set[idx1].population) * 0.3)
                prots = self._pop_set[idx1].get_first_num(swap_num)
                self._pop_set[idx2].del_end(swap_num)
                self._pop_set[idx2].add_prots_to_end(prots)
        for population in self._pop_set:
            random.shuffle(population.population)
            population.compute_fitness()



    def compute_push(self, population, pop_num):
        proteins_for_computing = []
        tred_number = self._tred_number  # put in config.ini

        # Find existing calcs
        for protein in population.population:
            if protein.sequence not in self._computed:
                proteins_for_computing.append(protein)

        # Print to output files
        chunk_size = len(proteins_for_computing) // tred_number
        remainder = len(proteins_for_computing) % tred_number

        # split the array
        chunks = []
        start = 0
        for i in range(tred_number):
            end = start + chunk_size + (1 if i < remainder else 0)
            chunks.append(proteins_for_computing[start:end])
            start = end
        number_of_chunks = 0
        # safe each chunk to a separate file
        for i, chunk in enumerate(chunks):
            if not chunk:
                continue
            number_of_chunks += 1
            with open(".tempfile", "w") as ouf:
                for protein in chunk:
                    for idx, g1, g2 in protein.get_differences():
                        ouf.write(f"{g1}/{idx}/{g2} ")
                    ouf.write("\n")
            os.rename(".tempfile", f'{self._output_file}_{pop_num}_{i+1}')

        return proteins_for_computing, number_of_chunks

    def compute_input(self, proteins_for_computing, chunk_numbers):
        # Wait results
        for pop_num, population in enumerate(self._pop_set):
            descriptors = []
            if not chunk_numbers[pop_num]:
                continue
            for tred_counter in range(1, chunk_numbers[pop_num] + 1):
                while not os.path.exists(f'{self._input_file}_{pop_num+1}_{tred_counter}'):
                    time.sleep(1)

                # Read results and save
                with open(f'{self._input_file}_{pop_num+1}_{tred_counter}') as inf:
                    for line in inf.readlines():
                        values = line.split()
                        descriptors.append([float(value) for value in values])
                os.remove(f'{self._input_file}_{pop_num+1}_{tred_counter}')
            len_for_each_pop = len(descriptors)
            for i in range(len_for_each_pop):
                self.save_computing(proteins_for_computing[i].sequence, descriptors[i])
            proteins_for_computing = proteins_for_computing[len_for_each_pop:]
            # Write values to proteins
            for protein in population.population:
                values = self._computed[protein.sequence]
                protein.descriptor = values
            population.compute_fitness()

    def is_stable_protein(self, protein):
        if self._checker is not None:
            return self._checker.check(protein)
        return True

    def save_computing(self, sequence, descriptors):
        if sequence not in self._computed:
            self._computed[sequence] = descriptors
            with open(self._save_file, 'a') as f:
                f.write(f"{sequence} {' '.join(map(str, descriptors))}\n")

    def read_computed(self):
        if os.path.exists(self._save_file):
            with open(self._save_file, 'r') as inf:
                for line in inf.readlines():
                    words = line.split()
                    self._computed[words[0]] = [float(desc) for desc in words[1::]]

    def print_current_population(self, population):
        for protein in population.population:
            self._logger(f"{protein.sequence}, descriptors {' '.join(map(str, protein.descriptor))}"
                         f" num of changes {protein.num_changes}\n")
            for idx, g1, g2 in protein.get_differences():
                self._logger(f"{g1}/{idx}/{g2} ")
            self._logger("\n")

    def generate_populations(self, default_sequence, default_descriptors, pop_size, pop_count, mut_prob, mut_num, cros_prob):
        distribution_weights = np.linspace(0, 1, pop_count)
        self.read_computed()
        for i in range(pop_count):
            population = []
            self.save_computing(default_sequence, default_descriptors)

            while len(population) < pop_size:
                protein = Protein.create_protein(default_sequence, default_sequence, descriptor=default_descriptors)
                population.append(protein)
            pop = Population(population, mut_prob, mut_num, cros_prob,
                             [distribution_weights[i], distribution_weights[::-1][i]])
            pop.compute_fitness()
            self._pop_set.append(pop)

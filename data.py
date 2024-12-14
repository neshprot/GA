from itertools import count
from typing import List, Tuple
import numpy as np

AMINOACIDS = "RHKDESTNQCGPAVILMFYW"
NON_CHARGED = "STNQCGPAVILMFYW"
CHARGED = "RHKDE"
POSITIVE_CHARGE = "RHK"
NEGATIVE_CHARGE = "DE"
POLAR = "STCNQKRYDEH"
NONPOLAR = "GAVILPMFW"
BULKY = "IML"


class Gene:
    def __init__(self, value) -> None:
        self.__value = value

    @property
    def charged(self):
        return self.value in CHARGED

    @property
    def charge(self):
        if self.value in POSITIVE_CHARGE:
            return 1
        if self.value in NEGATIVE_CHARGE:
            return -1
        return 0

    @property
    def polared(self):
        return self.value in POLAR

    @property
    def value(self):
        return self.__value

    @value.setter
    def value(self, x):
        self.__value = x

    def __copy__(self):
        copy = Gene(self.__value)

        return copy

    def __eq__(self, other: 'Gene'):
        return self.__value == other.__value


class Protein:
    @classmethod
    def create_protein(cls, sequence, origin_sequence, descriptor):
        protein = Protein(sequence, origin_sequence)

        protein.__descriptor = descriptor

        # Calc charge
        protein.__charge = 0
        for gene in protein.__genes:
            protein.__charge += gene.charge

        # Calc changes
        protein.__num_changes = 0
        for x1, x2 in zip(sequence, origin_sequence):
            if x1 != x2:
                protein.__num_changes += 1

        return protein

    def __init__(self, sequence, origin_sequence) -> None:
        self.__genes = [Gene(x) for x in sequence]
        self.__origin_genes = [Gene(x) for x in origin_sequence]
        self.__num_changes = None
        self.__descriptor = None
        self.__charge = None
        self.__dominance_depth = None
        self.__crowding_distance = None
        self.__fitness = None

    @property
    def charge(self):
        return self.__charge

    @property
    def sequence(self):
        return ''.join([str(x.value) for x in self.__genes])

    @property
    def origin_sequence(self):
        return ''.join([str(x.value) for x in self.__origin_genes])

    @property
    def genes(self):
        return self.__genes

    @property
    def num_changes(self):
        return self.__num_changes

    @property
    def descriptor(self):
        return self.__descriptor

    @descriptor.setter
    def descriptor(self, new_value):
        self.__descriptor = new_value

    @property
    def dominance_depth(self):
        return self.__dominance_depth

    @property
    def crowding_distance(self):
        return self.__crowding_distance

    @dominance_depth.setter
    def dominance_depth(self, new_value):
        self.__dominance_depth = new_value

    @crowding_distance.setter
    def crowding_distance(self, new_value):
        self.__crowding_distance = new_value

    @property
    def fitness(self):
        return self.__fitness

    @fitness.setter
    def fitness(self, value):
        self.__fitness = value

    def update_gene(self, idx, gene):
        """
        Обновляет значение (value) в гене.

        :param idx: позиция гена в sequence. 0 <= idx < len(sequqence)
        :param gene: новый ген
        :return:
        """
        if gene.value != self.__genes[idx].value:
            # Update num changes
            if self.__genes[idx].value != self.__origin_genes[idx].value:
                self.__num_changes -= 1

            # Update charge
            self.__charge -= self.__genes[idx].charge

            # Change gene
            self.__genes[idx] = gene

            # Update charge
            self.__charge += self.__genes[idx].charge

            # Update num changes
            if self.__genes[idx].value != self.__origin_genes[idx].value:
                self.__num_changes += 1

            # Update value

            self.__descriptor = None

            self.__dominance_depth = None

            self.__crowding_distance = None

            self.__fitness = None

    def get_differences(self) -> List[Tuple[int, Gene, Gene]]:
        differences = []
        for idx, g1, g2 in zip(count(1), self.origin_sequence, self.sequence):
            if g1 != g2:
                differences.append((idx, g1, g2))
        return differences

    def get_gene(self, idx):
        """
        Возвращает копию гена по индексу
        :param idx: индекс гена, 0 <= idx < len(genes)
        :return:
        """
        return self.__genes[idx].__copy__()

    def __copy__(self):
        copy = Protein(self.sequence, self.origin_sequence)
        copy.__descriptor = self.__descriptor
        copy.__charge = self.__charge
        copy.__num_changes = self.__num_changes
        copy.__dominance_depth = self.__dominance_depth
        copy.__crowding_distance = self.__crowding_distance
        copy.__fitness = self.__fitness

        return copy

    def __eq__(self, other: 'Protein'):
        return self.sequence == other.sequence


class Population:
    def __init__(self, population, mut_prob, mut_num, cros_prob, weights):
        self._population = population
        self._mut_prob = mut_prob
        self._mut_num = mut_num
        self._cros_prob = cros_prob
        self._weights = weights

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value):
        self._population = value

    @property
    def mut_prob(self):
        return self._mut_prob

    @mut_prob.setter
    def mut_prob(self, value):
        self._mut_prob = value

    @property
    def mut_num(self):
        return self._mut_num

    @mut_num.setter
    def mut_num(self, value):
        self._mut_num = value

    @property
    def cros_prob(self):
        return self._mut_num

    @cros_prob.setter
    def cros_prob(self, value):
        self._cros_prob = value

    def input_file(self):
        return self._input_file

    def output_file(self):
        return self._output_file

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, value):
        self._weights = value

    def add(self, value):
        self._population.append(value)

    def add_prots_to_end(self, value):
        self._population = self.population + value

    def del_end(self, number):
        new_pop = self._population
        self._population = new_pop[:-number:]

    def safe_first_num(self, number):
        new_pop = self.population
        self._population = new_pop[:number:]

    def get_pos(self, number):
        return self._population[number]

    def get_first_num(self, number):
        return self.population[:number:]

    def get_best_fitness(self):
        best_protein = max(self.population, key=lambda x: x.fitness)
        return best_protein.fitness, best_protein.descriptor

    def clear_population(self):
        new_population = []
        sequences = []
        for prot in self.population:
            if prot.sequence not in sequences:
                new_population.append(prot)
                sequences.append(prot.sequence)
        self.population = new_population

    def compute_fitness(self):
        min_descr = np.min(np.array([prot.descriptor for prot in self.population]), axis=0)
        max_descr = np.max(np.array([prot.descriptor for prot in self.population]), axis=0)
        for protein in self.population:
            def norm(vals):
                descriptors = []
                for idx, descr in enumerate(vals):
                    if max_descr[idx] != min_descr[idx]:
                        descriptors.append((descr - min_descr[idx]) / (max_descr[idx] - min_descr[idx]))
                    else:
                        descriptors.append(0)
                return descriptors

            def linear(vals):
                return np.dot(np.array(vals), np.array(self._weights))

            fit = linear(norm(protein.descriptor))
            protein.fitness = fit

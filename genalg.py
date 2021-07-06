import sys
from numpy.random import choice, randint, uniform
from numpy import ceil, zeros

class GenAlg:
    def __init__(self, data, pickle_path='results.pkl', n=10000, mating_rate=0.3, mutation_rate=0.1):
        self.data = data
        self.pickle_path = pickle_path
        self.n = n
        self.mating_rate = mating_rate
        self.mutation_rate = mutation_rate
        self.generation = 0

    def start_evolution(self):
        self.init_population()
        try:
            while True:
                self.calculate_fitness()
                self.create_children()
                self.mutation()
                self.generation += 1

                last_generation = self.population[self.population['Fitness'] != float('inf')]
                print('GENERATION {}'.format(self.generation))
                print('Best Fitness: {}\n'.format(last_generation.iloc[0]['Fitness']))
        except KeyboardInterrupt:
            last_generation.drop('New').reset_index().to_pickle(self.pickle_path)

    def mutation(self):
        individs_to_mutate = self.population.sample(frac=self.mutation_rate)

        for i, row in individs_to_mutate.iterrows():
            gene_to_mutate = choice(list(self.pools.keys()))
            self.population.loc[i, gene_to_mutate] = choice(self.pools[gene_to_mutate])

    def create_children(self):
        mean_fitness = self.population['Fitness'].mean()
        self.population['New'] = round(self.population['Fitness']/mean_fitness)
        parent_pool = self.population[self.population['New'] > 0]

        for _, row in parent_pool.iterrows():
            if uniform(0, 1) > self.mating_rate:
                self.population = self.population.append([row]*int(row['New']), ignore_index=True)
            else:
                new_child = self.mate([row, parent_pool.sample().squeeze()])
                self.population = self.population.append(new_child, ignore_index=True)

        self.population.sort_values(by=['Fitness'], inplace=True, ascending=False, ignore_index=True)
        self.population = self.population.iloc[:self.n]

    def calculate_fitness(self):
        if self.generation == 0:
            print('Initializing population...')
            fitness = zeros(self.n)
            for i, row in self.population.iterrows():
                progress = ceil(i*100.0/(self.n-1))
                sys.stdout.write('\r{}%'.format(round(progress)))
                sys.stdout.flush()
                fitness[i] = self.fitness(self.population.iloc[i])

            self.population['Fitness'] = fitness
            print('\nDone\n')
        else:
            self.population['Fitness'] = self.population.apply(self.fitness, axis=1)

    def mate(self, parents):
        draw = randint(0, 2, len(self.population.columns))
        new_child = {}
        for d, col in zip(draw, self.population.columns):
            new_child[col] = parents[d][col]

        new_child['Fitness'] = float('inf')
        return new_child

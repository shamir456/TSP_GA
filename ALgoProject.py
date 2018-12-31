import numpy as np
import matplotlib.pyplot as plt
import random

#Authors: Shameer-K164030
#         Hasnain-K164029
#

def is_empty(any_structure):
    if any_structure:
        return True
    else:
        return False

def swap(a, b):
    temp =a
    a = b
    b = temp
    return

def make_sym_matrix(n,vals):
  m = np.zeros([n,n])
  xs,ys = np.indices((0,n))
  
  return m




infile=open('/home/zen/Desktop/gr21.tsp','r')
Name = infile.readline().strip().split()[1]
TSP= infile.readline().strip().split()[1]
Comment=infile.readline().strip().split()[1]
Dimension=infile.readline().strip().split()[1]
EdgeWeightType=infile.readline().strip().split()[1]
EdgeWeightFormat=infile.readline().strip().split()[1]
infile.readline()


print(Name + " "+ Dimension+ " "+ TSP +" "+EdgeWeightFormat)

distance=[]

list2=[]

list1=infile.readlines()
infile.close()

list1.pop()




for i in list1:
       list1=list(map(int,i.strip().split()[0:]))
       if(is_empty(list1)):
        list2.append(list1)

arr=[]
for i in range(0,len(list2)):
     for j in list2[i]:
        arr.append(j)
list2.clear()

list1.clear()
#imp work
for i in arr:
  if(i==0):

        #print(list2)
        list1.extend(list2.copy())

       
        list2.clear()

  else:
   list2.append(i)
Full_matrix = np.zeros((21,21),dtype=int)
inds = np.tril_indices_from(Full_matrix, k = -1)
# use [:] to copy the list and avoid any problem if the init1al list is further needed
Full_matrix[inds] = list1[:]
Full_matrix[(inds[1], inds[0])] = list1[:]
dis_matrix = np.zeros((21,21),dtype=int)
dis_matrix=Full_matrix






class GA(object):
    def __init__(self, popsize, length, crossprob, mutationprob, dis_matrix):
        self.popsize = popsize
        self.length = length
        self.crossprob = crossprob
        self.mutationprob = mutationprob
        self.dis_matrix = dis_matrix

    # Population, an np two-dimensional matrix of population length * chromosome length, with a value of 0 or 1
    # Initialize the population and generate an np two-dimensional matrix of population number * chromosome length with a value of 0 or 1
    def pop_init(self):
        pop = np.zeros((self.popsize, self.length))
        for i in range(self.popsize):
            arr = np.arange(self.length, dtype=int)
            np.random.shuffle(arr)
            pop[i] = arr

        pop = pop.astype(int)

        return pop

    def mutate(self,individual, mutationRate):
        for swapped in range(len(individual)):
            if (random.random() < mutationRate):
                swapWith = int(random.random() * len(individual))

                city1 = individual[swapped]
                city2 = individual[swapWith]

                individual[swapped] = city2
                individual[swapWith] = city1
        return individual

    # ----------------------------------------------------------------------
    def mutatePopulation(self,population, mutationRate):
        mutatedPop = []

        for ind in range(0, len(population)):
            mutatedInd = self.mutate(population[ind], mutationRate)
            mutatedPop.append(mutatedInd)
        return mutatedPop

    # Cross operation
    def crossover(self, pop):

        for i in range(pop.shape[0]):
            randnum = np.random.rand(1)
            if randnum < self.crossprob:
                crosspoint = np.random.randint(1, pop.shape[1], size=1)  # Randomly generate an intersection
                crosspoint = crosspoint[0]
                crosschrom = np.random.randint(pop.shape[0], size=1)  # Randomly indicate the chromosome to cross with
                crosschrom = crosschrom[0]

               
                newchrom1 = np.hstack((pop[i, 0:crosspoint], pop[crosschrom, crosspoint:]))
                newchrom2 = np.hstack((pop[crosschrom, 0:crosspoint], pop[i, crosspoint:]))
                pop[i] = newchrom1
                pop[crosschrom] = newchrom2


                for j in range(crosspoint, self.length):
                    while pop[i, j] in pop[i, 0:crosspoint]:
                        for k in range(crosspoint):

                            if pop[i, j] == pop[i, k]:
                                pop[i, j] = pop[crosschrom, k]

                for j in range(crosspoint, self.length):
                    while pop[crosschrom, j] in pop[crosschrom, 0:crosspoint]:
                        for k in range(crosspoint):
                            if pop[crosschrom, j] == pop[crosschrom, k]:
                                pop[crosschrom, j] = pop[i, k]

        return pop

    def find(self,child,parent):
        for i in range(0,self.length-1):
            if(child==parent[i]):
                break

        return i




    def crossover2(self,pop):
        parent1=np.zeros(pop.shape[1],dtype=int)
        parent2=np.zeros(pop.shape[1],dtype=int)
        child1=np.empty(pop.shape[1],dtype=int)
        child2=np.empty(pop.shape[1],dtype=int)
        for i in range(pop.shape[0]):
              randnum = np.random.rand(1)
              if randnum <self.crossprob:
                  parent1=pop[i-1]
                  parent2=pop[i]
                  st1=0
                  st2=0
                  index1=0
                  index2=0
                  child1[st1]=parent2[st1]
                  while(st1<self.length-1):
                      index1=self.find(child1[st1],parent1)
                      index2=self.find(parent2[index1],parent1)
                      child2[st2]=parent2[index2]
                      #find=child2[st2]
                      st1=st1+1
                      index1=self.find(parent2[index2],parent1)
                      child1[st1]=parent2[index1]
                      st2=st2+1
                      index1=0
                      index2=0
        print(child1)
        print(child2)

         #     pop[i-1]=child1
          #    pop[i]=child2


        return pop





    def findchild(self,child,p1,p2):
        value=p1[0]
        flag=np.zeros(self.length,dtype=bool)

        for i in range(self.length):
            flag[i]=True

        i=0
        while(True):
            child[i]=value
            flag[i] = False
            value=p2[i]
            i=self.find(value,p1)
            if(child[i]==value):
                break
            #print(i)


        for i in range(0,self.length):
             if(flag[i]==True):
                 child[i]=p2[i]




        return child


    def cx(self,pop):
        parent1=np.zeros(pop.shape[1],dtype=int)
        parent2=np.zeros(pop.shape[1],dtype=int)
        child1=np.empty(pop.shape[1],dtype=int)
        child2=np.empty(pop.shape[1],dtype=int)
        newpop = np.zeros_like(pop)
        i=1
        while(i<pop.shape[0]):
              randnum = np.random.rand(1)
              if randnum <self.crossprob:
                  parent1=pop[i-1]
                  parent2=pop[i]
                  child1=self.findchild(child1,parent1,parent2)
                  child2 = self.findchild(child2, parent2, parent1)
                  newpop[i-1]=child1
                  newpop[i]=child2
                  i=i+1
              else:
                  i=i-1

        pop=newpop

        return pop

    # Mutation operation
    def mutation(self, pop):
        for i in range(self.popsize):
            for j in range(self.length):
                random_num = np.random.rand(1)[0]
                if random_num < self.mutationprob:
                    mutation_gene = np.random.randint(self.length, size=1)[0]
                    buffer = pop[i, j]
                    pop[i, j] = pop[i, mutation_gene]
                    pop[i, mutation_gene] = buffer


        return pop


    # Choose an action and use roulette
    def selection(self, pop):

        # Calculate the distance of each chromosome
        dist = np.zeros(self.popsize)
        for i in range(len(dist)):
            for j in range(self.length):
                if j == self.length - 1:
                    dist[i] = dist[i] + self.dis_matrix[pop[i, j], pop[i, 0]]
                else:
                    dist[i] = dist[i] + self.dis_matrix[pop[i, j], pop[i, j + 1]]

        # Find fitness for each individual decoded
        fitness = np.zeros(self.popsize)

        fitness = 1 / dist

        # Store optimal value
        bestfitness = np.max(fitness)
        best_y = 1 / bestfitness
        best_x = pop[np.argmax(fitness)]

        # Choose through roulette

        # Get the fitness table by accumulating
        fitnesslist = fitness
        for i in range(1, self.popsize):
            fitnesslist[i] = fitnesslist[i] + fitnesslist[i - 1]

        newpop = np.zeros_like(pop)
        # Selecting through the fitness table
        for i in range(self.popsize):
            randnum = np.random.rand(1) * fitnesslist[i]
            for j in range(self.popsize):
                if randnum < fitnesslist[j]:
                    newpop[i] = pop[j]
                    break
        return newpop, bestfitness,best_x,best_y





if __name__ == '__main__':
    popsize = 150
    length = 21
    crossprob = 0.80
    mutationprob = 0.10
    ga = GA(popsize, length, crossprob, mutationprob, dis_matrix)
    init_pop = ga.pop_init()


    iteration = 500  #generations
    Bestfitness = np.zeros(iteration)
    Meanfitness = np.zeros(iteration)
    Best_y = np.zeros(iteration)
    Best_x = np.zeros((iteration, length))


   # print(cross_pop.tolist())

    for i in range(iteration):

       #cross_pop = ga.cx(init_pop)

         #mut_pop = ga.mutation(cross_pop)
       init_pop = ga.pop_init()
       selected_pop, Bestfitness[i],Best_x[i],Best_y[i] = ga.selection(init_pop)
       #cross_pop = ga.crossover(init_pop)


       #crossx_pop = ga.crossover(selected_pop)
       cross_pop=ga.cx(selected_pop)
       #mut_pop = ga.mutation(cross_pop)
       mut_pop=arr = np.array(ga.mutatePopulation(cross_pop,ga.mutationprob))
       #print(mut_pop)
       init_pop, Bestfitness[i],Best_x[i],Best_y[i] = ga.selection(mut_pop)
       the_best_one = np.argmax(Bestfitness)
       the_best_y = Best_y[the_best_one]
       the_best_x = Best_x[the_best_one]
       print('\n%d' % (i))
       print(the_best_y)
       print(the_best_x)
       #print(the_best_one)


     # #Plotting result
    x = np.arange(iteration)
    plt.figure()
    plt.plot(x, Bestfitness)
    plt.xlabel('generation')
    plt.ylabel('fitness')
    plt.title('bestfitness')
    plt.show()


    plt.figure()
    plt.plot(x, Best_y)
    plt.xlabel('generation')
    plt.ylabel('y')
    plt.title('shortest_dis')
    plt.show()
    #
the_best_one = np.argmax(Bestfitness)
the_best_y = Best_y[the_best_one]
the_best_x = Best_x[the_best_one]
print('The best route %d' % the_best_y)
the_best_x = the_best_x.astype(int)
print(the_best_x)


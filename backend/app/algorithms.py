"""Algoritmos puros: cuadrícula de cuatro direcciones, costo unitario."""
from collections import deque
from heapq import heappop, heappush
from itertools import permutations
import random


def manhattan(a,b):
    return abs(a[0]-b[0])+abs(a[1]-b[1])


def search(grid,start,goal,method='bfs'):
    if method not in ('bfs','astar'): raise ValueError('Método desconocido')
    if not grid or not grid[0] or any(len(r)!=len(grid[0]) for r in grid):
        raise ValueError('La cuadrícula debe ser rectangular y no vacía')
    start,goal=tuple(start),tuple(goal)
    def valid(p):
        return 0<=p[0]<len(grid) and 0<=p[1]<len(grid[0]) and grid[p[0]][p[1]]==0
    if not valid(start) or not valid(goal): raise ValueError('Origen o destino inválido')
    queue=deque([start]); heap=[(manhattan(start,goal),0,start)]
    parents={start:None}; costs={start:0}; expanded=[]
    while queue if method=='bfs' else heap:
        if method=='bfs': current=queue.popleft()
        else:
            _,g,current=heappop(heap)
            if g!=costs[current]: continue
        expanded.append(current)
        if current==goal:
            path=[]; node=current
            while node is not None: path.append(node);node=parents[node]
            return dict(method=method,path=list(reversed(path)),cost=costs[current],expanded=len(expanded))
        for dr,dc in ((-1,0),(0,1),(1,0),(0,-1)):
            nxt=(current[0]+dr,current[1]+dc)
            if valid(nxt) and costs[current]+1<costs.get(nxt,float('inf')):
                costs[nxt]=costs[current]+1;parents[nxt]=current
                if method=='bfs': queue.append(nxt)
                else: heappush(heap,(costs[nxt]+manhattan(nxt,goal),costs[nxt],nxt))
    return dict(method=method,path=[],cost=None,expanded=len(expanded))


def route_cost(order,matrix):
    path=[0]+list(order)+[0]
    return sum(matrix[a][b] for a,b in zip(path,path[1:]))


def distance_matrix(grid,points):
    matrix=[]
    for a in points:
        row=[]
        for b in points:
            cost=search(grid,a,b,'bfs')['cost']
            if cost is None: raise ValueError('Existen visitas sin conexión')
            row.append(cost)
        matrix.append(row)
    return matrix


def hill_climbing(matrix):
    order=list(range(1,len(matrix)));initial=route_cost(order,matrix);history=[initial]
    while True:
        best=order;score=route_cost(order,matrix)
        for i in range(len(order)):
            for j in range(i+1,len(order)):
                neighbor=order.copy();neighbor[i],neighbor[j]=neighbor[j],neighbor[i]
                value=route_cost(neighbor,matrix)
                if value<score: best,score=neighbor,value
        if best==order: break
        order=best;history.append(score)
    return dict(method='hill_climbing',order=[0]+order+[0],cost=history[-1],initial=initial,history=history)


def genetic(matrix,seed=42,population_size=40,generations=80):
    rng=random.Random(seed);base=list(range(1,len(matrix)))
    population=[base.copy()]+[rng.sample(base,len(base)) for _ in range(population_size-1)]
    history=[]
    def score(x): return route_cost(x,matrix)
    for _ in range(generations):
        population.sort(key=score);history.append(score(population[0]))
        next_population=[x.copy() for x in population[:2]]
        while len(next_population)<population_size:
            a=min(rng.sample(population,3),key=score)
            b=min(rng.sample(population,3),key=score)
            # Cruce por orden: conserva una subsecuencia y completa sin duplicados.
            lo,hi=sorted(rng.sample(range(len(base)+1),2)) if base else (0,0)
            segment=a[lo:hi];remaining=[x for x in b if x not in segment]
            child=remaining[:lo]+segment+remaining[lo:]
            if len(child)>1 and rng.random()<0.25:
                i,j=rng.sample(range(len(child)),2);child[i],child[j]=child[j],child[i]
            next_population.append(child)
        population=next_population
    best=min(population,key=score);history.append(score(best))
    return dict(method='genetic',order=[0]+best+[0],cost=score(best),
                initial=score(base),history=history,seed=seed,
                population_size=population_size,generations=generations)


def exact(matrix):
    if len(matrix)>9: raise ValueError('Validación exacta limitada a ocho visitas')
    best=min(permutations(range(1,len(matrix))),key=lambda p:route_cost(p,matrix))
    return dict(method='exact',order=[0]+list(best)+[0],cost=route_cost(best,matrix))

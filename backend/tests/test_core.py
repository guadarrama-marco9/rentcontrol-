import unittest
from datetime import date
from itertools import permutations
from app.domain import alerts,truth_table
from app.demo import dataset,GRID,POINTS
from app.algorithms import search,distance_matrix,hill_climbing,genetic,exact,route_cost

class CoreTests(unittest.TestCase):
    def test_truth_table(self):
        table=truth_table();self.assertEqual(len(table),16)
        self.assertEqual(sum(r['A'] for r in table),9)
        for r in table:self.assertEqual(r['A'],(r['P'] and r['Q'] and r['R']) or r['S'])
    def test_partial_and_historical(self):
        rows=alerts(dataset(date(2026,10,27)),date(2026,10,27))
        self.assertEqual(len(rows),3)
        self.assertEqual(sum(r['balance_cents'] for r in rows),250000)
        self.assertEqual(len({r['id'] for r in rows}),3)
        self.assertTrue(any(r['kind']=='adeudo_historico' for r in rows))
    def test_due_today_is_not_overdue(self):
        p=dataset(date(2026,10,27));p[0]['charges'][0]['due']='2026-10-27'
        self.assertFalse(any(r['id']=='debt:c1' for r in alerts(p,date(2026,10,27))))
    def test_same_charge(self):
        p=dataset(date(2026,10,27))[1];p['charges'].append(dict(id='future',due='2027-01-01',amount_cents=100,paid_cents=0))
        self.assertEqual(alerts([p],date(2026,10,27)),[])
    def test_bfs_astar_and_valid_paths(self):
        for a in POINTS:
            for b in POINTS:
                bfs=search(GRID,a,b);astar=search(GRID,a,b,'astar')
                self.assertEqual(bfs['cost'],astar['cost'])
                for result in (bfs,astar):
                    self.assertEqual(result['path'][0],a);self.assertEqual(result['path'][-1],b)
                    for x,y in zip(result['path'],result['path'][1:]):
                        self.assertEqual(abs(x[0]-y[0])+abs(x[1]-y[1]),1)
                        self.assertEqual(GRID[y[0]][y[1]],0)
    def test_no_path(self):
        self.assertIsNone(search([[0,1,0]],(0,0),(0,2))['cost'])
    def test_bad_endpoint(self):
        with self.assertRaises(ValueError):search(GRID,(1,1),(0,0))
    def test_optimizers(self):
        matrix=distance_matrix(GRID,POINTS);opt=exact(matrix)['cost']
        for seed in range(10):
            for r in (hill_climbing(matrix),genetic(matrix,seed=seed)):
                self.assertEqual(sorted(r['order'][1:-1]),list(range(1,len(matrix))))
                self.assertEqual(r['cost'],route_cost(r['order'][1:-1],matrix))
                self.assertGreaterEqual(r['cost'],opt)
                self.assertLessEqual(r['cost'],r['initial'])
                self.assertTrue(all(a>=b for a,b in zip(r['history'],r['history'][1:])))
        self.assertEqual(genetic(matrix),genetic(matrix))

if __name__=='__main__':unittest.main()

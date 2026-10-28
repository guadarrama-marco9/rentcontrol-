from datetime import timedelta

def dataset(today):
    due=(today-timedelta(days=10)).isoformat()
    future=(today+timedelta(days=10)).isoformat()
    return [dict(id='p1',name='Departamento Alameda',contract_active=True,
        charges=[dict(id='c1',due=due,amount_cents=500000,paid_cents=300000)],
        maintenance=[dict(id='m1',urgent=True,status='pendiente',description='Fuga de agua en cocina')]),
        dict(id='p2',name='Casa Los Encinos',contract_active=True,
        charges=[dict(id='c2',due=due,amount_cents=650000,paid_cents=650000)],maintenance=[]),
        dict(id='p3',name='Estudio Centro',contract_active=False,
        charges=[dict(id='c3',due=due,amount_cents=400000,paid_cents=350000)],maintenance=[]),
        dict(id='p4',name='Departamento Universidad',contract_active=True,
        charges=[dict(id='c4',due=future,amount_cents=450000,paid_cents=0)],maintenance=[])]

GRID=[[0,0,0,0,0,0],[0,1,1,0,1,0],[0,0,0,0,1,0],
      [0,1,0,0,0,0],[0,0,0,1,1,0],[0,0,0,0,0,0]]
POINTS=[(0,0),(0,5),(5,5),(5,0),(2,2)]

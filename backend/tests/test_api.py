import os,unittest,tempfile
from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app

class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ,{'AUTH_REQUIRED':'0','STORAGE':'demo','RENTCONTROL_API_KEY':'','DEMO_DB_PATH':os.path.join(self.temp.name,'test.sqlite3')});self.env.start()
        self.client=TestClient(app)
    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()
    def test_create_property_persisted(self):
        payload={'name':'Casa nueva','address':'Calle Reforma 123','monthly_rent_cents':500000}
        response=self.client.post('/properties',json=payload)
        self.assertEqual(response.status_code,201)
        record=response.json()
        self.assertFalse(record['contract_active'])
        with TestClient(app) as other:
            self.assertIn(record,other.get('/properties').json())
            self.assertEqual(other.get('/dashboard').json()['metrics']['properties'],5)
    def test_invalid_property(self):
        for amount in (0,-1,1.5,True):
            self.assertEqual(self.client.post('/properties',json={'name':'Casa','address':'Calle 123','monthly_rent_cents':amount}).status_code,422)
    def test_property_requires_key(self):
        with patch.dict(os.environ,{'RENTCONTROL_API_KEY':'test'}):
            self.assertEqual(self.client.post('/properties',json={'name':'Casa','address':'Calle 123','monthly_rent_cents':100}).status_code,401)

    def test_dashboard_and_filter(self):
        r=self.client.get('/dashboard');self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['metrics']['overdue_cents'],250000)
        self.assertEqual(len(self.client.get('/alerts?kind=mantenimiento').json()),1)
        self.assertEqual(self.client.get('/alerts?priority=baja').json(),[])
    def test_csv_and_logic(self):
        r=self.client.get('/alerts.csv');self.assertEqual(r.status_code,200)
        self.assertIn('Departamento Alameda',r.text)
        self.assertEqual(len(self.client.get('/logic').json()['table']),16)
    def test_lab(self):
        r=self.client.get('/lab');self.assertEqual(r.status_code,200)
        self.assertEqual(r.json()['searches'][0]['cost'],r.json()['searches'][1]['cost'])
    def test_access_key(self):
        with patch.dict(os.environ,{'RENTCONTROL_API_KEY':'demo-test'}):
            self.assertEqual(self.client.get('/dashboard').status_code,401)
            self.assertEqual(self.client.get('/dashboard',headers={'X-API-Key':'demo-test'}).status_code,200)
    def test_empty_question(self):
        self.assertEqual(self.client.post('/agent',json={'message':''}).status_code,422)
    def test_agent_tool_cycle(self):
        from unittest.mock import MagicMock
        first=MagicMock();first.json.return_value={'message':{'role':'assistant','content':'','tool_calls':[{'function':{'name':'consultar_resumen','arguments':{}}}]}}
        second=MagicMock();second.json.return_value={'message':{'role':'assistant','content':'Hay tres alertas.'}}
        with patch('app.main.httpx.Client') as client:
            client.return_value.__enter__.return_value.post.side_effect=[first,second]
            r=self.client.post('/agent',json={'message':'¿Cuántas alertas hay?'})
            self.assertEqual(r.status_code,200);self.assertEqual(r.json()['answer'],'Hay tres alertas.')
    def test_reject_unknown_tool(self):
        from unittest.mock import MagicMock
        response=MagicMock();response.json.return_value={'message':{'role':'assistant','content':'','tool_calls':[{'function':{'name':'borrar_datos'}}]}}
        with patch('app.main.httpx.Client') as client:
            client.return_value.__enter__.return_value.post.return_value=response
            self.assertEqual(self.client.post('/agent',json={'message':'Consulta'}).status_code,502)

if __name__=='__main__':unittest.main()

class WorkflowTests(unittest.TestCase):
    setUp = ApiTests.setUp
    tearDown = ApiTests.tearDown
    def test_full_workflow(self):
        from app.main import today
        from datetime import timedelta
        p=self.client.post('/properties',json={'name':'Casa real','address':'Toluca centro 123','monthly_rent_cents':500000}).json()
        base='/properties/'+p['id']
        self.assertEqual(self.client.post(base+'/contract',json={'tenant':'Ana Pérez','start':today().isoformat(),'end':(today()+timedelta(days=365)).isoformat(),'monthly_rent_cents':500000}).status_code,200)
        charge=self.client.post(base+'/charges',json={'due':(today()-timedelta(days=1)).isoformat(),'amount_cents':500000}).json()
        path=base+'/charges/'+charge['id']+'/payments'
        self.assertEqual(self.client.post(path,json={'amount_cents':300000,'reference':'Transferencia'}).status_code,200)
        self.assertEqual(self.client.post(path,json={'amount_cents':300000}).status_code,422)
        row=next(a for a in self.client.get('/alerts').json() if a['property_id']==p['id'])
        self.assertEqual(row['balance_cents'],200000)
        self.client.post(path,json={'amount_cents':200000})
        self.assertFalse(any(a['property_id']==p['id'] for a in self.client.get('/alerts').json()))
        task=self.client.post(base+'/maintenance',json={'description':'Fuga de agua','urgent':True}).json()
        self.assertTrue(any(a['property_id']==p['id'] for a in self.client.get('/alerts').json()))
        self.assertEqual(self.client.post(base+'/maintenance/'+task['id']+'/resolve').status_code,200)
        self.assertFalse(any(a['property_id']==p['id'] for a in self.client.get('/alerts').json()))
        current=next(x for x in self.client.get('/properties').json() if x['id']==p['id'])
        self.assertEqual(len(current['payments']),2)
        self.assertEqual(sum(x['amount_cents'] for x in current['payments']),500000)

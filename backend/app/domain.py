from datetime import date
from itertools import product


def truth_table():
    return [dict(P=p,Q=q,R=r,S=s,C=p and q and r,A=(p and q and r) or s)
            for p,q,r,s in product((False,True), repeat=4)]


def alerts(properties, today):
    result=[]
    for p in properties:
        active=p['contract_active']
        # Existencia: Q y R deben pertenecer al MISMO cargo, no a cargos diferentes.
        overdue=[c for c in p['charges'] if date.fromisoformat(c['due']) < today
                 and c['amount_cents'] > c['paid_cents']]
        for c in overdue:
            balance=c['amount_cents']-c['paid_cents']
            result.append(dict(id=f"debt:{c['id']}",property_id=p['id'],property=p['name'],
                kind='cobranza' if active else 'adeudo_historico',priority='alta',
                reason=f"Cargo {c['id']} vencido el {c['due']}; saldo ${balance/100:,.2f}.",
                balance_cents=balance,rule='P ∧ Q ∧ R' if active else '¬P ∧ Q ∧ R'))
        for m in p['maintenance']:
            if m['urgent'] and m['status']!='resuelto':
                result.append(dict(id=f"maintenance:{m['id']}",property_id=p['id'],
                    property=p['name'],kind='mantenimiento',priority='critica',
                    reason=m['description'],balance_cents=0,rule='S'))
    return result


def metrics(properties, rows):
    charges=[c for p in properties for c in p['charges']]
    return dict(properties=len(properties),occupied=sum(p['contract_active'] for p in properties),
        charged_cents=sum(c['amount_cents'] for c in charges),
        collected_cents=sum(c['paid_cents'] for c in charges),
        overdue_cents=sum(r['balance_cents'] for r in rows),
        alerts_by_type={k:sum(r['kind']==k for r in rows)
                        for k in ('cobranza','adeudo_historico','mantenimiento')})

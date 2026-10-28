import csv
import io
import json
import os

from datetime import date, datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from . import property_store
from .algorithms import (
    distance_matrix,
    exact,
    genetic,
    hill_climbing,
    search,
)
from .auth import identify, tenants
from .auth import router as auth_router
from .demo import GRID, POINTS, dataset
from .domain import alerts, metrics, truth_table


app = FastAPI(
    title="RentControl",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8081",
    ).split(","),
    allow_methods=["GET", "POST"],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
        "Authorization",
    ],
)

app.include_router(auth_router)


def today():
    return datetime.now(
        ZoneInfo("America/Mexico_City")
    ).date()


def properties():
    mode = os.getenv("STORAGE", "demo")

    if mode == "demo":
        property_store.initialize(dataset(today()))
        return property_store.list_created()

    if mode != "mongo":
        raise HTTPException(
            503,
            "STORAGE debe ser demo o mongo",
        )

    try:
        from pymongo import MongoClient

        with MongoClient(
            os.environ["MONGO_URI"],
            serverSelectionTimeoutMS=3000,
        ) as client:
            collection = client[
                os.getenv("MONGO_DB", "rentcontrol")
            ].properties

            return list(
                collection.find({}, {"_id": 0})
            )

    except Exception:
        raise HTTPException(
            503,
            "MongoDB no disponible. Revisa la conexión.",
        )


def authorized(
    request: Request,
    x_api_key: str = Header(default=""),
    authorization: str = Header(default=""),
):
    if os.getenv("AUTH_REQUIRED", "1") != "0":
        user = identify(authorization)
    else:
        # Solo se utiliza para las pruebas del proyecto.
        user = {
            "id": 1,
            "name": "Pruebas",
            "email": "",
            "role": "admin",
        }

    configured = os.getenv("RENTCONTROL_API_KEY")

    if configured and x_api_key != configured:
        raise HTTPException(
            401,
            "Clave de acceso incorrecta",
        )

    if user["role"] != "admin":
        path = request.url.path

        if path == "/users":
            raise HTTPException(
                403,
                "Solo el administrador puede consultar usuarios",
            )

        if request.method != "GET" and path != "/agent":
            parts = path.strip("/").split("/")

            maintenance_report = (
                request.method == "POST"
                and len(parts) == 3
                and parts[0] == "properties"
                and parts[2] == "maintenance"
            )

            if not maintenance_report:
                raise HTTPException(
                    403,
                    "Esta operación requiere una cuenta administradora",
                )

            has_access = any(
                p["id"] == parts[1]
                and p.get("assigned_user_id") == user["id"]
                for p in properties()
            )

            if not has_access:
                raise HTTPException(
                    403,
                    "No tienes acceso a este inmueble",
                )

    return user


def visible_properties(user):
    rows = properties()

    if user["role"] == "admin":
        return rows

    return [
        p
        for p in rows
        if p.get("assigned_user_id") == user["id"]
    ]


def report(user):
    rows = visible_properties(user)
    alert_rows = alerts(rows, today())

    return {
        "mode": os.getenv("STORAGE", "demo"),
        "as_of": today().isoformat(),
        "properties": rows,
        "alerts": alert_rows,
        "metrics": metrics(rows, alert_rows),
    }


def mutate_property(identifier, operation):
    mode = os.getenv("STORAGE", "demo")

    if mode == "demo":
        properties()
        return property_store.mutate(
            identifier,
            operation,
        )

    if mode != "mongo":
        raise HTTPException(
            503,
            "Modo de almacenamiento no válido",
        )

    try:
        from pymongo import MongoClient

        with MongoClient(
            os.environ["MONGO_URI"],
            serverSelectionTimeoutMS=3000,
        ) as client:
            collection = client[
                os.getenv("MONGO_DB", "rentcontrol")
            ].properties

            original = collection.find_one(
                {"id": identifier}
            )

            if not original:
                raise HTTPException(
                    404,
                    "Propiedad no encontrada",
                )

            document = json.loads(
                json.dumps({
                    key: value
                    for key, value in original.items()
                    if key != "_id"
                })
            )

            result = operation(document)

            updated = collection.replace_one(
                original,
                document,
            )

            if updated.matched_count != 1:
                raise HTTPException(
                    409,
                    "El registro cambió. Actualiza antes de guardar.",
                )

            return result

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            503,
            "MongoDB no disponible",
        )


class PropertyCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    name: str = Field(min_length=3, max_length=120)
    address: str = Field(min_length=5, max_length=300)
    monthly_rent_cents: int = Field(
        strict=True,
        gt=0,
        le=100000000,
    )


class ContractCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    tenant: str = Field(min_length=3, max_length=120)
    start: date
    end: date
    monthly_rent_cents: int = Field(
        strict=True,
        gt=0,
        le=100000000,
    )


class ChargeCreate(BaseModel):
    due: date
    amount_cents: int = Field(
        strict=True,
        gt=0,
        le=100000000,
    )


class PaymentCreate(BaseModel):
    amount_cents: int = Field(
        strict=True,
        gt=0,
        le=100000000,
    )

    reference: str = Field(
        default="",
        max_length=120,
    )


class MaintenanceCreate(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
    )

    description: str = Field(
        min_length=5,
        max_length=1000,
    )

    urgent: bool = False


class AccessCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: int = Field(strict=True, gt=0)


class Question(BaseModel):
    message: str = Field(
        min_length=1,
        max_length=1000,
    )


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "0.1.0",
    }


@app.get("/dashboard")
def dashboard(user=Depends(authorized)):
    return report(user)


@app.get("/properties")
def get_properties(user=Depends(authorized)):
    return visible_properties(user)


@app.post(
    "/properties",
    status_code=201,
    dependencies=[Depends(authorized)],
)
def create_property(payload: PropertyCreate):
    document = {
        "id": str(uuid4()),
        **payload.model_dump(),
        "contract_active": False,
        "charges": [],
        "maintenance": [],
    }

    mode = os.getenv("STORAGE", "demo")

    if mode == "demo":
        property_store.insert(document)

    elif mode == "mongo":
        try:
            from pymongo import MongoClient

            with MongoClient(
                os.environ["MONGO_URI"],
                serverSelectionTimeoutMS=3000,
            ) as client:
                collection = client[
                    os.getenv("MONGO_DB", "rentcontrol")
                ].properties

                collection.insert_one(dict(document))

        except Exception:
            raise HTTPException(
                503,
                "No se pudo guardar la propiedad en MongoDB.",
            )

    else:
        raise HTTPException(
            503,
            "STORAGE debe ser demo o mongo",
        )

    return document


@app.post(
    "/properties/{identifier}/contract",
    dependencies=[Depends(authorized)],
)
def create_contract(
    identifier: str,
    payload: ContractCreate,
):
    if payload.end <= payload.start:
        raise HTTPException(
            422,
            "La fecha final debe ser posterior al inicio",
        )

    if not payload.start <= today() <= payload.end:
        raise HTTPException(
            422,
            "Las fechas deben incluir hoy para registrar un contrato vigente",
        )

    def operation(p):
        if p["contract_active"]:
            raise HTTPException(
                409,
                "La propiedad ya tiene contrato activo",
            )

        p["contract"] = {
            "id": str(uuid4()),
            "tenant": payload.tenant,
            "start": payload.start.isoformat(),
            "end": payload.end.isoformat(),
        }

        p["contract_active"] = True
        p["monthly_rent_cents"] = payload.monthly_rent_cents

        return p

    return mutate_property(identifier, operation)


@app.post(
    "/properties/{identifier}/charges",
    dependencies=[Depends(authorized)],
)
def create_charge(
    identifier: str,
    payload: ChargeCreate,
):
    def operation(p):
        if not p["contract_active"]:
            raise HTTPException(
                409,
                "Primero registra un contrato vigente",
            )

        charge = {
            "id": str(uuid4()),
            "due": payload.due.isoformat(),
            "amount_cents": payload.amount_cents,
            "paid_cents": 0,
        }

        p["charges"].append(charge)
        return charge

    return mutate_property(identifier, operation)


@app.post(
    "/properties/{identifier}/charges/{charge_id}/payments",
    dependencies=[Depends(authorized)],
)
def create_payment(
    identifier: str,
    charge_id: str,
    payload: PaymentCreate,
):
    def operation(p):
        charge = next(
            (
                c
                for c in p["charges"]
                if c["id"] == charge_id
            ),
            None,
        )

        if not charge:
            raise HTTPException(
                404,
                "Cargo no encontrado",
            )

        balance = (
            charge["amount_cents"]
            - charge["paid_cents"]
        )

        if payload.amount_cents > balance:
            raise HTTPException(
                422,
                "El pago supera el saldo pendiente",
            )

        payment = {
            "id": str(uuid4()),
            "charge_id": charge_id,
            "amount_cents": payload.amount_cents,
            "reference": payload.reference,
            "date": today().isoformat(),
        }

        charge["paid_cents"] += payload.amount_cents
        p.setdefault("payments", []).append(payment)

        return payment

    return mutate_property(identifier, operation)


@app.post(
    "/properties/{identifier}/maintenance",
    dependencies=[Depends(authorized)],
)
def create_maintenance(
    identifier: str,
    payload: MaintenanceCreate,
):
    def operation(p):
        item = {
            "id": str(uuid4()),
            "description": payload.description,
            "urgent": payload.urgent,
            "status": "pendiente",
            "date": today().isoformat(),
        }

        p["maintenance"].append(item)
        return item

    return mutate_property(identifier, operation)


@app.post(
    "/properties/{identifier}/maintenance/{item_id}/resolve",
    dependencies=[Depends(authorized)],
)
def resolve_maintenance(
    identifier: str,
    item_id: str,
):
    def operation(p):
        item = next(
            (
                m
                for m in p["maintenance"]
                if m["id"] == item_id
            ),
            None,
        )

        if not item:
            raise HTTPException(
                404,
                "Mantenimiento no encontrado",
            )

        item["status"] = "resuelto"
        item["resolved_date"] = today().isoformat()

        return item

    return mutate_property(identifier, operation)


@app.get(
    "/users",
    dependencies=[Depends(authorized)],
)
def list_users():
    return tenants()


@app.post(
    "/properties/{identifier}/access",
    dependencies=[Depends(authorized)],
)
def assign_access(
    identifier: str,
    payload: AccessCreate,
):
    exists = any(
        tenant["id"] == payload.user_id
        for tenant in tenants()
    )

    if not exists:
        raise HTTPException(
            422,
            "Selecciona una cuenta de usuario registrada",
        )

    def operation(p):
        p["assigned_user_id"] = payload.user_id

        return {
            "property_id": identifier,
            "user_id": payload.user_id,
        }

    return mutate_property(identifier, operation)


@app.get("/alerts")
def get_alerts(
    kind: str | None = None,
    priority: str | None = None,
    user=Depends(authorized),
):
    return [
        alert
        for alert in report(user)["alerts"]
        if (
            (not kind or alert["kind"] == kind)
            and (
                not priority
                or alert["priority"] == priority
            )
        )
    ]


@app.get("/alerts.csv")
def export(user=Depends(authorized)):
    stream = io.StringIO()
    writer = csv.writer(stream)

    writer.writerow([
        "ID",
        "Propiedad",
        "Tipo",
        "Prioridad",
        "Motivo",
        "Saldo MXN",
        "Regla",
    ])

    for alert in report(user)["alerts"]:
        writer.writerow([
            alert["id"],
            alert["property"],
            alert["kind"],
            alert["priority"],
            alert["reason"],
            alert["balance_cents"] / 100,
            alert["rule"],
        ])

    return Response(
        "\ufeff" + stream.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition":
                "attachment; filename=alertas.csv"
        },
    )


@app.get(
    "/logic",
    dependencies=[Depends(authorized)],
)
def logic():
    return {
        "rule": "(P ∧ Q ∧ R) ∨ S",
        "table": truth_table(),
    }


@app.get(
    "/lab",
    dependencies=[Depends(authorized)],
)
def lab():
    matrix = distance_matrix(GRID, POINTS)

    return {
        "grid": GRID,
        "points": POINTS,
        "matrix": matrix,
        "searches": [
            search(
                GRID,
                POINTS[0],
                POINTS[2],
                method,
            )
            for method in ("bfs", "astar")
        ],
        "routes": [
            hill_climbing(matrix),
            genetic(matrix),
            exact(matrix),
        ],
    }


@app.get("/explanations")
def explanations(user=Depends(authorized)):
    current = today()
    rows = visible_properties(user)
    details = []

    for p in rows:
        urgent = any(
            m["urgent"]
            and m["status"] != "resuelto"
            for m in p["maintenance"]
        )

        charges = []

        for charge in p["charges"]:
            due = date.fromisoformat(charge["due"])

            balance = (
                charge["amount_cents"]
                - charge["paid_cents"]
            )

            P = bool(p["contract_active"])
            Q = due < current
            R = balance > 0
            S = urgent

            charges.append({
                "id": charge["id"],
                "due": charge["due"],
                "amount_cents": charge["amount_cents"],
                "paid_cents": charge["paid_cents"],
                "balance_cents": balance,
                "days_overdue": max(
                    0,
                    (current - due).days,
                ),
                "P": P,
                "Q": Q,
                "R": R,
                "S": S,
                "collection": P and Q and R,
                "historical": (not P) and Q and R,
                "active_rule": (P and Q and R) or S,
            })

        details.append({
            "id": p["id"],
            "name": p["name"],
            "P": bool(p["contract_active"]),
            "S": urgent,
            "charges": charges,
        })

    data = report(user)
    summary = data["metrics"]

    charged = summary["charged_cents"]
    collected = summary["collected_cents"]
    total_properties = summary["properties"]
    occupied = summary["occupied"]

    return {
        "as_of": current.isoformat(),
        "properties": details,
        "table": truth_table(),
        "rule": "(P ∧ Q ∧ R) ∨ S",
        "historical_rule": "¬P ∧ Q ∧ R",
        "totals": {
            "charged_cents": charged,
            "collected_cents": collected,
            "pending_cents": charged - collected,
            "overdue_cents": summary["overdue_cents"],
            "collected_percent": (
                round(collected / charged * 100, 2)
                if charged
                else 0
            ),
            "occupied": occupied,
            "properties": total_properties,
            "occupancy_percent": (
                round(
                    occupied / total_properties * 100,
                    2,
                )
                if total_properties
                else 0
            ),
        },
    }


@app.post("/agent")
def agent(
    q: Question,
    user=Depends(authorized),
):
    tools = [{
        "type": "function",
        "function": {
            "name": "consultar_resumen",
            "description": (
                "Consulta alertas y totales de RentControl, "
                "sin datos personales."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    }]

    messages = [
        {
            "role": "system",
            "content": (
                "Eres RentIA. Responde en español. "
                "Usa consultar_resumen para datos del sistema. "
                "No inventes importes ni confirmes acciones. "
                "Solo tienes acceso de consulta."
            ),
        },
        {
            "role": "user",
            "content": q.message,
        },
    ]

    base = os.getenv(
        "OLLAMA_URL",
        "http://localhost:11434",
    )

    model = os.getenv(
        "OLLAMA_MODEL",
        "llama3.2",
    )

    try:
        with httpx.Client(timeout=90) as client:
            for _ in range(3):
                response = client.post(
                    base + "/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "tools": tools,
                        "stream": False,
                    },
                )

                response.raise_for_status()
                message = response.json()["message"]
                messages.append(message)

                calls = message.get("tool_calls", [])

                if not calls:
                    return {
                        "answer": message.get(
                            "content",
                            "Sin respuesta",
                        ),
                        "model": model,
                    }

                for call in calls:
                    function = call.get("function", {})

                    if function.get("name") != "consultar_resumen":
                        raise HTTPException(
                            502,
                            "Herramienta no permitida",
                        )

                    # Solo consulta los inmuebles visibles para esta cuenta.
                    data = report(user)

                    safe = {
                        "fecha": data["as_of"],
                        "modo": data["mode"],
                        "metricas": data["metrics"],
                        "nota": (
                            "Importes en centavos MXN. "
                            "Totales acumulados, no mensuales."
                        ),
                        "alertas": [
                            {
                                "tipo": alert["kind"],
                                "prioridad": alert["priority"],
                                "saldo_centavos": alert["balance_cents"],
                            }
                            for alert in data["alerts"]
                        ],
                    }

                    messages.append({
                        "role": "tool",
                        "tool_name": "consultar_resumen",
                        "content": json.dumps(
                            safe,
                            ensure_ascii=False,
                        ),
                    })

        raise HTTPException(
            502,
            "El agente excedió el límite de consultas",
        )

    except (
        httpx.HTTPError,
        KeyError,
        ValueError,
    ):
        raise HTTPException(
            503,
            "Ollama no respondió correctamente. "
            "Verifica el servicio y el modelo.",
        )
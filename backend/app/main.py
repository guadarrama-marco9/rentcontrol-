import csv
import io
import json
import os
import subprocess
import sys
import tempfile

from datetime import date, datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import httpx

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt

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

PYTHON_EXERCISES = (
    ("p01BFS.py", "Práctica 1 · Búsqueda BFS"),
    ("p02analisisdedatos_a.py", "Práctica 2A · Crear datos"),
    ("p02explorardatosia_b.py", "Práctica 2B · Explorar datos"),
    ("p02problemascalidaddatos_c.py", "Práctica 2C · Calidad de datos"),
    ("p02limpiezaguardadodatos_d.py", "Práctica 2D · Limpiar datos"),
    ("p02identificacionvariables_e.py", "Práctica 2E · Variables"),
    ("p03primermodelo.py", "Práctica 3 · Datos para clasificación"),
)


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


@app.get("/properties/{identifier}/contract.docx")
def download_contract(identifier: str, user=Depends(authorized)):
    property_row = next(
        (
            row
            for row in visible_properties(user)
            if row["id"] == identifier
        ),
        None,
    )
    if not property_row or not property_row.get("contract_active"):
        raise HTTPException(404, "Contrato no encontrado")

    contract = property_row.get("contract") or {}
    tenant = contract.get("tenant") or "Nombre por completar"
    address = property_row.get("address") or "Domicilio por completar"
    rent_cents = property_row.get("monthly_rent_cents")
    rent = (
        f"${rent_cents / 100:,.2f} MXN"
        if rent_cents
        else "Importe por acordar"
    )
    start = contract.get("start") or "Fecha de inicio por acordar"
    end = contract.get("end") or "Fecha de término por acordar"

    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.7)
    section.bottom_margin = Inches(0.7)
    document.styles["Normal"].font.name = "Arial"
    document.styles["Normal"].font.size = Pt(10)

    title = document.add_heading("Contrato de arrendamiento", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle = document.add_paragraph(
        "Borrador para completar y revisar antes de firmar"
    )
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

    document.add_paragraph(
        "En ______________________________, a ____ de __________________ de ______, "
        "comparecen por una parte el propietario o arrendador identificado abajo y, "
        "por otra, el arrendatario, quienes acuerdan celebrar el presente contrato "
        "respecto del inmueble y conforme a las cláusulas siguientes."
    )
    document.add_heading("Datos de las partes y del inmueble", level=1)
    for label, value in (
        ("Arrendador", "Nombre completo: ______________________________ · Identificación: ______________________________"),
        ("Arrendatario", f"{tenant} · Identificación: ______________________________"),
        ("Inmueble", property_row.get("name") or "Nombre por completar"),
        ("Domicilio", address),
        ("Renta mensual registrada", rent),
        ("Vigencia registrada", f"{start} al {end}"),
    ):
        paragraph = document.add_paragraph()
        paragraph.add_run(f"{label}: ").bold = True
        paragraph.add_run(value)

    clauses = (
        (
            "Primera. Objeto",
            "El arrendador concede al arrendatario el uso temporal del inmueble "
            "identificado en este documento, y el arrendatario acepta recibirlo en "
            "las condiciones que ambas partes hagan constar por escrito al momento "
            "de la entrega.",
        ),
        (
            "Segunda. Plazo",
            f"El plazo será del {start} al {end}. Toda renovación, prórroga o "
            "modificación deberá acordarse por escrito por ambas partes y sujetarse "
            "a la legislación aplicable.",
        ),
        (
            "Tercera. Renta y forma de pago",
            f"La renta mensual será de {rent}, o del importe que las partes "
            "completen y acuerden antes de firmar. Se pagará el día ______ de cada "
            "mes mediante ______________________________, a la cuenta o persona "
            "que indique el arrendador. Las partes conservarán comprobantes de pago.",
        ),
        (
            "Cuarta. Depósito y cargos adicionales",
            "Depósito, servicios, mantenimiento u otros cargos: las partes deberán "
            "indicar por escrito los importes, conceptos, responsables y condiciones "
            "de devolución que acuerden: __________________________________________.",
        ),
        (
            "Quinta. Uso y conservación",
            "El uso autorizado del inmueble será ______________________________. El "
            "arrendatario deberá darle el uso convenido, conservarlo razonablemente "
            "y comunicar oportunamente daños o reparaciones necesarias. Ninguna "
            "modificación material se realizará sin el consentimiento escrito que corresponda.",
        ),
        (
            "Sexta. Obligaciones de las partes",
            "El arrendador entregará el inmueble en la fecha acordada y atenderá las "
            "reparaciones que legalmente le correspondan. El arrendatario pagará la "
            "renta convenida, cuidará el inmueble y permitirá el acceso para "
            "reparaciones o inspecciones previamente acordadas, salvo urgencia y "
            "conforme a la ley.",
        ),
        (
            "Séptima. Terminación y entrega",
            "La terminación, avisos, causas y plazos se regirán por lo pactado por "
            "escrito y por la legislación aplicable. Al concluir la relación, las "
            "partes coordinarán la entrega del inmueble y documentarán su estado y "
            "los pagos pendientes.",
        ),
        (
            "Octava. Avisos y jurisdicción",
            "Domicilio o medio de aviso del arrendador: ______________________________. "
            "Del arrendatario: ______________________________. Para cualquier "
            "controversia, las partes se sujetarán a las autoridades competentes "
            "conforme a la legislación aplicable en ______________________________.",
        ),
    )

    document.add_heading("Declaraciones y cláusulas", level=1)
    document.add_paragraph(
        "Las partes manifiestan que cuentan con capacidad para celebrar este acuerdo, "
        "que revisarán y completarán la información pendiente y que recibirán una "
        "copia firmada del documento."
    )
    for heading, text in clauses:
        document.add_heading(heading, level=2)
        document.add_paragraph(text)

    document.add_paragraph(
        "Leído el presente borrador y enteradas las partes de su contenido, lo firman "
        "en dos ejemplares en el lugar y fecha que completen arriba."
    )
    document.add_heading("Firmas", level=1)
    signatures = document.add_table(rows=2, cols=2)
    signatures.autofit = False
    for cell in signatures.rows[0].cells:
        cell.width = Inches(3)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].add_run("\n\n\n____________________________")
    for cell, label in zip(
        signatures.rows[1].cells,
        (
            "Propietario / arrendador\nNombre y firma",
            f"Cliente / arrendatario\n{tenant}\nNombre y firma",
        ),
    ):
        cell.width = Inches(3)
        cell.paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.paragraphs[0].add_run(label)

    notice = document.add_paragraph()
    notice.paragraph_format.space_before = Pt(16)
    notice.add_run(
        "Borrador generado por RentControl. Completa todos los espacios, verifica "
        "los datos y solicita revisión conforme a la legislación local antes de "
        "firmar. Este formato general no constituye asesoría legal ni sustituye "
        "un contrato revisado por un profesional."
    ).italic = True

    output = io.BytesIO()
    document.save(output)
    output.seek(0)
    return StreamingResponse(
        output,
        media_type=(
            "application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document"
        ),
        headers={
            "Content-Disposition": (
                'attachment; filename="contrato-arrendamiento.docx"'
            ),
            "Cache-Control": "no-store",
        },
    )


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


@app.get("/python-exercises")
def run_python_exercises(user=Depends(authorized)):
    current = today()
    records = []

    for index, property_row in enumerate(
        visible_properties(user),
        start=1,
    ):
        charges = property_row.get("charges", [])
        charged_cents = sum(
            charge["amount_cents"]
            for charge in charges
        )
        paid_cents = sum(
            charge["paid_cents"]
            for charge in charges
        )
        overdue_days = []

        for charge in charges:
            due = date.fromisoformat(charge["due"])
            if (
                charge["amount_cents"] > charge["paid_cents"]
                and due < current
            ):
                overdue_days.append((current - due).days)

        monthly_rent_cents = property_row.get(
            "monthly_rent_cents"
        )
        if monthly_rent_cents is None:
            monthly_rent_cents = (
                charges[0]["amount_cents"] if charges else 0
            )

        records.append({
            "inquilino": f"Registro {index}",
            "tipo_inmueble": "No especificado",
            "renta_mensual": round(monthly_rent_cents / 100),
            "atrasos_previos": len(overdue_days),
            "porcentaje_pagado": (
                round(paid_cents * 100 / charged_cents)
                if charged_cents
                else 100
            ),
            "dias_atraso": max(overdue_days, default=0),
            "mora": int(bool(overdue_days)),
        })

    scripts_dir = Path(__file__).resolve().parents[1] / "scripts"

    with tempfile.TemporaryDirectory(
        prefix="rentcontrol-exercises-"
    ) as working_directory:
        Path(
            working_directory,
            "rentcontrol_data.json",
        ).write_text(
            json.dumps(records, ensure_ascii=False),
            encoding="utf-8",
        )

        scripts = []
        for filename, title in PYTHON_EXERCISES:
            script = scripts_dir / filename
            if not script.is_file():
                raise HTTPException(
                    500,
                    f"No se encontró el ejercicio {filename}.",
                )
            scripts.append((str(script), title))

        runner = """
import contextlib
import io
import json
import runpy
import sys
from pathlib import Path

results = []
for filename, title in json.loads(sys.argv[1]):
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        runpy.run_path(filename, run_name="__main__")
    results.append({
        "id": Path(filename).name,
        "title": title,
        "output": output.getvalue().strip(),
    })
print(json.dumps(results))
"""

        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    runner,
                    json.dumps(scripts),
                ],
                cwd=working_directory,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise HTTPException(
                504,
                "Las prácticas excedieron el tiempo límite.",
            ) from exc

        if completed.returncode != 0:
            details = (
                completed.stderr or completed.stdout
            ).strip()[-2000:]
            raise HTTPException(
                500,
                f"No se pudieron ejecutar las prácticas: {details}",
            )

        try:
            results = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise HTTPException(
                500,
                "No se pudieron leer los resultados de las prácticas.",
            ) from exc

        for result in results:
            if len(result["output"]) > 20000:
                result["output"] = (
                    result["output"][:20000]
                    + "\n… salida truncada …"
                )
            elif not result["output"]:
                result["output"] = "El ejercicio terminó sin salida."

    return {"results": results}


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
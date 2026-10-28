import hashlib
import hmac
import secrets
import sqlite3
import time

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .property_store import connection

router = APIRouter(prefix="/auth")


def database():
    db = connection()

    db.execute("""
        CREATE TABLE IF NOT EXISTS account (
            id INTEGER PRIMARY KEY CHECK(id = 1),
            name TEXT,
            email TEXT UNIQUE,
            salt TEXT,
            password TEXT
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            token TEXT PRIMARY KEY,
            expires INTEGER
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            salt TEXT,
            password TEXT,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user'))
        )
    """)

    columns = {
        row[1]
        for row in db.execute("PRAGMA table_info(sessions)")
    }

    if "user_id" not in columns:
        db.execute("""
            ALTER TABLE sessions
            ADD COLUMN user_id INTEGER NOT NULL DEFAULT 1
        """)

    # Conserva la cuenta anterior como administradora.
    previous = db.execute("""
        SELECT id, name, email, salt, password
        FROM account
        WHERE id = 1
    """).fetchone()

    if previous:
        db.execute("""
            INSERT OR IGNORE INTO users
                (id, name, email, salt, password, role)
            VALUES (?, ?, ?, ?, ?, 'admin')
        """, previous)

    db.commit()
    return db


class Credentials(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,
        extra="forbid",
    )

    email: str = Field(min_length=5, max_length=200)
    password: str = Field(min_length=10, max_length=128)


class Setup(Credentials):
    name: str = Field(min_length=2, max_length=100)


def digest(password, salt):
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode(),
        bytes.fromhex(salt),
        600000,
    ).hex()


def identify(authorization: str):
    token = (
        authorization[7:]
        if authorization.startswith("Bearer ")
        else ""
    )

    token_hash = hashlib.sha256(token.encode()).hexdigest()
    db = database()

    try:
        session = db.execute("""
            SELECT expires, user_id
            FROM sessions
            WHERE token = ?
        """, (token_hash,)).fetchone()

        if not token or not session or session[0] <= time.time():
            raise HTTPException(
                status_code=401,
                detail="Tu sesión terminó. Inicia sesión nuevamente.",
            )

        user = db.execute("""
            SELECT id, name, email, role
            FROM users
            WHERE id = ?
        """, (session[1],)).fetchone()

        if not user:
            raise HTTPException(401, "Cuenta no disponible")

        return {
            "id": user[0],
            "name": user[1],
            "email": user[2],
            "role": user[3],
        }
    finally:
        db.close()


@router.get("/status")
def status():
    db = database()

    try:
        count = db.execute("""
            SELECT COUNT(*)
            FROM users
            WHERE role = 'admin'
        """).fetchone()[0]

        return {"setup_required": count == 0}
    finally:
        db.close()


@router.post("/setup", status_code=201)
def setup(data: Setup):
    if "@" not in data.email:
        raise HTTPException(422, "Escribe un correo válido")

    salt = secrets.token_hex(16)
    password_hash = digest(data.password, salt)
    db = database()

    try:
        with db:
            db.execute("BEGIN IMMEDIATE")

            exists = db.execute("""
                SELECT 1
                FROM users
                WHERE role = 'admin'
            """).fetchone()

            if exists:
                raise HTTPException(
                    409,
                    "La cuenta administradora ya existe. Inicia sesión.",
                )

            values = (
                data.name,
                data.email.lower(),
                salt,
                password_hash,
            )

            db.execute("""
                INSERT INTO account
                VALUES (1, ?, ?, ?, ?)
            """, values)

            db.execute("""
                INSERT INTO users
                    (id, name, email, salt, password, role)
                VALUES (1, ?, ?, ?, ?, 'admin')
            """, values)

        return {
            "message": "Cuenta administradora creada. Ya puedes iniciar sesión."
        }
    finally:
        db.close()


@router.post("/register", status_code=201)
def register(data: Setup):
    if "@" not in data.email:
        raise HTTPException(422, "Escribe un correo válido")

    salt = secrets.token_hex(16)
    password_hash = digest(data.password, salt)
    db = database()

    try:
        with db:
            db.execute("BEGIN IMMEDIATE")

            admin_exists = db.execute("""
                SELECT 1
                FROM users
                WHERE role = 'admin'
            """).fetchone()

            if not admin_exists:
                raise HTTPException(
                    409,
                    "Primero debe configurarse la cuenta administradora.",
                )

            # El registro público siempre crea usuarios, nunca administradores.
            db.execute("""
                INSERT INTO users
                    (name, email, salt, password, role)
                VALUES (?, ?, ?, ?, 'user')
            """, (
                data.name,
                data.email.lower(),
                salt,
                password_hash,
            ))

        return {
            "message": (
                "Cuenta de usuario creada. "
                "El administrador debe asignarte un inmueble."
            )
        }
    except sqlite3.IntegrityError:
        raise HTTPException(
            409,
            "Ya existe una cuenta con ese correo.",
        )
    finally:
        db.close()


@router.post("/login")
def login(data: Credentials):
    db = database()

    try:
        user = db.execute("""
            SELECT salt, password, id, role
            FROM users
            WHERE email = ?
        """, (data.email.lower(),)).fetchone()

        salt = user[0] if user else "00" * 16
        candidate = digest(data.password, salt)

        if not user or not hmac.compare_digest(candidate, user[1]):
            raise HTTPException(
                401,
                "Correo o contraseña incorrectos.",
            )

        token = secrets.token_urlsafe(40)
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        expires = int(time.time()) + 28800

        with db:
            db.execute("""
                DELETE FROM sessions
                WHERE expires <= ?
            """, (int(time.time()),))

            db.execute("""
                INSERT INTO sessions (token, expires, user_id)
                VALUES (?, ?, ?)
            """, (token_hash, expires, user[2]))

        return {
            "access_token": token,
            "expires_in": 28800,
            "role": user[3],
        }
    finally:
        db.close()


@router.get("/me")
def me(authorization: str = Header(default="")):
    return identify(authorization)


@router.post("/logout")
def logout(authorization: str = Header(default="")):
    identify(authorization)

    token_hash = hashlib.sha256(
        authorization[7:].encode()
    ).hexdigest()

    db = database()

    try:
        with db:
            db.execute("""
                DELETE FROM sessions
                WHERE token = ?
            """, (token_hash,))

        return {"message": "Sesión cerrada"}
    finally:
        db.close()


def tenants():
    db = database()

    try:
        rows = db.execute("""
            SELECT id, name, email
            FROM users
            WHERE role = 'user'
            ORDER BY name
        """)

        return [
            {"id": row[0], "name": row[1], "email": row[2]}
            for row in rows
        ]
    finally:
        db.close()
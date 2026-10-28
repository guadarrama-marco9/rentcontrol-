"""Propiedades nuevas del modo demo; SQLite conserva los registros entre reinicios."""
import json
import os
import sqlite3
from pathlib import Path


def connection():
    path = Path(os.getenv('DEMO_DB_PATH', str(Path(__file__).resolve().parents[1] / 'data' / 'demo.sqlite3')))
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS properties (id TEXT PRIMARY KEY, document TEXT NOT NULL)')
    return db


def list_created():
    db = connection()
    try:
        return [json.loads(row[0]) for row in db.execute('SELECT document FROM properties ORDER BY rowid')]
    finally:
        db.close()


def insert(document):
    db = connection()
    try:
        with db:
            db.execute('INSERT INTO properties VALUES (?, ?)', (document['id'], json.dumps(document, ensure_ascii=False)))
    finally:
        db.close()


def initialize(documents):
    db = connection()
    try:
        with db:
            for doc in documents:
                db.execute('INSERT OR IGNORE INTO properties VALUES (?, ?)', (doc['id'], json.dumps(doc)))
    finally:
        db.close()


def mutate(identifier, operation):
    db = connection()
    try:
        with db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT document FROM properties WHERE id=?', (identifier,)).fetchone()
            if not row:
                from fastapi import HTTPException
                raise HTTPException(404, 'Propiedad no encontrada')
            document = json.loads(row[0])
            result = operation(document)
            db.execute('UPDATE properties SET document=? WHERE id=?', (json.dumps(document), identifier))
            return result
    finally:
        db.close()

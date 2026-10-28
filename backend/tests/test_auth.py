import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

        self.env = patch.dict(
            os.environ,
            {
                "AUTH_REQUIRED": "1",
                "STORAGE": "demo",
                "RENTCONTROL_API_KEY": "",
                "DEMO_DB_PATH": os.path.join(
                    self.tmp.name,
                    "local.db",
                ),
            },
        )

        self.env.start()
        self.client = TestClient(app)

    def tearDown(self):
        self.client.close()
        self.env.stop()
        self.tmp.cleanup()

    def test_setup_login_protection_logout(self):
        status = self.client.get("/auth/status")
        self.assertTrue(status.json()["setup_required"])

        self.assertEqual(
            self.client.get("/dashboard").status_code,
            401,
        )

        credentials = {
            "email": "admin@example.com",
            "password": "ClaveSegura2026",
        }

        response = self.client.post(
            "/auth/setup",
            json={
                **credentials,
                "name": "Admin local",
            },
        )
        self.assertEqual(response.status_code, 201)

        status = self.client.get("/auth/status")
        self.assertFalse(status.json()["setup_required"])

        response = self.client.post(
            "/auth/setup",
            json={
                **credentials,
                "name": "Otra cuenta",
            },
        )
        self.assertEqual(response.status_code, 409)

        response = self.client.post(
            "/auth/login",
            json={
                **credentials,
                "password": "incorrecta123",
            },
        )
        self.assertEqual(response.status_code, 401)

        response = self.client.post(
            "/auth/login",
            json=credentials,
        )
        self.assertEqual(response.status_code, 200)

        token = response.json()["access_token"]
        headers = {"Authorization": "Bearer " + token}

        response = self.client.get(
            "/auth/me",
            headers=headers,
        )
        self.assertEqual(
            response.json()["name"],
            "Admin local",
        )

        for path in (
            "/dashboard",
            "/properties",
            "/alerts",
            "/alerts.csv",
            "/lab",
            "/logic",
        ):
            self.assertEqual(
                self.client.get(
                    path,
                    headers=headers,
                ).status_code,
                200,
            )

            self.assertEqual(
                self.client.get(path).status_code,
                401,
            )

        response = self.client.post(
            "/auth/logout",
            headers=headers,
        )
        self.assertEqual(response.status_code, 200)

        self.assertEqual(
            self.client.get(
                "/dashboard",
                headers=headers,
            ).status_code,
            401,
        )

    def test_password_and_session_not_stored_in_cleartext(self):
        credentials = {
            "email": "admin@example.com",
            "password": "ClaveSegura2026",
        }

        response = self.client.post(
            "/auth/setup",
            json={
                **credentials,
                "name": "Admin local",
            },
        )
        self.assertEqual(response.status_code, 201)

        response = self.client.post(
            "/auth/login",
            json=credentials,
        )
        self.assertEqual(response.status_code, 200)

        token = response.json()["access_token"]

        db = sqlite3.connect(
            os.environ["DEMO_DB_PATH"]
        )

        try:
            stored_password = db.execute(
                "SELECT password FROM account"
            ).fetchone()[0]

            self.assertNotEqual(
                stored_password,
                credentials["password"],
            )

            stored_token = db.execute(
                "SELECT token FROM sessions"
            ).fetchone()[0]

            self.assertNotEqual(
                stored_token,
                token,
            )

            db.execute(
                "UPDATE sessions SET expires = 0"
            )
            db.commit()

        finally:
            db.close()

        response = self.client.get(
            "/dashboard",
            headers={
                "Authorization": "Bearer " + token,
            },
        )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
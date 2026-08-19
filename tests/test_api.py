import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from src.api.main import app

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "HEALTHY")

    def test_metrics_endpoint(self):
        response = self.client.get("/metrics")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn("system", json_data)

if __name__ == "__main__":
    unittest.main()

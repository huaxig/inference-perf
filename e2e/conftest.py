import time

import pytest
import requests


@pytest.fixture(scope="session")
def vllm_sim_server():
    """
    Waits for the vllm-d-inference-sim server to be ready.
    The server is expected to be started as a service in the CI workflow.
    """
    # In github actions, the service is available at localhost:<host_port>
    health_url = "http://localhost:8000/health"
    for i in range(30):
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                print(f"vllm-sim-server is ready at attempt {i+1}")
                yield
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(1)
    else:
        pytest.fail("Failed to connect to vllm-d-inference-sim server")

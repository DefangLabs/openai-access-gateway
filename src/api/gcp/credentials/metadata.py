import logging
import requests

from google.auth import default
from google.auth.transport.requests import Request as AuthRequest

from api.setting import GCP_PROJECT_ID, GCP_REGION


# GCP credentials and project details
credentials = None
project_id = None
location = None

def get_gcp_project_details():
    from google.auth import default

    # Try metadata server for region
    credentials = None
    project_id = GCP_PROJECT_ID
    location = GCP_REGION

    try:
        credentials, project = default()
        if not project_id:
            project_id = project

        if not location:
            zone = requests.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/zone",
                headers={"Metadata-Flavor": "Google"},
                timeout=1
            ).text
            location = zone.split("/")[-1].rsplit("-", 1)[0]

    except Exception:
        logging.warning(f"Error: Failed to get project and location from metadata server. Using local settings.")

    return credentials, project_id, location

credentials, project_id, location = get_gcp_project_details()

# Utility: get service account access token
def get_access_token():
    credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    auth_request = AuthRequest()
    credentials.refresh(auth_request)
    return credentials.token

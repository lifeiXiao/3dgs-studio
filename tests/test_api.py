import json
from fastapi.testclient import TestClient
from studio.server import app

client=TestClient(app)


def test_project_directory_is_confined():
    assert client.get('/api/browse',params={'path':'../'}).status_code==400
    assert client.get('/api/browse',params={'path':'.venv'}).status_code==400
    assert client.get('/api/browse').status_code==200


def test_local_origin_rejects_foreign_mutations():
    response=client.post('/api/datasets/inspect',headers={'Origin':'https://other-site.example'},json={'path':'.'})
    assert response.status_code==403


def test_upload_cannot_escape_dataset_folder():
    response=client.post('/api/datasets/upload',data={'paths':json.dumps(['../escape.png'])},files=[('files',('escape.png',b'x','image/png'))])
    assert response.status_code==400
    assert '路径无效' in response.json()['detail']


def test_training_parameter_validation():
    response=client.post('/api/runs',json={'dataset':'data/datasets/demo-orbit','gaussians':2000001})
    assert response.status_code==422

# insight-stream
An Insight bot for streaming YouTube videos


### Version 1 - insight-stream-0.1
> Local Streamlit App

### Version 2 - insight-stream-1.0
> FastAPI Query Engine

- FastAPI Endpoints Deployment
- Platform: Google Cloud
- Product: Cloud Run

Build the Docker image

```
gcloud builds submit --tag gcr.io/<project_id>/<project_name>
```

Deploy the Docker image

```
gcloud run deploy <project_service_name> --image gcr.io/<project_id>/<project_name> --allow-unauthenticated --env-vars-file <path_to_secrets_file>
```


### Version 2.1 - insight-stream-1.1
> FastAPI Chat Engine


### Version 3 - insight-stream-2.0
> Streaming Videos

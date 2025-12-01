Docker build and run instructions for this project

Build the image (tag it `cs456-project-2`):

```bash
docker build -t cs456-project-2 .
```

Run the container (expose port 8501):

```bash
docker run --rm -p 8501:8501 \
  -e GOOGLE_API_KEY="$GOOGLE_API_KEY" \
  -v "$(pwd)/chroma_langchain_db:/home/appuser/app/chroma_langchain_db" \
  cs456-project-2
```

Notes:
- The app uses Streamlit and by default listens on port 8501.
- If you rely on the local Chroma DB directory (`chroma_langchain_db`), bind-mount it to preserve persistence.
- Set your Google API key or other secrets via environment variables at runtime; do not bake them into the image.
- If some packages in `requirements.txt` need system libs (eg. for vector DBs or optional libs), install them on the base image as necessary. The Dockerfile includes a small set of build tools to help with that.

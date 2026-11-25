FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY mdp ./mdp
COPY results ./results
RUN pip install --no-cache-dir -e ".[api]"
EXPOSE 8000
CMD ["uvicorn", "mdp.api:app", "--host", "0.0.0.0", "--port", "8000"]

FROM public.ecr.aws/lambda/python:3.10

# Install system deps
RUN yum install -y git gcc g++ make

# Copy requirements
COPY requirements.txt .

# Install deps (with faiss fix)
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt 
RUN pip install --no-cache-dir faiss-cpu==1.7.4

# Copy app code
COPY query_agent.py ${LAMBDA_TASK_ROOT}

CMD ["query_agent.lambda_handler"]

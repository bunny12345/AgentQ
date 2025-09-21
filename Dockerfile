FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum install -y git gcc g++ make

# Copy requirements
COPY requirements.txt .

# Install Python deps
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt --target "${LAMBDA_TASK_ROOT}"
pip install -qU langchain-community faiss-cpu
# Copy app code
COPY agent.py ${LAMBDA_TASK_ROOT}

# Set the Lambda handler
CMD ["agent.lambda_handler"]

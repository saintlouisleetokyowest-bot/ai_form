# TO DO:
# 1. Create api and two endpoints
# 2. Check fastapi
# 3. What comes in and out?
# 4. What will we return in the endpoint?


#IMPORTS
from fastapi import FastAPI


#CODE

app = FastAPI()

# Define a root `/` endpoint
@app.get('/')
def index():
    return {'Live tracker': True} #MediaPipe

@app.get('/ghost')
def index():
    return {'Real ghost': True} #Ghost

#RUN CODE IN TERMINAL

#The following creates the server and outputs whatever needs to be returned
#uvicorn api_endpoints:app --reload

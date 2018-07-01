import requests
from glob import glob

API_URL = "http://localhost:5000/predict"


files = glob('*.jpeg')
for file in files:
    image = open(file, "rb").read()
    payload = {"image": image}
    response = requests.post(API_URL, files=payload).json()
    if response["success"]:
        print("{0} {1}".format(file, "is safe for work" if bool(response["is_safe"]) else "is not safe for work"))
    else:
        print("Request failed")

import json
import pandas as pd
import requests

cenote_base_url = "https://cenote.sidero.services/"
project_id = "pid7d2b5163511"
write_key = "undefined"
read_key = "undefined"

def read_data(collection, latest):

    # Read data
    r = requests.get(
      url = cenote_base_url + "api/projects/" + project_id + "/queries/extraction/?readKey=" + read_key + "&event_collection=" + collection + "&latest=" + str(latest)
    )

    return json.loads(r.text)

def insert_data(event_collection):

    data = {
      "payload": [
        {
          "data": {
            "globalactivepower": 1.25,
            "globalreactivepower": 3.36,
            "voltage": 240.12,
            "globalintensity": 3.1
          }
        },
        {
          "data": {
            "globalactivepower": 1.3,
            "globalreactivepower": 2.41,
            "voltage": 238.72,
            "globalintensity": 9.1
          }
        }
      ]
    }

    r = requests.post(
      url = cenote_base_url + "api/projects/" + project_id + "/events/" + event_collection + "?writeKey=" + write_key,
      json = data
    )

    print(r.text)


# Store data into Cenote
insert_data("testcollection")

# Read already imported data
data = read_data("consumption", 50)
print(json.dumps(data, indent=3))

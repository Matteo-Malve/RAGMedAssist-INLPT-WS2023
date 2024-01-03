## To convert .csv to .json
    python -c "import pandas as pd; pd.read_csv('pubmed_data.csv').to_json('pubmed_data.json', orient='records')


## To run/test/stop docker cluster
Must have Docker Desktop installed. No more installations are required, the images are downloaded automatically.

Make sure to run the folloqing code in the same directory with the docker-compose.yml file.

    docker-compose up -d
    docker-compose ps
    docker-compose down

## To create index in OpenSearch
In the docker-compose file we specified port 5601 for Opensearch Dashboard.

In any browser, go to localhost:5601 to access the Dashboard.

From there, go to the console.

    PUT /pubmed_data
    {
      "mappings":
      {
        "properties": {
          "PMID": {
            "type": "integer"
          },
          "Title": {
            "type": "text"
          },
          "Abstract": {
            "type": "text"
          },
          "Authors": {
            "type": "keyword"
          },
          "Publication_Date": {
            "type": "text"
          },
          "DOI": {
            "type": "keyword"
          }
        }
      }
    }

## To load the json to OpenSearch
The dashboard is on port 5601, but the actual port of OpenSearch is 9200.

The following should be enough to upload the json file. I believe it should not work since the json file is not structured properly in the syntax.
But try with one of the example jsons given at exercise session. They should work. I got an error in response of the server itself...

    curl -X POST "localhost:9200/pubmed_data/_doc/_bulk" -H 'Content-Type: application/json' --data-binary @pubmed_data.json

If you get asked for permission, uppend
    -u username:password

which by default should be admin:admin.

THis last one was a test to see if I get response from the port. Not sure of how it works.

    curl –u admin:admin -X GET http://127.0.0.1:9200

import pymongo

MONGO_URI = "mongodb://root:9AsYmXYKmYLHcNsShmCb3L5DZMXH77rQ9GBRxm0HKownNWLwdzH9dW7zhPG9mpuR@46.4.101.229:8281/?directConnection=true"
COLLECTION_NAME = "tts_data"

client = pymongo.MongoClient(MONGO_URI)
db = client["tts_data"]
collection = db[COLLECTION_NAME]

# Get all documents from the collection that does not have a "transcription" field
documents = collection.find({"transcription": {"$exists": False}})

for document in documents:
    print(document)
    break

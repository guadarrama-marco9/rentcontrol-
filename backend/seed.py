"""Carga explícita de datos ficticios. No borra colecciones ni sobrescribe registros."""
import os
from datetime import date
from pymongo import MongoClient
from app.demo import dataset

if __name__=='__main__':
    with MongoClient(os.environ['MONGO_URI']) as client:
        collection=client[os.getenv('MONGO_DB','rentcontrol')].properties
        collection.create_index('id',unique=True)
        for p in dataset(date.today()):
            collection.update_one({'id':p['id']},{'$setOnInsert':p},upsert=True)
        print('Datos ficticios insertados; registros existentes conservados.')

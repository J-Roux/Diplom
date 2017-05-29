import pymongo
import json


class DatabaseModule:
    conn = None

    def __init__(self, ip_address, port):
        self.conn = pymongo.MongoClient(ip_address, port)
        self.conn = self.conn.mydb

    def track_model_to_json(self, track):
        return {'label': json.dumps(track.label),
                'feature': json.dumps(track.to_vector().tolist())}

    def store(self, track):
        json = self.track_model_to_json(track)
        print json
        self.conn.tracks.save(json)

import os
from redis import Redis
from rq import Connection, Worker

listen = ['default']
redis_url = os.getenv('REDIS_URL', 'redis://redis:6379/0')

if __name__ == '__main__':
    conn = Redis.from_url(redis_url)
    with Connection(conn):
        worker = Worker(listen)
        worker.work()

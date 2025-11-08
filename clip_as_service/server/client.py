from clip_client import Client
from multiprocessing import Pool
import time

c = Client("grpc://0.0.0.0:51000")
# for i in range(1000):
#     stat = c.profile()               # 单次往返
#     rt_ms = stat['Roundtrip']        # 例如 15.7 ms
#     qps = 1000 / rt_ms               # ≈ 63.7 QPS
#     print(f'~{qps:.1f} req/s')
# raise
im = 'data:image/gif;base64,R0lGODlhEAAQAMQAAORHHOVSKudfOulrSOp3WOyDZu6QdvCchPGolfO0o/XBs/fNwfjZ0frl3/zy7////wAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACH5BAkAABAALAAAAAAQABAAAAVVICSOZGlCQAosJ6mu7fiyZeKqNKToQGDsM8hBADgUXoGAiqhSvp5QAnQKGIgUhwFUYLCVDFCrKUE1lBavAViFIDlTImbKC5Gm2hB0SlBCBMQiB0UjIQA7'
r = c.encode(['First do it',])

# print(r.shape)  # [3, 512] 
# batch_data = [[im]*32] * 64


# def batch_call():
#     pool = Pool(16)
#     h = pool.map(c.encode, batch_data)
#     for i in h:
#         print(i.shape)
#     pool.close()
#     pool.join()

# t1 = time.time()
# batch_call()
# t2 = time.time()
# print(t2 - t1, len(batch_data) * len(batch_data[0]) / (t2 - t1))